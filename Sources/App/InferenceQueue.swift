import Foundation

/// Serialises inference so that exactly one request is decoded at a time, in arrival
/// order.
///
/// Clients batch: a single document can produce a dozen concurrent calls, and more than
/// twenty may be in flight at once. All of them must be accepted — refusing with 503
/// would abort the client's whole run — and all of them must be processed one after
/// another, because there is one model and one context.
///
/// Serialisation is not merely a resource decision. Decoding two requests concurrently
/// changes the order of floating-point reductions, and with greedy sampling that is
/// enough to flip a token when two logits are nearly tied. Measurements taken that way
/// are not reproducible.
///
/// An `actor` alone does not give this. Actors are re-entrant and make no ordering
/// promise about tasks suspended on them, so the twentieth request could well be served
/// before the third. The explicit continuation queue below is what makes arrival order
/// hold.
actor InferenceQueue {
    private var isBusy = false
    /// Resumed with `true` when the slot is handed over, `false` when the waiter was
    /// abandoned after cancellation. Both paths resume the continuation, so the value is
    /// the only way to tell whether this waiter now owns the slot — and releasing a slot
    /// one never held would let two requests decode at once, which is precisely the
    /// failure this type exists to prevent.
    private var waiters: [(id: UUID, continuation: CheckedContinuation<Bool, Never>)] = []

    private(set) var queuedCount = 0
    private(set) var completedCount = 0

    /// Number of requests waiting plus the one in flight. Surfaced in the UI so an
    /// operator can tell a stalled server from a merely busy one.
    var depth: Int { queuedCount + (isBusy ? 1 : 0) }

    /// Runs `work` exclusively. Callers queue in arrival order.
    ///
    /// If the calling task is cancelled while waiting — which is how a client
    /// disconnecting is surfaced — its slot is abandoned and the queue moves on. Without
    /// that, a departed client would still hold its place and every request behind it
    /// would wait for work nobody is going to read.
    func run<T: Sendable>(_ work: @Sendable () async throws -> T) async throws -> T {
        try await acquire()
        defer { release() }
        try Task.checkCancellation()
        let value = try await work()
        completedCount += 1
        return value
    }

    private func acquire() async throws {
        if !isBusy {
            isBusy = true
            return
        }
        let id = UUID()
        queuedCount += 1
        let granted = await withTaskCancellationHandler {
            await withCheckedContinuation { (c: CheckedContinuation<Bool, Never>) in
                waiters.append((id: id, continuation: c))
            }
        } onCancel: {
            Task { await self.abandon(id) }
        }
        queuedCount -= 1
        guard granted else {
            // Abandoned after cancellation: no slot was taken, so none may be released.
            throw CancellationError()
        }
        // Slot held. `run` re-checks cancellation and its `defer` passes the slot on.
    }

    private func release() {
        if waiters.isEmpty {
            isBusy = false
        } else {
            // Front of the queue: arrival order is the whole point. `isBusy` stays true —
            // the slot is handed straight over rather than released and re-taken.
            let next = waiters.removeFirst()
            next.continuation.resume(returning: true)
        }
    }

    /// Resumes and drops a waiter whose task was cancelled, so it does not keep a place
    /// in the queue it can no longer use. Resuming with `false` tells it that it holds no
    /// slot.
    ///
    /// A waiter already handed the slot by `release` is no longer in `waiters`, so the
    /// lookup fails and this is a no-op — the two paths cannot both fire.
    private func abandon(_ id: UUID) {
        guard let idx = waiters.firstIndex(where: { $0.id == id }) else { return }
        let waiter = waiters.remove(at: idx)
        waiter.continuation.resume(returning: false)
    }
}
