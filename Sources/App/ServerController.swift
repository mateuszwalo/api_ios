import Foundation
import UIKit
import LlamaBridge

/// Everything the screens bind to: the model, the server, the queue and the log.
@MainActor
@Observable
final class ServerController {

    // MARK: Settings

    var port: UInt16 = 8080
    var contextLength = 32768
    var batchSize = 512
    var threadCount = 0                    // 0 = as many as there are cores
    var flashAttention = true
    var useMemoryMapping = true
    /// Off by default so every prefill is measured cold. See docs/DECISIONS.md.
    var reuseKVCache = false
    var defaultMaxTokens = 4096

    // MARK: State

    private(set) var serverState: LocalHTTPServer.State = .stopped
    private(set) var isLoadingModel = false
    private(set) var loadError: String?
    private(set) var selectedPair: ModelPair?
    private(set) var startedAt: Date?

    let log = RequestLog()
    let store = ModelStore()
    let downloader: ModelDownloader

    private let engine = LlamaInferenceEngine()
    private let queue = InferenceQueue()
    private let grammar = GrammarCompiler()
    private var server: LocalHTTPServer?
    private var memoryPressureSource: DispatchSourceMemoryPressure?

    // Mirrors of actor state, refreshed on a timer so SwiftUI has something synchronous to
    // read. The actors remain the source of truth; these are for display only.
    private(set) var queueDepth = 0
    private(set) var footprintBytes: UInt64 = 0
    private(set) var availableBytes: UInt64 = 0
    private(set) var thermalState = MemoryProbe.thermalStateName()
    private var ticker: Task<Void, Never>?

    init() {
        downloader = ModelDownloader(destinationDirectory: store.directory)
        observeMemoryPressure()
        startTicker()
    }

    var isRunning: Bool {
        if case .running = serverState { return true }
        return false
    }

    var isModelLoaded: Bool { selectedPair != nil && loadError == nil && !isLoadingModel }

    /// The address to paste into a client, one line, ready to copy.
    var baseURLs: [String] {
        guard case .running(let port) = serverState else { return [] }
        return NetworkInterfaces.localIPv4().map { "http://\($0.ip):\(port)/v1" }
    }

    // MARK: Model

    func load(_ pair: ModelPair) async {
        isLoadingModel = true
        loadError = nil
        defer { isLoadingModel = false }

        let options = LLMLoadOptions.defaults()
        options.contextLength = contextLength
        options.batchSize = batchSize
        options.microBatchSize = batchSize
        options.threadCount = threadCount
        options.flashAttention = flashAttention
        options.useMemoryMapping = useMemoryMapping
        options.reuseKVCacheBetweenRequests = reuseKVCache

        do {
            try await engine.load(.init(name: pair.name,
                                        modelPath: pair.model.url,
                                        projectorPath: pair.projector?.url),
                                  options: options)
            selectedPair = pair
        } catch {
            selectedPair = nil
            loadError = describeLoadFailure(error)
        }
    }

    func unloadModel() async {
        await engine.unload()
        selectedPair = nil
    }

    /// Out of memory is the expected failure on this hardware, so it gets a message that
    /// says what to do rather than the allocator's.
    private func describeLoadFailure(_ error: Error) -> String {
        let text = (error as NSError).localizedDescription
        guard MemoryProbe.isUnderPressure() || text.lowercased().contains("memory") else {
            return text
        }
        // The numbers, not just the verdict: whoever reads this has to decide what to lower,
        // and "out of memory" on its own says neither by how much nor which setting to touch.
        return """
        \(text)

        Available to this process: \(MemoryProbe.format(MemoryProbe.availableBytes())). \
        Tried to load at context \(contextLength) with batch \(batchSize).

        Both are worth lowering, in the Server tab, before loading again — 8192 and 128 are \
        a reasonable first retry. A 4B model at 32k context does not fit on an 8 GB iPad, and \
        the increased-memory-limit entitlement does not survive signing with a free Apple ID.
        """
    }

    // MARK: Server

    func startServer() {
        let router = APIRouter(
            engine: engine,
            queue: queue,
            grammar: grammar,
            defaultMaxTokens: defaultMaxTokens,
            record: { [log] entry in await MainActor.run { log.record(entry) } },
            stats: { [weak self] in await self?.snapshot() ?? StatsSnapshot.empty },
            logExport: { [log] in await MainActor.run { log.exportText() } },
            runSelfTest: { [engine, grammar] in await SelfTest(engine: engine, grammar: grammar).run() }
        )

        let server = LocalHTTPServer(
            handler: { request in await router.route(request) },
            onStateChange: { [weak self] state in
                Task { @MainActor in self?.apply(state) }
            })
        self.server = server
        server.start(port: port)
        startedAt = Date()

        // The screen must stay awake: the server only runs in the foreground, and a device
        // that locks itself after four minutes would end a twenty-minute run from the
        // inside.
        UIApplication.shared.isIdleTimerDisabled = true
    }

    func stopServer() {
        server?.stop()
        server = nil
        startedAt = nil
        UIApplication.shared.isIdleTimerDisabled = false
    }

    private func apply(_ state: LocalHTTPServer.State) {
        serverState = state
        if case .failed = state { UIApplication.shared.isIdleTimerDisabled = false }
    }

    // MARK: Stats

    func snapshot() async -> StatsSnapshot {
        StatsSnapshot(
            modelLoaded: await engine.isLoaded,
            model: await engine.loadedModelName,
            contextLength: await engine.contextLength,
            supportsImages: await engine.supportsImages,
            queueDepth: await queue.depth,
            served: log.servedCount,
            failed: log.failedCount,
            footprintBytes: MemoryProbe.footprintBytes(),
            peakFootprintBytes: log.peakFootprintBytes,
            availableMemoryBytes: MemoryProbe.availableBytes(),
            thermalState: MemoryProbe.thermalStateName(),
            uptimeSeconds: startedAt.map { Int(Date().timeIntervalSince($0)) } ?? 0,
            lastError: loadError)
    }

    private func startTicker() {
        ticker = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                self.queueDepth = await self.queue.depth
                self.footprintBytes = MemoryProbe.footprintBytes()
                self.availableBytes = MemoryProbe.availableBytes()
                self.thermalState = MemoryProbe.thermalStateName()
                try? await Task.sleep(for: .seconds(1))
            }
        }
    }

    /// Under memory pressure the model is dropped rather than waited on.
    ///
    /// The alternative is not "carry on": it is the process being killed with no log line,
    /// in the middle of a run, leaving whoever is watching to guess. Unloading turns that
    /// into requests answered with 503, which the client retries once a model is back.
    private func observeMemoryPressure() {
        let source = DispatchSource.makeMemoryPressureSource(eventMask: [.critical], queue: .main)
        source.setEventHandler { [weak self] in
            Task { @MainActor in
                guard let self, self.selectedPair != nil else { return }
                await self.unloadModel()
                self.loadError = """
                The system reported critical memory pressure and the model was unloaded to \
                avoid the app being terminated. Requests will be answered with 503 until a \
                model is loaded again.
                """
            }
        }
        source.resume()
        memoryPressureSource = source
    }
}

extension StatsSnapshot {
    static let empty = StatsSnapshot(modelLoaded: false, model: nil, contextLength: 0,
                                     supportsImages: false, queueDepth: 0, served: 0, failed: 0,
                                     footprintBytes: 0, peakFootprintBytes: 0,
                                     availableMemoryBytes: 0, thermalState: "unknown",
                                     uptimeSeconds: 0, lastError: nil)
}
