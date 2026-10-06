import Foundation
import LlamaBridge

/// `InferenceEngine` backed by llama.cpp.
///
/// Every call into the bridge blocks for as long as the work takes — minutes for a long
/// prompt. That work is pushed onto a dedicated thread rather than run on the actor
/// directly: Swift's cooperative pool has one thread per core, and parking one of them for
/// twenty minutes starves everything else in the app, the HTTP accept loop included. The
/// server would stop acknowledging connections precisely while it is busiest.
actor LlamaInferenceEngine: InferenceEngine {

    private let bridge = LLMBridge()
    private let worker = DispatchQueue(label: "llm.inference", qos: .userInitiated)

    private var modelName: String?
    private var options = LLMLoadOptions.defaults()

    // MARK: Loading

    struct LoadedModel: Sendable, Equatable {
        let name: String
        let modelPath: URL
        let projectorPath: URL?
    }

    private(set) var loaded: LoadedModel?

    var isLoaded: Bool { loaded != nil }

    /// The loaded configuration in one line, as it appears in every response's timings.
    /// Compact on purpose: it goes into each result row of a comparison.
    var configurationLabel: String {
        guard let loaded else { return "" }
        return [loaded.modelPath.deletingPathExtension().lastPathComponent,
                loaded.projectorPath == nil ? "text" : "vision",
                "ctx=\(options.contextLength)",
                "batch=\(options.batchSize)",
                "fa=\(options.flashAttention ? "auto" : "off")",
                "kv=\(options.kvCacheType == 1 ? "q8_0" : "f16")",
                "reuse=\(options.reuseKVCacheBetweenRequests ? "on" : "off")",
                "mmap=\(options.useMemoryMapping ? "on" : "off")"].joined(separator: " ")
    }
    var loadedModelName: String? { modelName }
    var contextLength: Int { bridge.contextLength }
    var supportsImages: Bool { bridge.supportsImages }
    var loadOptions: LLMLoadOptions { options }

    func load(_ model: LoadedModel, options: LLMLoadOptions) async throws {
        self.options = options
        LLMBridge.noteEvent("load \(model.modelPath.lastPathComponent)"
                            + " projector=\(model.projectorPath?.lastPathComponent ?? "none")"
                            + " ctx=\(options.contextLength) batch=\(options.batchSize)"
                            + " fa=\(options.flashAttention) mmap=\(options.useMemoryMapping)"
                            + " available=\(MemoryProbe.format(MemoryProbe.availableBytes()))")
        do {
            try await onWorker { [bridge] in
                try bridge.loadModel(atPath: model.modelPath.path,
                                     projectorPath: model.projectorPath?.path,
                                     options: options)
            }
        } catch {
            LLMBridge.noteEvent("load failed: \((error as NSError).localizedDescription)")
            throw error
        }
        LLMBridge.noteEvent("load ok, footprint=\(MemoryProbe.format(MemoryProbe.footprintBytes()))")
        loaded = model
        modelName = model.name
    }

    func unload() async {
        try? await onWorker { [bridge] in bridge.unload() }
        loaded = nil
        modelName = nil
    }

    // MARK: Inference

    func measurePrompt(_ prompt: EnginePrompt) async throws -> Int {
        let turns = Self.turns(from: prompt)
        do {
            return try await onWorker { [bridge] in
                var error: NSError?
                let n = bridge.measurePromptTokens(turns, error: &error)
                if n < 0 { throw error ?? EngineError.backend("tokenisation failed") }
                return n
            }
        } catch let e as NSError where e.domain == LLMBridgeErrorDomain {
            throw Self.translate(e)
        }
    }

    /// The prompt as the model's own chat template renders it. Used by the self-test, where
    /// seeing the text is the only way to settle a template mismatch.
    func renderedPromptText(_ prompt: EnginePrompt) async -> String? {
        let turns = Self.turns(from: prompt)
        return try? await onWorker { [bridge] in bridge.renderedPrompt(for: turns) }
    }

    func generate(_ request: EngineRequest) async throws -> EngineResult {
        let turns = Self.turns(from: request.prompt)
        let opts = LLMGenerationOptions.defaults()
        opts.maxTokens = request.maxTokens
        opts.temperature = request.temperature
        opts.grammar = request.grammar
        opts.stopSequences = request.stopSequences
        opts.seed = request.seed.map { Int64($0) } ?? -1

        let imageCount = request.prompt.turns.reduce(0) { $0 + $1.images.count }
        LLMBridge.noteEvent("generate start: max_tokens=\(request.maxTokens)"
                            + " grammar=\(request.grammar == nil ? "no" : "\(request.grammar!.utf8.count)B")"
                            + " images=\(imageCount) temperature=\(request.temperature)")

        // The flag is read from the inference thread and written from whichever task is
        // cancelled, so it cannot be ordinary mutable state.
        let cancelled = CancellationFlag()

        do {
            var result = try await withTaskCancellationHandler {
                try await onWorker { [bridge] in
                    // Imported as throwing: the Objective-C method returns a nullable object
                    // with an NSError out-parameter, which is the convention Swift folds into
                    // `throws`. The NSError still arrives, as the thrown value.
                    let r = try bridge.generate(with: turns,
                                                options: opts,
                                                isCancelled: { cancelled.isSet })
                    return EngineResult(text: r.text,
                                        promptTokens: r.promptTokens,
                                        completionTokens: r.completionTokens,
                                        prefillMilliseconds: r.prefillMilliseconds,
                                        decodeMilliseconds: r.decodeMilliseconds,
                                        hitTokenLimit: r.hitTokenLimit,
                                        cachedPromptTokens: r.cachedPromptTokens)
                }
            } onCancel: {
                cancelled.set()
            }
            result.configuration = configurationLabel
            LLMBridge.noteEvent("generate done: \(result.promptTokens)->\(result.completionTokens) tok"
                                + " (\(result.cachedPromptTokens) cached),"
                                + " prefill \(result.prefillMilliseconds)ms, decode \(result.decodeMilliseconds)ms")
            return result
        } catch let e as NSError where e.domain == LLMBridgeErrorDomain {
            LLMBridge.noteEvent("generate failed: \(e.localizedDescription)")
            if e.code == LLMBridgeErrorCode.cancelled.rawValue { throw CancellationError() }
            throw Self.translate(e)
        }
    }

    // MARK: Plumbing

    /// Hops to the inference thread and suspends until it is done. `nonisolated` so the
    /// actor is free while the work runs — the serialisation that matters is the queue in
    /// front of the engine, not this actor.
    private nonisolated func onWorker<T: Sendable>(
        _ body: @escaping @Sendable () throws -> T
    ) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            worker.async {
                do { continuation.resume(returning: try body()) }
                catch { continuation.resume(throwing: error) }
            }
        }
    }

    private static func turns(from prompt: EnginePrompt) -> [LLMTurn] {
        prompt.turns.map { turn in
            LLMTurn(role: turn.role,
                    text: turn.text,
                    images: turn.images.map { LLMImage(data: $0.data, mimeType: $0.mimeType) })
        }
    }

    private static func translate(_ error: NSError) -> EngineError {
        let message = error.localizedDescription
        switch LLMBridgeErrorCode(rawValue: error.code) {
        case .notLoaded:        return .notLoaded
        case .contextOverflow:  return .backend(message)
        case .grammarInvalid:   return .grammarRejected(message)
        default:                return .backend(message)
        }
    }
}

/// A one-way flag, settable from any thread.
///
/// `OSAllocatedUnfairLock` rather than an actor: it is read between every pair of tokens,
/// and an `await` in that position would turn a cancellation check into a suspension point
/// inside the hottest loop in the app.
final class CancellationFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false

    var isSet: Bool {
        lock.lock(); defer { lock.unlock() }
        return value
    }

    func set() {
        lock.lock(); value = true; lock.unlock()
    }
}

// MARK: - Grammar compilation

/// Compiles JSON Schema to GBNF, remembering what it has already compiled.
///
/// Conversion is pure and fast, but a batch run sends the same schema dozens of times and
/// the schemas are large; caching keeps that off the critical path. Keyed by the
/// canonicalised schema text, so two requests spelling the same schema differently still
/// share an entry.
final class GrammarCompiler: @unchecked Sendable {
    private let lock = NSLock()
    private var cache: [String: String] = [:]

    enum Failure: Error, CustomStringConvertible {
        case rejected(String)
        var description: String {
            switch self {
            case .rejected(let why): return why
            }
        }
    }

    func compile(_ schemaJSON: String) throws -> String {
        lock.lock()
        if let hit = cache[schemaJSON] {
            lock.unlock()
            return hit
        }
        lock.unlock()

        let grammar: String
        do {
            grammar = try LLMBridge.grammar(fromJSONSchema: schemaJSON)
        } catch {
            throw Failure.rejected((error as NSError).localizedDescription)
        }

        lock.lock()
        cache[schemaJSON] = grammar
        lock.unlock()
        return grammar
    }
}
