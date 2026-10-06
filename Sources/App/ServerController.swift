import Foundation
import UIKit
import LlamaBridge

/// Everything the screens bind to: the model, the server, the queue and the log.
@MainActor
@Observable
final class ServerController {

    // MARK: Settings

    // Persisted on every change and restored at launch. They used to reset to defaults on
    // each start, so a tester who relaunched after a crash silently ran the next test on a
    // different configuration from the one written down.
    var port: UInt16 = 8080 { didSet { persistSettings() } }
    var contextLength = 32768 { didSet { persistSettings() } }
    var batchSize = 512 { didSet { persistSettings() } }
    var threadCount = 0                    // 0 = as many as there are cores
    var flashAttention = true { didSet { persistSettings() } }
    var useMemoryMapping = true { didSet { persistSettings() } }
    /// Off by default so every prefill is measured cold. See docs/DECISIONS.md.
    var reuseKVCache = false { didSet { persistSettings() } }
    /// 0 = f16 (reference), 1 = q8_0.
    var kvCacheType = 0 { didSet { persistSettings() } }
    var defaultMaxTokens = 4096

    /// Everything a load depends on, in one value: what is persisted, what the admin API
    /// reads and writes, and what a test log should record next to its results.
    struct TuningSettings: Codable, Sendable, Equatable {
        var port: UInt16 = 8080
        var contextLength = 32768
        var batchSize = 512
        var flashAttention = true
        var useMemoryMapping = true
        var reuseKVCache = false
        var kvCacheType = "f16"

        enum CodingKeys: String, CodingKey {
            case port
            case contextLength = "context_length"
            case batchSize = "batch_size"
            case flashAttention = "flash_attention"
            case useMemoryMapping = "memory_mapping"
            case reuseKVCache = "reuse_kv_cache"
            case kvCacheType = "kv_cache_type"
        }
    }

    var settings: TuningSettings {
        TuningSettings(port: port, contextLength: contextLength, batchSize: batchSize,
                       flashAttention: flashAttention, useMemoryMapping: useMemoryMapping,
                       reuseKVCache: reuseKVCache, kvCacheType: kvCacheType == 1 ? "q8_0" : "f16")
    }

    @ObservationIgnored private var restoringSettings = false
    private static let settingsKey = "tuningSettings"

    private func persistSettings() {
        guard !restoringSettings, let data = try? JSONEncoder().encode(settings) else { return }
        UserDefaults.standard.set(data, forKey: Self.settingsKey)
    }

    private func restoreSettings() {
        guard let data = UserDefaults.standard.data(forKey: Self.settingsKey),
              let saved = try? JSONDecoder().decode(TuningSettings.self, from: data) else { return }
        apply(saved)
    }

    /// Applies a whole configuration at once, persisting it once rather than per field.
    func apply(_ new: TuningSettings) {
        restoringSettings = true
        port = new.port
        contextLength = new.contextLength
        batchSize = new.batchSize
        flashAttention = new.flashAttention
        useMemoryMapping = new.useMemoryMapping
        reuseKVCache = new.reuseKVCache
        kvCacheType = new.kvCacheType.lowercased() == "q8_0" ? 1 : 0
        restoringSettings = false
        persistSettings()
    }

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
        restoreSettings()
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

    /// `useProjector` is a choice, not an inference.
    ///
    /// Pairing a model with "the only projector present" put a 4B vision tower on a 1B text
    /// model, and mtmd does not return an error for that — it aborts, taking the app with
    /// it. The pairing heuristic cannot tell the two apart from filenames alone, because the
    /// published projector is called `mmproj-model-f16.gguf` and shares nothing with the
    /// model's name, so the decision belongs to whoever can see both.
    func load(_ pair: ModelPair, useProjector: Bool = true) async {
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
        options.kvCacheType = kvCacheType

        do {
            try await engine.load(.init(name: pair.name,
                                        modelPath: pair.model.url,
                                        projectorPath: useProjector ? pair.projector?.url : nil),
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

    // MARK: Remote configuration

    /// A load request from the admin API. Every field is optional: whatever is left out keeps
    /// its current value, so a sweep can change one setting per call.
    struct RemoteLoadRequest: Decodable {
        let model: String?
        let projector: Bool?
        let contextLength: Int?
        let batchSize: Int?
        let flashAttention: Bool?
        let memoryMapping: Bool?
        let reuseKVCache: Bool?
        let kvCacheType: String?

        enum CodingKeys: String, CodingKey {
            case model, projector
            case contextLength = "context_length"
            case batchSize = "batch_size"
            case flashAttention = "flash_attention"
            case memoryMapping = "memory_mapping"
            case reuseKVCache = "reuse_kv_cache"
            case kvCacheType = "kv_cache_type"
        }
    }

    struct RemoteStatus: Encodable {
        let ok: Bool
        let error: String?
        let loadedModel: String?
        let vision: Bool
        let configuration: String
        let settings: TuningSettings
        let models: [String]
        let footprintBytes: UInt64
        let availableMemoryBytes: UInt64

        enum CodingKeys: String, CodingKey {
            case ok, error, vision, configuration, settings, models
            case loadedModel = "loaded_model"
            case footprintBytes = "footprint_bytes"
            case availableMemoryBytes = "available_memory_bytes"
        }
    }

    func remoteStatus(ok: Bool = true, error: String? = nil) async -> RemoteStatus {
        store.refresh()
        return RemoteStatus(ok: ok, error: error,
                            loadedModel: selectedPair?.name,
                            vision: await engine.supportsImages,
                            configuration: await engine.configurationLabel,
                            settings: settings,
                            models: store.pairs.map(\.name),
                            footprintBytes: MemoryProbe.footprintBytes(),
                            availableMemoryBytes: MemoryProbe.availableBytes())
    }

    /// Reconfigures and reloads from the network, so a parameter sweep runs from a laptop
    /// without anyone touching the device between steps. Touching it is exactly what spoils
    /// a thermal measurement and what a tester forgets to do the same way twice.
    ///
    /// The port is deliberately not settable here: changing it would cut the connection the
    /// request arrived on, and the caller would never learn whether it worked.
    func remoteLoad(_ body: Data) async -> (status: Int, body: RemoteStatus) {
        let request: RemoteLoadRequest
        do {
            request = try JSONDecoder().decode(RemoteLoadRequest.self, from: body.isEmpty ? Data("{}".utf8) : body)
        } catch {
            return (400, await remoteStatus(ok: false, error: "body is not a valid load request: \(error)"))
        }
        if let value = request.contextLength, !(2048...131072).contains(value) {
            return (400, await remoteStatus(ok: false, error: "context_length must be within 2048...131072"))
        }
        if let value = request.batchSize, !(32...4096).contains(value) {
            return (400, await remoteStatus(ok: false, error: "batch_size must be within 32...4096"))
        }
        if let value = request.kvCacheType, !["f16", "q8_0"].contains(value.lowercased()) {
            return (400, await remoteStatus(ok: false, error: "kv_cache_type must be f16 or q8_0"))
        }

        store.refresh()
        guard let name = request.model ?? selectedPair?.name,
              let pair = store.pairs.first(where: {
                  $0.name.caseInsensitiveCompare(name) == .orderedSame ||
                  $0.model.name.caseInsensitiveCompare(name) == .orderedSame
              }) else {
            return (404, await remoteStatus(ok: false,
                                            error: "no model named \(request.model ?? "(none given)") on the device"))
        }

        var next = settings
        if let value = request.contextLength { next.contextLength = value }
        if let value = request.batchSize { next.batchSize = value }
        if let value = request.flashAttention { next.flashAttention = value }
        if let value = request.memoryMapping { next.useMemoryMapping = value }
        if let value = request.reuseKVCache { next.reuseKVCache = value }
        if let value = request.kvCacheType { next.kvCacheType = value.lowercased() }
        apply(next)

        LLMBridge.noteEvent("remote load requested: \(pair.name) \(next)")
        await load(pair, useProjector: request.projector ?? pair.supportsVision)
        if let failure = loadError {
            return (500, await remoteStatus(ok: false, error: failure))
        }
        return (200, await remoteStatus())
    }

    func remoteUnload() async -> RemoteStatus {
        await unloadModel()
        return await remoteStatus()
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
            runSelfTest: { [engine, grammar] in await SelfTest(engine: engine, grammar: grammar).run() },
            adminConfig: { [weak self] in
                guard let self else { return (503, Data()) }
                let status = await self.remoteStatus()
                return (200, (try? JSONEncoder.api.encode(status)) ?? Data())
            },
            adminLoad: { [weak self] body in
                guard let self else { return (503, Data()) }
                let (code, status) = await self.remoteLoad(body)
                return (code, (try? JSONEncoder.api.encode(status)) ?? Data())
            },
            adminUnload: { [weak self] in
                guard let self else { return (503, Data()) }
                let status = await self.remoteUnload()
                return (200, (try? JSONEncoder.api.encode(status)) ?? Data())
            }
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
