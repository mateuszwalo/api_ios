import Foundation

/// Maps HTTP requests onto the inference stack and produces the log entry for each one.
///
/// Deliberately free of UI: everything it needs arrives as a closure, so the whole request
/// path can be exercised in tests with a stub engine and no server, no model and no device.
struct APIRouter: Sendable {

    let engine: any InferenceEngine
    let queue: InferenceQueue
    let grammar: GrammarCompiler
    let defaultMaxTokens: Int
    let record: @Sendable (RequestLogEntry) async -> Void
    let stats: @Sendable () async -> StatsSnapshot
    let logExport: @Sendable () async -> String
    let runSelfTest: @Sendable () async -> Data

    func route(_ request: HTTPRequestMessage) async -> HTTPResponseMessage {
        switch (request.method, request.path) {
        case ("POST", "/v1/chat/completions"):
            return await chatCompletions(request)
        case ("GET", "/v1/models"), ("GET", "/models"):
            return await models()
        case ("GET", "/health"), ("GET", "/healthz"):
            return await health()
        case ("GET", "/v1/stats"), ("GET", "/stats"):
            return await statsResponse()
        case ("GET", "/logs"), ("GET", "/logs.jsonl"):
            return .text(200, await logExport(), contentType: "application/x-ndjson")
        case ("GET", "/v1/selftest"), ("GET", "/selftest"):
            return .json(200, await runSelfTest())
        case ("GET", "/"):
            return .text(200, "LocalLLM Server. POST /v1/chat/completions\n")
        default:
            return error(404, "no route for \(request.method) \(request.path)", code: "not_found")
        }
    }

    // MARK: - Chat completions

    private func chatCompletions(_ http: HTTPRequestMessage) async -> HTTPResponseMessage {
        let started = Date()
        let queueDepth = await queue.depth
        let redactedRequest = BodyRedactor.redact(String(decoding: http.body, as: UTF8.self))

        let decoded: ChatCompletionRequest
        do {
            decoded = try JSONDecoder().decode(ChatCompletionRequest.self, from: http.body)
        } catch {
            let message = "request body is not a valid chat completion: \(error)"
            await logFailure(path: http.path, status: 400, message: message, started: started,
                             queueDepth: queueDepth, requestBody: redactedRequest)
            return self.error(400, message, code: "invalid_request")
        }

        let imageCount = decoded.messages.reduce(0) { $0 + $1.content.images.count }
        let hasSchema = decoded.responseFormat?.jsonSchema != nil

        let handler = ChatCompletionsHandler(engine: engine,
                                             queue: queue,
                                             grammarCompiler: { try grammar.compile($0) },
                                             defaultMaxTokens: defaultMaxTokens)
        do {
            let response = try await handler.handle(decoded)
            let payload = try JSONEncoder.api.encode(response)
            let entry = RequestLogEntry(
                id: UUID(), timestamp: started, path: http.path, status: 200,
                model: response.model,
                promptTokens: response.usage.promptTokens,
                completionTokens: response.usage.completionTokens,
                prefillMs: response.timings?.prefillMs ?? 0,
                decodeMs: response.timings?.decodeMs ?? 0,
                totalMs: Int(Date().timeIntervalSince(started) * 1000),
                imageCount: imageCount, hadGrammar: hasSchema,
                queueDepthOnArrival: queueDepth,
                footprintBytes: MemoryProbe.footprintBytes(),
                thermalState: MemoryProbe.thermalStateName(),
                error: nil,
                requestBody: redactedRequest,
                responseBody: BodyRedactor.redact(String(decoding: payload, as: UTF8.self)))
            await record(entry)
            return .json(200, payload)
        } catch let failure as ChatCompletionsHandler.Failure {
            await logFailure(path: http.path, status: failure.status,
                             message: failure.body.error.message, started: started,
                             queueDepth: queueDepth, requestBody: redactedRequest,
                             imageCount: imageCount, hadGrammar: hasSchema)
            var headers: [String: String] = [:]
            if let retryAfter = failure.retryAfter { headers["Retry-After"] = String(retryAfter) }
            let body = (try? JSONEncoder.api.encode(failure.body)) ?? Data()
            return .json(failure.status, body, extraHeaders: headers)
        } catch {
            let message = "unexpected failure: \(error)"
            await logFailure(path: http.path, status: 500, message: message, started: started,
                             queueDepth: queueDepth, requestBody: redactedRequest,
                             imageCount: imageCount, hadGrammar: hasSchema)
            return self.error(500, message, code: "internal_error")
        }
    }

    // MARK: - Status endpoints

    private func models() async -> HTTPResponseMessage {
        let name = await engine.loadedModelName
        let entries = name.map {
            [ModelsListResponse.Entry(id: $0, created: Int(Date().timeIntervalSince1970))]
        } ?? []
        let body = (try? JSONEncoder.api.encode(ModelsListResponse(data: entries))) ?? Data()
        return .json(200, body)
    }

    private func health() async -> HTTPResponseMessage {
        let loaded = await engine.isLoaded
        let snapshot = await stats()
        let body = (try? JSONEncoder.api.encode(snapshot)) ?? Data()
        // 503 while no model is loaded, so a probe fails for the same reason a request
        // would, rather than reporting health the inference path cannot deliver.
        return .json(loaded ? 200 : 503, body)
    }

    private func statsResponse() async -> HTTPResponseMessage {
        let body = (try? JSONEncoder.api.encode(await stats())) ?? Data()
        return .json(200, body)
    }

    // MARK: - Helpers

    private func error(_ status: Int, _ message: String, code: String) -> HTTPResponseMessage {
        let body = status >= 500
            ? OpenAIErrorBody.server(message, code: code)
            : OpenAIErrorBody.invalidRequest(message, code: code)
        return .json(status, (try? JSONEncoder.api.encode(body)) ?? Data())
    }

    private func logFailure(path: String, status: Int, message: String, started: Date,
                            queueDepth: Int, requestBody: String,
                            imageCount: Int = 0, hadGrammar: Bool = false) async {
        await record(RequestLogEntry(
            id: UUID(), timestamp: started, path: path, status: status,
            model: await engine.loadedModelName,
            promptTokens: 0, completionTokens: 0, prefillMs: 0, decodeMs: 0,
            totalMs: Int(Date().timeIntervalSince(started) * 1000),
            imageCount: imageCount, hadGrammar: hadGrammar,
            queueDepthOnArrival: queueDepth,
            footprintBytes: MemoryProbe.footprintBytes(),
            thermalState: MemoryProbe.thermalStateName(),
            error: message,
            requestBody: requestBody, responseBody: ""))
    }
}

/// What `/v1/stats` and `/health` report, and what the status screen shows.
struct StatsSnapshot: Codable, Sendable {
    let modelLoaded: Bool
    let model: String?
    let contextLength: Int
    let supportsImages: Bool
    let queueDepth: Int
    let served: Int
    let failed: Int
    let footprintBytes: UInt64
    let peakFootprintBytes: UInt64
    let availableMemoryBytes: UInt64
    let thermalState: String
    let uptimeSeconds: Int
    /// Why the last load attempt failed, if it did. Surfaced over HTTP because the person
    /// who can read the screen and the person who can fix the code are rarely the same one,
    /// and retyping a message from a photograph of an iPad loses the details that matter.
    let lastError: String?

    enum CodingKeys: String, CodingKey {
        case modelLoaded = "model_loaded"
        case model
        case contextLength = "context_length"
        case supportsImages = "supports_images"
        case queueDepth = "queue_depth"
        case served
        case failed
        case footprintBytes = "footprint_bytes"
        case peakFootprintBytes = "peak_footprint_bytes"
        case availableMemoryBytes = "available_memory_bytes"
        case thermalState = "thermal_state"
        case uptimeSeconds = "uptime_seconds"
        case lastError = "last_error"
    }
}

extension JSONEncoder {
    /// One encoder configuration for everything that goes on the wire. `sortedKeys` keeps
    /// captured bodies diffable between runs, which is how a change in output gets noticed.
    static let api: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        return encoder
    }()
}
