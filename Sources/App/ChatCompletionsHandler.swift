import Foundation

/// Turns a decoded `ChatCompletionRequest` into a response, or into an HTTP failure that
/// the client's own retry logic can act on.
///
/// Status codes matter more than usual here. The reference client retries 429, 500, 502,
/// 503, 504 and 529, and does not retry anything else — so a transient condition reported
/// as 400 ends the client's entire run, while a permanent one reported as 503 has it
/// retry a request that can never succeed.
struct ChatCompletionsHandler: Sendable {
    let engine: any InferenceEngine
    let queue: InferenceQueue
    let grammarCompiler: @Sendable (String) throws -> String
    let defaultMaxTokens: Int

    struct Failure: Error {
        let status: Int
        let body: OpenAIErrorBody
        /// Seconds the client should wait before retrying, for statuses where retrying is
        /// the right response.
        let retryAfter: Int?

        init(status: Int, message: String, code: String? = nil, retryAfter: Int? = nil) {
            self.status = status
            self.retryAfter = retryAfter
            self.body = status >= 500
                ? .server(message, code: code)
                : .invalidRequest(message, code: code)
        }
    }

    func handle(_ request: ChatCompletionRequest) async throws -> ChatCompletionResponse {
        guard await engine.isLoaded else {
            // 503, not 400: the model may still be loading, and the client is entitled to
            // come back. A 400 would abort a run that was about to become possible.
            throw Failure(status: 503,
                          message: "no model is loaded; load one in the app and retry",
                          code: "model_not_loaded",
                          retryAfter: 5)
        }

        let prompt = try buildPrompt(request)
        let grammar = try compileGrammar(request)
        let limit = try await resolveTokenLimit(request, prompt: prompt)

        let engineRequest = EngineRequest(
            prompt: prompt,
            grammar: grammar,
            maxTokens: limit,
            // Sampling is fixed except for temperature, which a request may override.
            // Everything else stays at the reference configuration; see docs/DECISIONS.md.
            temperature: request.temperature ?? 0,
            stopSequences: request.stop?.values ?? [],
            seed: request.seed
        )

        let result: EngineResult
        do {
            result = try await queue.run { try await engine.generate(engineRequest) }
        } catch is CancellationError {
            // The client went away. Nothing to report to it; the queue has already moved on.
            throw Failure(status: 499, message: "client disconnected", code: "cancelled")
        } catch let e as EngineError {
            throw Failure(status: 500, message: e.description, code: "inference_failed")
        }

        let modelName = request.model ?? (await engine.loadedModelName) ?? "local"
        return ChatCompletionResponse(
            id: "chatcmpl-" + String(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24)),
            created: Int(Date().timeIntervalSince1970),
            model: modelName,
            choices: [.init(index: 0,
                            message: .init(content: result.text),
                            finishReason: result.hitTokenLimit ? "length" : "stop")],
            usage: .init(promptTokens: result.promptTokens,
                         completionTokens: result.completionTokens,
                         totalTokens: result.promptTokens + result.completionTokens),
            timings: .init(prefillMs: result.prefillMilliseconds,
                           decodeMs: result.decodeMilliseconds,
                           prefillTps: result.prefillTokensPerSecond,
                           decodeTps: result.decodeTokensPerSecond)
        )
    }

    // MARK: - Pieces

    private func buildPrompt(_ request: ChatCompletionRequest) throws -> EnginePrompt {
        var turns: [EnginePrompt.Turn] = []
        for message in request.messages {
            var images: [ImagePayload] = []
            for ref in message.content.images {
                do {
                    images.append(try ImagePayload.parse(ref.url))
                } catch let e as ImagePayload.ParseError {
                    throw Failure(status: 400, message: e.description, code: "invalid_image")
                }
            }
            if message.role != "user" && !images.isEmpty {
                // Images in a system message are not something the reference client sends,
                // and the template has no place to put them.
                throw Failure(status: 400,
                              message: "images are only accepted in user messages",
                              code: "invalid_image_role")
            }
            turns.append(.init(role: message.role,
                               text: message.content.plainText,
                               images: images))
        }
        guard !turns.isEmpty else {
            throw Failure(status: 400, message: "messages must not be empty")
        }
        return EnginePrompt(turns: turns)
    }

    private func compileGrammar(_ request: ChatCompletionRequest) throws -> String? {
        guard let format = request.responseFormat else { return nil }
        switch format.type {
        case "text":
            return nil
        case "json_object":
            // No schema: constrain to syntactically valid JSON and nothing more.
            return Self.anyJSONGrammar
        case "json_schema":
            guard let spec = format.jsonSchema else {
                throw Failure(status: 400,
                              message: "response_format.json_schema is missing",
                              code: "invalid_response_format")
            }
            do {
                return try grammarCompiler(spec.schema.serialized())
            } catch {
                // Never fall back to describing the schema in the prompt. A silent
                // degradation produces output that looks right and is no longer
                // guaranteed, and every measurement taken afterwards is contaminated
                // without anyone being told.
                throw Failure(status: 400,
                              message: EngineError.grammarRejected("\(error)").description,
                              code: "schema_not_convertible")
            }
        default:
            throw Failure(status: 400,
                          message: "unsupported response_format.type '\(format.type)'",
                          code: "invalid_response_format")
        }
    }

    /// Decides how many tokens may be generated.
    ///
    /// The limit arrives as `max_completion_tokens`; `max_tokens` is the older spelling and
    /// is accepted too. Reading only the older one lets a request generate unbounded until
    /// it hits the context wall — and it does so quietly, which is worse than failing.
    private func resolveTokenLimit(_ request: ChatCompletionRequest,
                                   prompt: EnginePrompt) async throws -> Int {
        let contextLength = await engine.contextLength
        let promptTokens: Int
        do {
            promptTokens = try await engine.measurePrompt(prompt)
        } catch let e as EngineError {
            throw Failure(status: 500, message: e.description, code: "tokenize_failed")
        }

        let available = contextLength - promptTokens
        guard available > 0 else {
            throw Failure(status: 400,
                          message: EngineError.promptTooLong(promptTokens: promptTokens,
                                                             contextLength: contextLength).description,
                          code: "context_length_exceeded")
        }

        let requested = request.generationLimit ?? defaultMaxTokens
        // Trim rather than refuse: a request whose prompt fits but whose ceiling does not
        // is answerable, and the client learns what happened from finish_reason "length".
        return min(requested, available)
    }

    /// Grammar admitting any well-formed JSON value, for `response_format: json_object`.
    static let anyJSONGrammar = """
    root   ::= object | array
    value  ::= object | array | string | number | ("true" | "false" | "null") ws
    object ::= "{" ws ( string ":" ws value ("," ws string ":" ws value)* )? "}" ws
    array  ::= "[" ws ( value ("," ws value)* )? "]" ws
    string ::= "\\"" ( [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4}) )* "\\"" ws
    number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
    ws     ::= [ \\t\\n]{0,20}
    """
}
