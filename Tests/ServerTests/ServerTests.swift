import XCTest
@testable import LocalLLMServer

// MARK: - HTTP parsing

final class HTTPParserTests: XCTestCase {

    private func parse(_ raw: String) -> HTTPRequestParser.Outcome {
        var parser = HTTPRequestParser()
        parser.feed(Data(raw.utf8))
        return parser.next()
    }

    func testSimplePost() throws {
        let outcome = parse("POST /v1/chat/completions HTTP/1.1\r\nHost: a\r\nContent-Length: 2\r\n\r\n{}")
        guard case .complete(let request) = outcome else { return XCTFail("expected a complete request") }
        XCTAssertEqual(request.method, "POST")
        XCTAssertEqual(request.path, "/v1/chat/completions")
        XCTAssertEqual(request.body, Data("{}".utf8))
    }

    /// A header split across two reads is the normal case for a large request, not an edge
    /// case: the first TCP segment rarely contains the whole head.
    func testHeadSplitAcrossReads() {
        var parser = HTTPRequestParser()
        parser.feed(Data("POST /v1/chat/completions HTTP/1.1\r\nContent-Len".utf8))
        XCTAssertEqual(parser.next(), .incomplete)
        parser.feed(Data("gth: 5\r\n\r\nhel".utf8))
        XCTAssertEqual(parser.next(), .incomplete)
        parser.feed(Data("lo".utf8))
        guard case .complete(let request) = parser.next() else { return XCTFail("expected completion") }
        XCTAssertEqual(request.body, Data("hello".utf8))
    }

    /// Two requests arriving in one segment must both be served. The client pools
    /// connections, so this is how a batch arrives.
    func testPipelinedRequests() {
        var parser = HTTPRequestParser()
        parser.feed(Data("GET /health HTTP/1.1\r\n\r\nGET /v1/models HTTP/1.1\r\n\r\n".utf8))
        guard case .complete(let first) = parser.next() else { return XCTFail("expected the first request") }
        XCTAssertEqual(first.path, "/health")
        guard case .complete(let second) = parser.next() else { return XCTFail("expected the second request") }
        XCTAssertEqual(second.path, "/v1/models")
        XCTAssertEqual(parser.next(), .incomplete)
    }

    func testChunkedIsRefusedClearly() {
        guard case .failure(let status, let message) =
                parse("POST /v1/chat/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n") else {
            return XCTFail("chunked should be refused")
        }
        XCTAssertEqual(status, 411)
        XCTAssertTrue(message.contains("Content-Length"))
    }

    func testOversizedBodyIsRefusedBeforeItArrives() {
        guard case .failure(let status, _) =
                parse("POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 999999999\r\n\r\n") else {
            return XCTFail("an oversized body should be refused")
        }
        XCTAssertEqual(status, 413)
    }

    func testQueryParsing() {
        guard case .complete(let request) = parse("GET /logs?limit=10&format=jsonl HTTP/1.1\r\n\r\n") else {
            return XCTFail("expected completion")
        }
        XCTAssertEqual(request.path, "/logs")
        XCTAssertEqual(request.query["limit"], "10")
        XCTAssertEqual(request.query["format"], "jsonl")
    }

    /// HTTP/1.1 keeps the connection open unless told otherwise. Getting this backwards
    /// would make the client reconnect between every pair of long requests.
    func testKeepAliveDefaultsOn() {
        guard case .complete(let request) = parse("GET /health HTTP/1.1\r\n\r\n") else {
            return XCTFail("expected completion")
        }
        XCTAssertTrue(HTTPRequestParser.keepAlive(request))

        guard case .complete(let closing) = parse("GET /health HTTP/1.1\r\nConnection: close\r\n\r\n") else {
            return XCTFail("expected completion")
        }
        XCTAssertFalse(HTTPRequestParser.keepAlive(closing))
    }

    func testResponseSerialisation() {
        let response = HTTPResponseMessage.json(200, Data("{\"a\":1}".utf8))
        let text = String(decoding: response.serialized(keepAlive: true), as: UTF8.self)
        XCTAssertTrue(text.hasPrefix("HTTP/1.1 200 OK\r\n"))
        XCTAssertTrue(text.contains("Content-Length: 7\r\n"))
        XCTAssertTrue(text.contains("Connection: keep-alive\r\n"))
        XCTAssertTrue(text.hasSuffix("\r\n\r\n{\"a\":1}"))
    }
}

// MARK: - Redaction

final class RedactionTests: XCTestCase {

    func testImagePayloadIsReplacedByItsSize() {
        let body = "{\"url\":\"data:image/jpeg;base64,\(String(repeating: "A", count: 4000))\"}"
        let redacted = BodyRedactor.redact(body)
        XCTAssertFalse(redacted.contains(String(repeating: "A", count: 100)))
        XCTAssertTrue(redacted.contains("image/jpeg"))
        XCTAssertTrue(redacted.contains("base64"))
        XCTAssertLessThan(redacted.count, 200)
    }

    func testTextAroundTheImageIsKept() {
        let body = "{\"text\":\"transcribe this\",\"url\":\"data:image/png;base64,AAAA\"}"
        let redacted = BodyRedactor.redact(body)
        XCTAssertTrue(redacted.contains("transcribe this"))
    }

    func testBodyWithoutImagesIsUnchanged() {
        let body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}"
        XCTAssertEqual(BodyRedactor.redact(body), body)
    }
}

// MARK: - Queue

final class InferenceQueueTests: XCTestCase {

    /// Twenty concurrent requests must all be served, one at a time, in the order they
    /// arrived. Arrival order is not something an actor provides on its own.
    func testRequestsAreServedOneAtATime() async throws {
        let queue = InferenceQueue()
        let tracker = ConcurrencyTracker()

        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<20 {
                group.addTask {
                    _ = try? await queue.run {
                        await tracker.enter()
                        try? await Task.sleep(for: .milliseconds(5))
                        await tracker.leave()
                    }
                }
            }
        }

        let peak = await tracker.peakConcurrent
        XCTAssertEqual(peak, 1, "two requests decoded at once; measurements would not be reproducible")
        let completed = await queue.completedCount
        XCTAssertEqual(completed, 20, "every request must be served, none refused")
    }

    /// A client that hangs up must release its place. Otherwise everything behind it waits
    /// for an answer nobody is going to read.
    func testCancelledWaiterDoesNotBlockTheQueue() async throws {
        let queue = InferenceQueue()
        let holder = Task {
            try await queue.run { try? await Task.sleep(for: .milliseconds(200)) }
        }
        try await Task.sleep(for: .milliseconds(20))

        let abandoned = Task {
            try await queue.run { }
        }
        try await Task.sleep(for: .milliseconds(20))
        abandoned.cancel()

        _ = try await holder.value
        let served = try await withThrowingTaskGroup(of: Bool.self) { group -> Bool in
            group.addTask { try await queue.run { true } }
            return try await group.next() ?? false
        }
        XCTAssertTrue(served, "the queue stalled behind a waiter that had gone away")
    }
}

private actor ConcurrencyTracker {
    private var current = 0
    private(set) var peakConcurrent = 0

    func enter() {
        current += 1
        peakConcurrent = max(peakConcurrent, current)
    }

    func leave() { current -= 1 }
}

// MARK: - Wire contract

final class WireContractTests: XCTestCase {

    /// The generation limit arrives as `max_completion_tokens`. A server reading only
    /// `max_tokens` would generate until the context wall on every request, and would do it
    /// silently.
    func testMaxCompletionTokensIsRead() throws {
        let json = """
        {"model":"m","messages":[{"role":"user","content":"hi"}],
         "max_completion_tokens":12000,"stream":false,"temperature":0}
        """
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(json.utf8))
        XCTAssertEqual(request.generationLimit, 12000)
    }

    func testMaxTokensIsStillAccepted() throws {
        let json = """
        {"messages":[{"role":"user","content":"hi"}],"max_tokens":512}
        """
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(json.utf8))
        XCTAssertEqual(request.generationLimit, 512)
    }

    /// The image part precedes the text part in a real request, and `json_schema` carries no
    /// `strict` key. Both are the shapes observed on the wire, not the ones the protocol
    /// documentation suggests.
    func testMultimodalMessageWithImageFirstAndNoStrictKey() throws {
        let json = """
        {"messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,AAAA"}},
            {"type":"text","text":"transcribe"}]}],
         "response_format":{"type":"json_schema","json_schema":{"name":"R","schema":{"type":"object"}}}}
        """
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(json.utf8))
        XCTAssertEqual(request.messages[0].content.images.count, 1)
        XCTAssertEqual(request.messages[0].content.plainText, "transcribe")
        XCTAssertNil(request.responseFormat?.jsonSchema?.strict)
        XCTAssertEqual(request.responseFormat?.jsonSchema?.name, "R")
    }

    /// The schema must reach the converter exactly as it arrived: `$defs` and
    /// `additionalProperties` included. Losing a key here would still produce a grammar,
    /// just a more permissive one.
    func testSchemaSurvivesDecodingIntact() throws {
        let schema = """
        {"$defs":{"P":{"type":"object","additionalProperties":false,
                       "properties":{"a":{"enum":["x","y"]}},"required":["a"]}},
         "type":"object","additionalProperties":false,
         "properties":{"p":{"$ref":"#/$defs/P"}},"required":["p"]}
        """
        let json = """
        {"messages":[{"role":"user","content":"hi"}],
         "response_format":{"type":"json_schema","json_schema":{"name":"R","schema":\(schema)}}}
        """
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(json.utf8))
        let round = try XCTUnwrap(request.responseFormat?.jsonSchema?.schema.serialized())
        XCTAssertTrue(round.contains("$defs"))
        XCTAssertTrue(round.contains("additionalProperties"))
        XCTAssertTrue(round.contains("#/$defs/P"))
        XCTAssertTrue(round.contains("\"required\""))
    }

    /// `minItems: 1` re-encoded as `1.0` is a schema the converter can reject. JSON has one
    /// number type and Swift does not.
    func testIntegerConstraintsStayIntegers() throws {
        let value = try JSONDecoder().decode(JSONValue.self,
                                             from: Data("{\"minItems\":1,\"minimum\":2}".utf8))
        let text = try value.serialized()
        XCTAssertTrue(text.contains("\"minItems\":1"))
        XCTAssertFalse(text.contains("1.0"))
    }

    func testResponseCarriesEveryFieldTheClientRequires() throws {
        let response = ChatCompletionResponse(
            id: "chatcmpl-1", created: 1, model: "m",
            choices: [.init(index: 0, message: .init(content: "{}"), finishReason: "stop")],
            usage: .init(promptTokens: 10, completionTokens: 2, totalTokens: 12),
            timings: .init(prefillMs: 100, decodeMs: 50, prefillTps: 100, decodeTps: 40))
        let encoded = try JSONEncoder.api.encode(response)
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: encoded) as? [String: Any])

        for key in ["id", "object", "created", "model", "choices", "usage"] {
            XCTAssertNotNil(object[key], "the client's response model requires \(key)")
        }
        XCTAssertEqual(object["object"] as? String, "chat.completion")
        let usage = try XCTUnwrap(object["usage"] as? [String: Any])
        XCTAssertEqual(usage["prompt_tokens"] as? Int, 10)
        // Measurements go beside usage, never inside it.
        XCTAssertNil(usage["prefill_ms"])
        XCTAssertNotNil(object["timings"])
    }
}

// MARK: - Handler behaviour

final class ChatCompletionsHandlerTests: XCTestCase {

    private func handler(engine: StubEngine,
                         grammar: @escaping @Sendable (String) throws -> String = { _ in "root ::= \"x\"" })
    -> ChatCompletionsHandler {
        ChatCompletionsHandler(engine: engine, queue: InferenceQueue(),
                               grammarCompiler: grammar, defaultMaxTokens: 4096)
    }

    private func request(_ json: String) throws -> ChatCompletionRequest {
        try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(json.utf8))
    }

    /// 503 rather than 400 while no model is loaded: the client retries 503 and abandons the
    /// whole run on 400.
    func testNoModelLoadedIsRetryable() async throws {
        let engine = StubEngine(loaded: false)
        do {
            _ = try await handler(engine: engine)
                .handle(try request("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"))
            XCTFail("expected a failure")
        } catch let failure as ChatCompletionsHandler.Failure {
            XCTAssertEqual(failure.status, 503)
            XCTAssertNotNil(failure.retryAfter)
        }
    }

    /// A ceiling that does not fit is trimmed, not refused: the request is answerable and
    /// `finish_reason` tells the client what happened.
    func testTokenLimitIsTrimmedToTheContext() async throws {
        let engine = StubEngine(loaded: true, promptTokens: 30_000, contextLength: 32_768)
        let response = try await handler(engine: engine).handle(
            try request("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_completion_tokens\":12000}"))
        let requested = await engine.lastRequest?.maxTokens
        XCTAssertEqual(requested, 2_768)
        XCTAssertEqual(response.choices[0].finishReason, "length")
    }

    func testPromptLongerThanTheContextIsRejected() async throws {
        let engine = StubEngine(loaded: true, promptTokens: 40_000, contextLength: 32_768)
        do {
            _ = try await handler(engine: engine)
                .handle(try request("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"))
            XCTFail("expected a failure")
        } catch let failure as ChatCompletionsHandler.Failure {
            XCTAssertEqual(failure.status, 400)
            XCTAssertEqual(failure.body.error.code, "context_length_exceeded")
        }
    }

    /// A schema that cannot be compiled fails the request. It must never quietly become a
    /// hint in the prompt: the output would look right and no longer be guaranteed.
    func testUnconvertibleSchemaFailsRatherThanDegrades() async throws {
        let engine = StubEngine(loaded: true)
        let failing: @Sendable (String) throws -> String = { _ in
            throw GrammarCompiler.Failure.rejected("unsupported construct")
        }
        do {
            _ = try await handler(engine: engine, grammar: failing).handle(try request("""
            {"messages":[{"role":"user","content":"hi"}],
             "response_format":{"type":"json_schema","json_schema":{"name":"R","schema":{"type":"object"}}}}
            """))
            XCTFail("expected a failure")
        } catch let failure as ChatCompletionsHandler.Failure {
            XCTAssertEqual(failure.status, 400)
            XCTAssertEqual(failure.body.error.code, "schema_not_convertible")
        }
        let used = await engine.lastRequest
        XCTAssertNil(used, "nothing should have been generated")
    }

    func testGrammarReachesTheEngineWhenASchemaIsPresent() async throws {
        let engine = StubEngine(loaded: true)
        _ = try await handler(engine: engine).handle(try request("""
        {"messages":[{"role":"user","content":"hi"}],
         "response_format":{"type":"json_schema","json_schema":{"name":"R","schema":{"type":"object"}}}}
        """))
        let grammar = await engine.lastRequest?.grammar
        XCTAssertEqual(grammar, "root ::= \"x\"")
    }

    func testJSONObjectModeConstrainsToJSON() async throws {
        let engine = StubEngine(loaded: true)
        _ = try await handler(engine: engine).handle(try request("""
        {"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_object"}}
        """))
        let grammar = await engine.lastRequest?.grammar
        XCTAssertEqual(grammar, ChatCompletionsHandler.anyJSONGrammar)
    }

    func testRemoteImageURLIsRefused() async throws {
        let engine = StubEngine(loaded: true)
        do {
            _ = try await handler(engine: engine).handle(try request("""
            {"messages":[{"role":"user","content":[
                {"type":"image_url","image_url":{"url":"https://example.com/a.jpg"}}]}]}
            """))
            XCTFail("expected a failure")
        } catch let failure as ChatCompletionsHandler.Failure {
            XCTAssertEqual(failure.status, 400)
            XCTAssertTrue(failure.body.error.message.contains("data:"))
        }
    }

    func testModelNameIsEchoedBack() async throws {
        let engine = StubEngine(loaded: true)
        let response = try await handler(engine: engine).handle(
            try request("{\"model\":\"gemma3-4b-quality:latest\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"))
        XCTAssertEqual(response.model, "gemma3-4b-quality:latest")
    }

    func testUsageCarriesRealCounts() async throws {
        let engine = StubEngine(loaded: true, promptTokens: 6116, completionTokens: 1629)
        let response = try await handler(engine: engine).handle(
            try request("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"))
        XCTAssertEqual(response.usage.promptTokens, 6116)
        XCTAssertEqual(response.usage.completionTokens, 1629)
        XCTAssertEqual(response.usage.totalTokens, 7745)
    }
}

/// Stand-in engine: records what it was asked for and answers immediately.
actor StubEngine: InferenceEngine {
    private let loaded: Bool
    private let promptTokens: Int
    private let completionTokens: Int
    private let context: Int
    private(set) var lastRequest: EngineRequest?

    init(loaded: Bool, promptTokens: Int = 10, completionTokens: Int = 5, contextLength: Int = 32_768) {
        self.loaded = loaded
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.context = contextLength
    }

    var isLoaded: Bool { loaded }
    var loadedModelName: String? { loaded ? "stub" : nil }
    var contextLength: Int { context }

    func measurePrompt(_ prompt: EnginePrompt) async throws -> Int { promptTokens }

    func generate(_ request: EngineRequest) async throws -> EngineResult {
        lastRequest = request
        return EngineResult(text: "{}",
                            promptTokens: promptTokens,
                            completionTokens: completionTokens,
                            prefillMilliseconds: 1000,
                            decodeMilliseconds: 500,
                            hitTokenLimit: promptTokens + request.maxTokens >= context)
    }
}
