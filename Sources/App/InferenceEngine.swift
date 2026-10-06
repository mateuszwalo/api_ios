import Foundation

/// What the HTTP layer needs from an inference backend.
///
/// The protocol exists so the request/response contract can be tested without llama.cpp,
/// a model file or a device. Contract bugs — a missing `object` field, the wrong token
/// limit key, a schema that silently stops being enforced — are cheap to catch here and
/// expensive to catch on an iPad with a 3 GB model loaded.
protocol InferenceEngine: Sendable {
    var isLoaded: Bool { get async }
    var loadedModelName: String? { get async }
    var contextLength: Int { get async }

    /// Tokens the prompt will occupy, including image tokens. Used to decide whether the
    /// requested generation limit fits, before any decoding starts.
    func measurePrompt(_ prompt: EnginePrompt) async throws -> Int

    func generate(_ request: EngineRequest) async throws -> EngineResult
}

/// A prompt in the form the engine consumes: roles and text, plus any images.
///
/// The chat template is applied by the engine from the model's own GGUF metadata, never
/// assembled here. A template written by hand would drift from the one the reference
/// runtime applies, and the resulting difference in output would be indistinguishable
/// from a hardware effect.
struct EnginePrompt: Sendable {
    struct Turn: Sendable {
        let role: String
        let text: String
        let images: [ImagePayload]
    }
    let turns: [Turn]
}

struct EngineRequest: Sendable {
    let prompt: EnginePrompt
    /// GBNF grammar compiled from the request's JSON schema, or nil for free generation.
    let grammar: String?
    let maxTokens: Int
    let temperature: Double
    let stopSequences: [String]
    let seed: UInt32?
}

struct EngineResult: Sendable {
    let text: String
    let promptTokens: Int
    let completionTokens: Int
    let prefillMilliseconds: Int
    let decodeMilliseconds: Int
    /// True when generation stopped because the token budget ran out rather than because
    /// the model emitted a stop token.
    let hitTokenLimit: Bool
    /// Prompt tokens taken from the cache instead of evaluated; zero with reuse off.
    var cachedPromptTokens: Int = 0
    /// A short description of the settings the engine ran under, echoed into the response.
    var configuration: String = ""

    /// Over the tokens actually evaluated. With prefix reuse most of a prompt may come from
    /// the cache, and dividing the whole prompt by the time spent on its tail would report a
    /// prefill rate many times faster than the hardware is.
    var prefillTokensPerSecond: Double {
        let evaluated = promptTokens - cachedPromptTokens
        return prefillMilliseconds > 0 ? Double(evaluated) * 1000.0 / Double(prefillMilliseconds) : 0
    }

    var decodeTokensPerSecond: Double {
        decodeMilliseconds > 0 ? Double(completionTokens) * 1000.0 / Double(decodeMilliseconds) : 0
    }
}

enum EngineError: Error, CustomStringConvertible {
    case notLoaded
    case promptTooLong(promptTokens: Int, contextLength: Int)
    case grammarRejected(String)
    case backend(String)

    var description: String {
        switch self {
        case .notLoaded:
            return "no model is loaded"
        case .promptTooLong(let p, let c):
            return "prompt of \(p) tokens does not fit in a context of \(c)"
        case .grammarRejected(let why):
            return "response_format.json_schema could not be compiled to a grammar: \(why)"
        case .backend(let why):
            return why
        }
    }
}
