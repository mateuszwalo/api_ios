import Foundation

// Wire types for the OpenAI Chat Completions protocol.
//
// Shapes here were verified against a request captured from the reference client
// (an HTTP stub plus a real agent call), not inferred from documentation. Three details
// differ from what the protocol docs suggest, and each one breaks the integration
// silently rather than loudly:
//
//   1. the generation limit arrives as `max_completion_tokens`; `max_tokens` is omitted
//   2. `json_schema` carries only `name` and `schema` — there is no `strict` key
//   3. in a multimodal message the image part precedes the text part
//
// See docs/DECISIONS.md section 3.

// MARK: - Request

struct ChatCompletionRequest: Decodable {
    let model: String?
    let messages: [Message]
    let responseFormat: ResponseFormat?
    let temperature: Double?
    let stream: Bool?
    let stop: StopSequences?
    let seed: UInt32?

    /// Generation limit. The reference client sends `max_completion_tokens` and omits
    /// `max_tokens`; older clients do the reverse. Reading only one of them means a
    /// request silently generates until it hits the context wall.
    let maxCompletionTokens: Int?
    let maxTokens: Int?

    var generationLimit: Int? { maxCompletionTokens ?? maxTokens }

    enum CodingKeys: String, CodingKey {
        case model, messages, temperature, stream, stop, seed
        case responseFormat = "response_format"
        case maxCompletionTokens = "max_completion_tokens"
        case maxTokens = "max_tokens"
    }

    struct Message: Decodable {
        let role: String
        let content: Content

        /// `content` is either a bare string or an array of typed parts. Both forms occur
        /// in the same conversation: system messages are strings, a page-plus-instruction
        /// user message is an array.
        enum Content: Decodable {
            case text(String)
            case parts([Part])

            init(from decoder: Decoder) throws {
                let single = try decoder.singleValueContainer()
                if let s = try? single.decode(String.self) {
                    self = .text(s)
                } else {
                    self = .parts(try single.decode([Part].self))
                }
            }

            /// Concatenated text, ignoring images. Image markers are inserted by the
            /// multimodal tokenizer, not here.
            var plainText: String {
                switch self {
                case .text(let s): return s
                case .parts(let p): return p.compactMap(\.text).joined(separator: "\n")
                }
            }

            var images: [ImageRef] {
                guard case .parts(let p) = self else { return [] }
                return p.compactMap(\.imageURL)
            }
        }

        struct Part: Decodable {
            let type: String
            let text: String?
            let imageURL: ImageRef?

            enum CodingKeys: String, CodingKey {
                case type, text
                case imageURL = "image_url"
            }
        }

        struct ImageRef: Decodable {
            let url: String
        }
    }

    struct ResponseFormat: Decodable {
        let type: String
        let jsonSchema: JSONSchemaSpec?

        enum CodingKeys: String, CodingKey {
            case type
            case jsonSchema = "json_schema"
        }

        /// Note the absence of `strict`. The reference client never sends it, so its
        /// absence must not be read as permission to relax enforcement — the schema is
        /// always binding. It is decoded only so that a client which does send it is
        /// not rejected.
        struct JSONSchemaSpec: Decodable {
            let name: String
            let schema: JSONValue
            let strict: Bool?
            let description: String?
        }
    }

    /// `stop` is either a string or an array of them.
    enum StopSequences: Decodable {
        case one(String)
        case many([String])

        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let s = try? c.decode(String.self) { self = .one(s) }
            else { self = .many(try c.decode([String].self)) }
        }

        var values: [String] {
            switch self {
            case .one(let s): return [s]
            case .many(let a): return a
            }
        }
    }
}

// MARK: - Response

/// `id`, `object`, `created`, `model` and `choices` are required by the client SDK's own
/// response model. Omitting any of them raises a decode error on the client before the
/// generated content is ever examined — which looks, from the outside, exactly like the
/// model having produced nothing.
struct ChatCompletionResponse: Encodable {
    let id: String
    let object: String = "chat.completion"
    let created: Int
    let model: String
    let choices: [Choice]
    let usage: Usage
    let timings: Timings?

    struct Choice: Encodable {
        let index: Int
        let message: Message
        let finishReason: String

        enum CodingKeys: String, CodingKey {
            case index, message
            case finishReason = "finish_reason"
        }

        struct Message: Encodable {
            let role: String = "assistant"
            let content: String
        }
    }

    /// Optional to the SDK, mandatory here: these counts are the measurement. Returning
    /// zeros would leave the whole exercise without a result.
    struct Usage: Encodable {
        let promptTokens: Int
        let completionTokens: Int
        let totalTokens: Int

        enum CodingKeys: String, CodingKey {
            case promptTokens = "prompt_tokens"
            case completionTokens = "completion_tokens"
            case totalTokens = "total_tokens"
        }
    }

    /// Deliberately a sibling of `usage`, never a member of it: the shape of `usage` is
    /// part of the contract, and clients are entitled to parse it strictly.
    struct Timings: Encodable {
        let prefillMs: Int
        let decodeMs: Int
        let prefillTps: Double
        let decodeTps: Double
        /// Prompt tokens served from the prefix cache rather than evaluated. prefill_tps is
        /// computed over the rest, so the two together are the honest prefill figure.
        var cachedTokens: Int = 0
        /// The settings this response was produced under. When configurations are compared,
        /// every result says which one it came from, and no spreadsheet has to remember it.
        var config: String? = nil

        enum CodingKeys: String, CodingKey {
            case prefillMs = "prefill_ms"
            case decodeMs = "decode_ms"
            case prefillTps = "prefill_tps"
            case decodeTps = "decode_tps"
            case cachedTokens = "cached_tokens"
            case config
        }
    }
}

// MARK: - Errors

struct OpenAIErrorBody: Encodable {
    let error: Detail

    struct Detail: Encodable {
        let message: String
        let type: String
        let code: String?
    }

    static func invalidRequest(_ message: String, code: String? = nil) -> OpenAIErrorBody {
        .init(error: .init(message: message, type: "invalid_request_error", code: code))
    }

    static func server(_ message: String, code: String? = nil) -> OpenAIErrorBody {
        .init(error: .init(message: message, type: "server_error", code: code))
    }
}

// MARK: - Models listing

struct ModelsListResponse: Encodable {
    let object: String = "list"
    let data: [Entry]

    struct Entry: Encodable {
        let id: String
        let object: String = "model"
        let created: Int
        let ownedBy: String = "local"

        enum CodingKeys: String, CodingKey {
            case id, object, created
            case ownedBy = "owned_by"
        }
    }
}
