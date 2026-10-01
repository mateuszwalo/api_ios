import Foundation

/// A JSON value decoded without a fixed shape.
///
/// The request's `response_format.json_schema.schema` must reach the grammar converter
/// byte-for-byte as the client sent it. Decoding it into anything narrower would drop
/// keys the converter needs — `additionalProperties` and `$defs` among them — and the
/// loss would be silent: the grammar would still build, just permitting outputs the
/// client believes it has forbidden.
enum JSONValue: Codable, Sendable, Equatable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() {
            self = .null
        } else if let v = try? c.decode(Bool.self) {
            self = .bool(v)
        } else if let v = try? c.decode(Int.self) {
            // Before Double: JSON has one number type, and re-encoding an integral
            // `minItems: 1` as `1.0` produces a schema the converter may reject.
            self = .int(v)
        } else if let v = try? c.decode(Double.self) {
            self = .double(v)
        } else if let v = try? c.decode(String.self) {
            self = .string(v)
        } else if let v = try? c.decode([JSONValue].self) {
            self = .array(v)
        } else if let v = try? c.decode([String: JSONValue].self) {
            self = .object(v)
        } else {
            throw DecodingError.dataCorruptedError(
                in: c, debugDescription: "value is not valid JSON")
        }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let v): try c.encode(v)
        case .int(let v): try c.encode(v)
        case .double(let v): try c.encode(v)
        case .string(let v): try c.encode(v)
        case .array(let v): try c.encode(v)
        case .object(let v): try c.encode(v)
        }
    }

    /// Serialised form handed to the grammar converter.
    ///
    /// `sortedKeys` is deliberate: the converter is called once per distinct schema and
    /// the result is cached by this string, so two requests carrying the same schema with
    /// keys in a different order must produce the same cache key. `withoutEscapingSlashes`
    /// keeps `$ref` pointers such as `#/$defs/Entry` readable in logs and error messages.
    func serialized() throws -> String {
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        guard let s = String(data: try enc.encode(self), encoding: .utf8) else {
            throw JSONValueError.notUTF8
        }
        return s
    }

    subscript(key: String) -> JSONValue? {
        guard case .object(let o) = self else { return nil }
        return o[key]
    }
}

enum JSONValueError: Error {
    case notUTF8
}
