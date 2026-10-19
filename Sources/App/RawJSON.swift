import Foundation

/// Finds a value inside a JSON document by its object-key path and returns its exact text.
///
/// This exists for one reason: the grammar converter must see the schema with its keys in the
/// order the client wrote them. Decoding through Foundation loses that order before any code
/// can observe it — objects come back as dictionaries — and the grammar then forces the model
/// to emit fields alphabetically. For a schema that begins with `name`, that made models fill
/// in `categories` and `components` before they had written the name of the thing they were
/// classifying, and they guessed. Measured by the client: around 20 points out of 100 lost on
/// correctly read menus.
///
/// So the schema is not decoded at all on its way to the converter. This scanner walks the raw
/// bytes far enough to find where the value starts and ends, and hands back those bytes
/// untouched. It validates only as much structure as it needs to stay in step; the full
/// request has already been decoded by `JSONDecoder` by the time this runs, so malformed input
/// never reaches it.
enum RawJSON {

    static func value(at path: [String], in data: Data) -> Data? {
        var scanner = Scanner(bytes: [UInt8](data))
        return scanner.find(path[...])
    }

    static func string(at path: [String], in data: Data) -> String? {
        value(at: path, in: data).flatMap { String(data: $0, encoding: .utf8) }
    }

    private struct Scanner {
        let bytes: [UInt8]
        var index = 0

        mutating func find(_ path: ArraySlice<String>) -> Data? {
            skipWhitespace()
            guard let wanted = path.first else {
                let start = index
                guard skipValue() else { return nil }
                return Data(bytes[start..<index])
            }
            guard peek() == UInt8(ascii: "{") else { return nil }
            index += 1
            while true {
                skipWhitespace()
                if peek() == UInt8(ascii: "}") { return nil }
                guard let key = readString() else { return nil }
                skipWhitespace()
                guard peek() == UInt8(ascii: ":") else { return nil }
                index += 1
                if key == wanted { return find(path.dropFirst()) }
                guard skipValue() else { return nil }
                skipWhitespace()
                switch peek() {
                case UInt8(ascii: ","): index += 1
                default: return nil   // "}" — the key was not here — or malformed
                }
            }
        }

        // MARK: Primitives

        func peek() -> UInt8? { index < bytes.count ? bytes[index] : nil }

        mutating func skipWhitespace() {
            while let byte = peek(), byte == 0x20 || byte == 0x09 || byte == 0x0A || byte == 0x0D {
                index += 1
            }
        }

        /// Skips one complete value of any kind, leaving `index` just past it.
        mutating func skipValue() -> Bool {
            skipWhitespace()
            guard let first = peek() else { return false }
            switch first {
            case UInt8(ascii: "\""):
                return skipString()
            case UInt8(ascii: "{"), UInt8(ascii: "["):
                // Track depth across both bracket kinds, stepping over strings whole so that a
                // brace inside a description cannot unbalance the count.
                var depth = 0
                while let byte = peek() {
                    switch byte {
                    case UInt8(ascii: "\""):
                        guard skipString() else { return false }
                        continue
                    case UInt8(ascii: "{"), UInt8(ascii: "["):
                        depth += 1
                    case UInt8(ascii: "}"), UInt8(ascii: "]"):
                        depth -= 1
                        if depth == 0 { index += 1; return true }
                    default:
                        break
                    }
                    index += 1
                }
                return false
            default:
                // Number, true, false or null: runs until a structural character or whitespace.
                let start = index
                while let byte = peek(),
                      byte != UInt8(ascii: ","), byte != UInt8(ascii: "}"), byte != UInt8(ascii: "]"),
                      byte != 0x20, byte != 0x09, byte != 0x0A, byte != 0x0D {
                    index += 1
                }
                return index > start
            }
        }

        mutating func skipString() -> Bool {
            guard peek() == UInt8(ascii: "\"") else { return false }
            index += 1
            while let byte = peek() {
                index += 1
                if byte == UInt8(ascii: "\\") { index += 1; continue }   // escaped character
                if byte == UInt8(ascii: "\"") { return true }
            }
            return false
        }

        /// Reads a string and returns its decoded value, for comparing keys.
        mutating func readString() -> String? {
            let start = index
            guard skipString() else { return nil }
            let literal = Data(bytes[start..<index])
            // The literal is valid JSON on its own, so the standard decoder handles escapes —
            // including \u sequences and surrogate pairs — without a second implementation.
            return try? JSONDecoder().decode(String.self, from: literal)
        }
    }
}
