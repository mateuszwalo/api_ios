import Foundation

/// An image extracted from a request, ready for the multimodal tokenizer.
struct ImagePayload: Sendable {
    let data: Data
    let mimeType: String

    /// Accepted image types. Anything else is rejected rather than guessed at: a
    /// mislabelled image reaching the vision encoder fails deep inside llama.cpp with a
    /// message that says nothing about the request that caused it.
    static let allowedMimeTypes: Set<String> = ["image/jpeg", "image/png", "image/webp"]

    /// Reject oversized payloads before base64 decoding rather than after — decoding
    /// first would allocate the full image just to refuse it.
    static let maxEncodedBytes = 64 * 1024 * 1024

    enum ParseError: Error, CustomStringConvertible {
        case remoteURLNotAllowed
        case notADataURI
        case unsupportedMimeType(String)
        case notBase64
        case tooLarge(Int)
        case empty

        var description: String {
            switch self {
            case .remoteURLNotAllowed:
                return "image_url must be a data: URI; this server does not fetch remote images"
            case .notADataURI:
                return "image_url is not a data: URI"
            case .unsupportedMimeType(let m):
                return "unsupported image type \(m); expected one of \(ImagePayload.allowedMimeTypes.sorted().joined(separator: ", "))"
            case .notBase64:
                return "image_url payload is not valid base64"
            case .tooLarge(let n):
                return "image payload of \(n) bytes exceeds the \(ImagePayload.maxEncodedBytes) byte limit"
            case .empty:
                return "image payload is empty"
            }
        }
    }

    /// Parses `data:image/jpeg;base64,<payload>`.
    ///
    /// Remote URLs are refused explicitly instead of falling through to a generic parse
    /// error: a server that fetches URLs on a client's behalf is a different and more
    /// dangerous thing than one that does not, and the distinction deserves its own
    /// message.
    static func parse(_ urlString: String) throws -> ImagePayload {
        let lowered = urlString.lowercased()
        if lowered.hasPrefix("http://") || lowered.hasPrefix("https://") {
            throw ParseError.remoteURLNotAllowed
        }
        guard lowered.hasPrefix("data:") else { throw ParseError.notADataURI }
        guard urlString.utf8.count <= maxEncodedBytes else {
            throw ParseError.tooLarge(urlString.utf8.count)
        }

        // data:[<mediatype>][;base64],<data>
        guard let comma = urlString.firstIndex(of: ",") else { throw ParseError.notADataURI }
        let header = String(urlString[urlString.index(urlString.startIndex, offsetBy: 5)..<comma])
        let payload = String(urlString[urlString.index(after: comma)...])

        let segments = header.split(separator: ";", omittingEmptySubsequences: true).map(String.init)
        guard segments.contains(where: { $0.caseInsensitiveCompare("base64") == .orderedSame }) else {
            throw ParseError.notBase64
        }
        let mime = (segments.first.map { $0.lowercased() } ?? "").isEmpty
            ? "image/jpeg" : segments[0].lowercased()
        guard allowedMimeTypes.contains(mime) else { throw ParseError.unsupportedMimeType(mime) }

        guard !payload.isEmpty else { throw ParseError.empty }
        // Some encoders wrap base64 at column boundaries; the strict decoder rejects the
        // newlines outright, so allow them explicitly rather than silently failing.
        guard let data = Data(base64Encoded: payload, options: [.ignoreUnknownCharacters]) else {
            throw ParseError.notBase64
        }
        guard !data.isEmpty else { throw ParseError.empty }
        return ImagePayload(data: data, mimeType: mime)
    }
}
