import Foundation

/// A parsed HTTP/1.1 request.
struct HTTPRequestMessage: Sendable, Equatable {
    let method: String
    let path: String
    let query: [String: String]
    let headers: [String: String]      // lowercased names
    let body: Data

    func header(_ name: String) -> String? { headers[name.lowercased()] }
}

struct HTTPResponseMessage: Sendable {
    var status: Int
    var reason: String
    var headers: [String: String] = [:]
    var body: Data = Data()

    static func json(_ status: Int, _ body: Data, extraHeaders: [String: String] = [:]) -> HTTPResponseMessage {
        var response = HTTPResponseMessage(status: status, reason: HTTPResponseMessage.reason(for: status))
        response.headers["Content-Type"] = "application/json"
        response.headers.merge(extraHeaders) { _, new in new }
        response.body = body
        return response
    }

    static func text(_ status: Int, _ string: String, contentType: String = "text/plain; charset=utf-8") -> HTTPResponseMessage {
        var response = HTTPResponseMessage(status: status, reason: HTTPResponseMessage.reason(for: status))
        response.headers["Content-Type"] = contentType
        response.body = Data(string.utf8)
        return response
    }

    func serialized(keepAlive: Bool) -> Data {
        var head = "HTTP/1.1 \(status) \(reason)\r\n"
        var headers = self.headers
        headers["Content-Length"] = String(body.count)
        headers["Connection"] = keepAlive ? "keep-alive" : "close"
        for (name, value) in headers.sorted(by: { $0.key < $1.key }) {
            head += "\(name): \(value)\r\n"
        }
        head += "\r\n"
        var data = Data(head.utf8)
        data.append(body)
        return data
    }

    static func reason(for status: Int) -> String {
        switch status {
        case 200: return "OK"
        case 400: return "Bad Request"
        case 404: return "Not Found"
        case 405: return "Method Not Allowed"
        case 411: return "Length Required"
        case 413: return "Payload Too Large"
        case 499: return "Client Closed Request"
        case 500: return "Internal Server Error"
        case 503: return "Service Unavailable"
        default:  return "Status \(status)"
        }
    }
}

/// Incremental HTTP/1.1 request parser.
///
/// Written by hand rather than taken from a server framework. The requirements here are
/// unusual in one respect and plain in every other: a response may take twenty minutes to
/// produce, so nothing in the stack may impose a timeout, and the bodies carry megabyte
/// images. Everything else is one JSON POST at a time. A hand-written parser whose every
/// branch is covered by tests is a smaller risk than a dependency whose timeout defaults
/// have to be discovered and overridden.
///
/// Pure: feed it bytes, ask what it has. All the socket handling lives elsewhere, which is
/// what lets the interesting cases — a header split across two reads, a truncated body, a
/// chunked upload — be tested without a network.
struct HTTPRequestParser {

    enum Outcome: Equatable {
        case incomplete
        case complete(HTTPRequestMessage)
        case failure(status: Int, message: String)
    }

    /// Larger than any plausible request: a page image arrives as roughly 100 KB of base64,
    /// and the limit exists to bound memory, not to express an expectation.
    static let maxBodyBytes = 64 * 1024 * 1024
    static let maxHeadBytes = 256 * 1024

    private(set) var buffer = Data()

    mutating func feed(_ data: Data) {
        buffer.append(data)
    }

    /// Consumes one request from the front of the buffer, if a whole one is there.
    mutating func next() -> Outcome {
        guard let headEnd = Self.findHeadEnd(buffer) else {
            return buffer.count > Self.maxHeadBytes
                ? .failure(status: 413, message: "request head too large")
                : .incomplete
        }

        let headData = buffer[buffer.startIndex..<headEnd.lowerBound]
        guard let head = String(data: headData, encoding: .utf8) else {
            return .failure(status: 400, message: "request head is not valid UTF-8")
        }

        var lines = head.components(separatedBy: "\r\n")
        guard let requestLine = lines.first else {
            return .failure(status: 400, message: "empty request")
        }
        lines.removeFirst()

        let parts = requestLine.split(separator: " ", maxSplits: 2, omittingEmptySubsequences: false)
        guard parts.count >= 2 else {
            return .failure(status: 400, message: "malformed request line")
        }
        let method = String(parts[0]).uppercased()
        let target = String(parts[1])

        var headers: [String: String] = [:]
        for line in lines where !line.isEmpty {
            guard let colon = line.firstIndex(of: ":") else {
                return .failure(status: 400, message: "malformed header line")
            }
            let name = line[line.startIndex..<colon].trimmingCharacters(in: .whitespaces).lowercased()
            let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            headers[name] = value
        }

        if let encoding = headers["transfer-encoding"], encoding.lowercased().contains("chunked") {
            // Refused rather than implemented: the reference client sends Content-Length,
            // and a half-tested chunked path would fail on exactly the large multimodal
            // bodies that matter most.
            return .failure(status: 411, message: "chunked transfer encoding is not supported; send Content-Length")
        }

        let declaredLength = headers["content-length"].flatMap { Int($0) } ?? 0
        guard declaredLength >= 0 else {
            return .failure(status: 400, message: "negative Content-Length")
        }
        guard declaredLength <= Self.maxBodyBytes else {
            return .failure(status: 413, message: "body of \(declaredLength) bytes exceeds the limit of \(Self.maxBodyBytes)")
        }

        let bodyStart = headEnd.upperBound
        let available = buffer.distance(from: bodyStart, to: buffer.endIndex)
        guard available >= declaredLength else { return .incomplete }

        let bodyEnd = buffer.index(bodyStart, offsetBy: declaredLength)
        let body = Data(buffer[bodyStart..<bodyEnd])
        buffer.removeSubrange(buffer.startIndex..<bodyEnd)

        let (path, query) = Self.splitTarget(target)
        return .complete(HTTPRequestMessage(method: method, path: path, query: query,
                                            headers: headers, body: body))
    }

    /// Whether the connection should stay open after this request.
    ///
    /// Default-on for HTTP/1.1, and worth getting right: the client sends a batch of
    /// requests down a pooled connection, and closing after each one would make it
    /// reconnect between every pair of twenty-minute calls.
    static func keepAlive(_ request: HTTPRequestMessage) -> Bool {
        guard let connection = request.header("connection")?.lowercased() else { return true }
        return !connection.contains("close")
    }

    private static func findHeadEnd(_ data: Data) -> Range<Data.Index>? {
        let terminator = Data("\r\n\r\n".utf8)
        return data.range(of: terminator)
    }

    private static func splitTarget(_ target: String) -> (String, [String: String]) {
        guard let mark = target.firstIndex(of: "?") else { return (target, [:]) }
        let path = String(target[target.startIndex..<mark])
        var query: [String: String] = [:]
        for pair in target[target.index(after: mark)...].split(separator: "&") {
            let kv = pair.split(separator: "=", maxSplits: 1)
            let key = String(kv[0]).removingPercentEncoding ?? String(kv[0])
            let value = kv.count > 1 ? (String(kv[1]).removingPercentEncoding ?? String(kv[1])) : ""
            query[key] = value
        }
        return (path, query)
    }
}
