import Foundation

/// One served request, as it appears in the live log.
struct RequestLogEntry: Identifiable, Codable, Sendable, Equatable {
    let id: UUID
    let timestamp: Date
    let path: String
    let status: Int
    let model: String?
    let promptTokens: Int
    let completionTokens: Int
    let prefillMs: Int
    let decodeMs: Int
    let totalMs: Int
    let imageCount: Int
    let hadGrammar: Bool
    let queueDepthOnArrival: Int
    let footprintBytes: UInt64
    let thermalState: String
    let error: String?
    /// Request and response bodies with image payloads replaced by their size.
    let requestBody: String
    let responseBody: String

    var prefillTokensPerSecond: Double {
        prefillMs > 0 ? Double(promptTokens) * 1000 / Double(prefillMs) : 0
    }
    var decodeTokensPerSecond: Double {
        decodeMs > 0 ? Double(completionTokens) * 1000 / Double(decodeMs) : 0
    }

    /// The single line shown in the list.
    var summaryLine: String {
        let time = RequestLogEntry.timeFormatter.string(from: timestamp)
        let tokens = "\(promptTokens)→\(completionTokens)"
        let speed = String(format: "%.1f/%.1f tok/s", prefillTokensPerSecond, decodeTokensPerSecond)
        let seconds = String(format: "%.1fs", Double(totalMs) / 1000)
        return "\(time)  \(status)  \(tokens)  \(seconds)  \(speed)"
    }

    static let timeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f
    }()
}

/// Live request log: a bounded window in memory for the UI, and an unbounded JSONL file on
/// disk for everything else.
///
/// The two have different jobs. The window is what an operator scrolls through; the file is
/// what gets copied off the device after a run that nobody watched. Keeping the whole
/// history in memory instead would compete with the model for the only resource that
/// actually constrains this app.
@MainActor
@Observable
final class RequestLog {

    /// Newest first. Five hundred entries is a few hours of batch traffic and a few
    /// megabytes of text.
    private(set) var entries: [RequestLogEntry] = []
    private(set) var servedCount = 0
    private(set) var failedCount = 0
    private(set) var peakFootprintBytes: UInt64 = 0

    /// Rolling window of generation speed, for the chart that makes thermal throttling
    /// visible. Nothing else in the app would show a device that has quietly halved its
    /// clock.
    private(set) var decodeRateHistory: [Double] = []

    var persistBodies = true

    private let maxEntries = 500
    private let maxRateSamples = 120
    private var writer: JSONLWriter?

    init() {
        writer = try? JSONLWriter(directory: RequestLog.logDirectory())
    }

    func record(_ entry: RequestLogEntry) {
        entries.insert(entry, at: 0)
        if entries.count > maxEntries { entries.removeLast(entries.count - maxEntries) }

        if entry.error == nil { servedCount += 1 } else { failedCount += 1 }
        peakFootprintBytes = max(peakFootprintBytes, entry.footprintBytes)

        if entry.decodeTokensPerSecond > 0 {
            decodeRateHistory.append(entry.decodeTokensPerSecond)
            if decodeRateHistory.count > maxRateSamples {
                decodeRateHistory.removeFirst(decodeRateHistory.count - maxRateSamples)
            }
        }

        writer?.append(entry, includeBodies: persistBodies)
    }

    func clear() {
        entries.removeAll()
        decodeRateHistory.removeAll()
    }

    var logFileURL: URL? { writer?.url }

    /// The whole in-memory window as JSONL, for the copy and share buttons.
    func exportText() -> String {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return entries.reversed().compactMap { entry in
            (try? encoder.encode(entry)).flatMap { String(data: $0, encoding: .utf8) }
        }.joined(separator: "\n")
    }

    static func logDirectory() -> URL {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let directory = documents.appendingPathComponent("logs", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
}

/// Appends entries to a JSONL file, rotating it when it grows past a limit.
final class JSONLWriter {
    let url: URL
    private var handle: FileHandle?
    private let encoder = JSONEncoder()
    private let maxBytes = 64 * 1024 * 1024

    init(directory: URL) throws {
        let stamp = ISO8601DateFormatter.filenameSafe.string(from: Date())
        url = directory.appendingPathComponent("requests-\(stamp).jsonl")
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handle = try FileHandle(forWritingTo: url)
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    }

    deinit { try? handle?.close() }

    func append(_ entry: RequestLogEntry, includeBodies: Bool) {
        var record = entry
        if !includeBodies {
            record = RequestLogEntry(id: entry.id, timestamp: entry.timestamp, path: entry.path,
                                     status: entry.status, model: entry.model,
                                     promptTokens: entry.promptTokens,
                                     completionTokens: entry.completionTokens,
                                     prefillMs: entry.prefillMs, decodeMs: entry.decodeMs,
                                     totalMs: entry.totalMs, imageCount: entry.imageCount,
                                     hadGrammar: entry.hadGrammar,
                                     queueDepthOnArrival: entry.queueDepthOnArrival,
                                     footprintBytes: entry.footprintBytes,
                                     thermalState: entry.thermalState, error: entry.error,
                                     requestBody: "", responseBody: "")
        }
        guard let data = try? encoder.encode(record), let handle else { return }
        do {
            if try handle.offset() > UInt64(maxBytes) { return }
            try handle.write(contentsOf: data)
            try handle.write(contentsOf: Data("\n".utf8))
        } catch {
            // A log that cannot be written must not take the request down with it.
        }
    }
}

extension ISO8601DateFormatter {
    static let filenameSafe: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withYear, .withMonth, .withDay, .withTime]
        return f
    }()
}

// MARK: - Redaction

enum BodyRedactor {

    /// Replaces base64 image payloads with their size.
    ///
    /// A single page image is around a hundred kilobytes of base64. Logging it verbatim
    /// would make the log file larger than everything else in the app combined and make
    /// the UI unusable, while adding nothing: what matters about an image in a request log
    /// is that there was one and how big it was.
    static func redact(_ body: String, maxLength: Int = 40_000) -> String {
        var output = ""
        output.reserveCapacity(min(body.count, maxLength))

        var index = body.startIndex
        while let start = body.range(of: "data:image/", range: index..<body.endIndex) {
            output += body[index..<start.lowerBound]
            // Payload ends at the closing quote of the JSON string that carries it.
            let afterPrefix = start.upperBound
            guard let quote = body.range(of: "\"", range: afterPrefix..<body.endIndex) else {
                output += body[start.lowerBound...]
                index = body.endIndex
                break
            }
            let payload = body[start.lowerBound..<quote.lowerBound]
            let mime = payload.prefix(while: { $0 != ";" }).dropFirst("data:".count)
            output += "<\(mime), \(byteCount(payload.count)) base64>"
            index = quote.lowerBound
        }
        output += body[index...]

        if output.count > maxLength {
            return String(output.prefix(maxLength)) + "… [truncated]"
        }
        return output
    }

    static func byteCount(_ bytes: Int) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }
}
