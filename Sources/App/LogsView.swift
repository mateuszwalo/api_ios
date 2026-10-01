import SwiftUI

/// Live request log: one line per request, expandable to the bodies.
struct LogsView: View {
    @Environment(ServerController.self) private var controller
    @State private var shareItem: ShareItem?

    var body: some View {
        // Bound to the log rather than to the controller: `log` is a `let`, so a binding
        // through the controller would need a writable key path it does not have.
        @Bindable var log = controller.log

        NavigationStack {
            List {
                if log.entries.isEmpty {
                    ContentUnavailableView("No requests yet",
                                           systemImage: "list.bullet.rectangle",
                                           description: Text("Served requests appear here as they complete."))
                }
                ForEach(log.entries) { entry in
                    NavigationLink {
                        LogDetailView(entry: entry)
                    } label: {
                        LogRow(entry: entry)
                    }
                }
            }
            .navigationTitle("Requests")
            .toolbar {
                ToolbarItemGroup(placement: .topBarTrailing) {
                    Button {
                        UIPasteboard.general.string = log.exportText()
                    } label: {
                        Image(systemName: "doc.on.doc")
                    }
                    Button {
                        if let url = log.logFileURL { shareItem = ShareItem(url: url) }
                    } label: {
                        Image(systemName: "square.and.arrow.up")
                    }
                    Menu {
                        Toggle("Write bodies to the log file", isOn: $log.persistBodies)
                        Button("Clear list", role: .destructive) { log.clear() }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                }
            }
            .sheet(item: $shareItem) { item in
                ShareSheet(url: item.url)
            }
        }
    }
}

private struct LogRow: View {
    let entry: RequestLogEntry

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 8) {
                Text(RequestLogEntry.timeFormatter.string(from: entry.timestamp))
                    .font(.system(.caption, design: .monospaced))
                Text("\(entry.status)")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(entry.status == 200 ? .green : .red)
                Text("\(entry.promptTokens)→\(entry.completionTokens) tok")
                    .font(.system(.caption, design: .monospaced))
                Spacer()
                if entry.imageCount > 0 {
                    Image(systemName: "photo").font(.caption2).foregroundStyle(.secondary)
                }
                if entry.hadGrammar {
                    Image(systemName: "curlybraces").font(.caption2).foregroundStyle(.secondary)
                }
            }
            Text(String(format: "%.1fs · prefill %.1f tok/s · decode %.1f tok/s",
                        Double(entry.totalMs) / 1000,
                        entry.prefillTokensPerSecond,
                        entry.decodeTokensPerSecond))
                .font(.caption2)
                .foregroundStyle(.secondary)
            if let error = entry.error {
                Text(error).font(.caption2).foregroundStyle(.red).lineLimit(2)
            }
        }
    }
}

struct LogDetailView: View {
    let entry: RequestLogEntry

    var body: some View {
        List {
            Section("Timing") {
                LabeledContent("Total", value: String(format: "%.2f s", Double(entry.totalMs) / 1000))
                LabeledContent("Prefill", value: String(format: "%.2f s (%.1f tok/s)",
                                                        Double(entry.prefillMs) / 1000,
                                                        entry.prefillTokensPerSecond))
                LabeledContent("Generation", value: String(format: "%.2f s (%.1f tok/s)",
                                                           Double(entry.decodeMs) / 1000,
                                                           entry.decodeTokensPerSecond))
                LabeledContent("Queue on arrival", value: "\(entry.queueDepthOnArrival)")
            }
            Section("Request") {
                LabeledContent("Status", value: "\(entry.status)")
                LabeledContent("Model", value: entry.model ?? "—")
                LabeledContent("Tokens", value: "\(entry.promptTokens) in, \(entry.completionTokens) out")
                LabeledContent("Images", value: "\(entry.imageCount)")
                LabeledContent("Schema enforced", value: entry.hadGrammar ? "yes" : "no")
                LabeledContent("Footprint", value: MemoryProbe.format(entry.footprintBytes))
                LabeledContent("Thermal", value: entry.thermalState)
            }
            if let error = entry.error {
                Section("Error") { Text(error).foregroundStyle(.red) }
            }
            Section("Request body") {
                BodyText(text: entry.requestBody)
            }
            Section("Response body") {
                BodyText(text: entry.responseBody)
            }
        }
        .navigationTitle(RequestLogEntry.timeFormatter.string(from: entry.timestamp))
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            Button {
                UIPasteboard.general.string = entry.requestBody + "\n\n" + entry.responseBody
            } label: {
                Image(systemName: "doc.on.doc")
            }
        }
    }
}

private struct BodyText: View {
    let text: String

    var body: some View {
        if text.isEmpty {
            Text("empty").foregroundStyle(.secondary)
        } else {
            Text(text)
                .font(.system(.caption2, design: .monospaced))
                .textSelection(.enabled)
        }
    }
}

struct ShareItem: Identifiable {
    let url: URL
    var id: String { url.path }
}

struct ShareSheet: UIViewControllerRepresentable {
    let url: URL
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: [url], applicationActivities: nil)
    }
    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}
