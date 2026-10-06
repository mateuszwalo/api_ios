import SwiftUI

/// Status screen: the address to point a client at, the start/stop control, and the three
/// numbers that cannot be observed from anywhere else — memory, queue depth and thermal
/// state.
struct ServerView: View {
    @Environment(ServerController.self) private var controller
    @State private var copied: String?

    var body: some View {
        @Bindable var controller = controller

        NavigationStack {
            Form {
                Section("Status") {
                    LabeledContent("Server", value: statusText)
                    if let pair = controller.selectedPair {
                        LabeledContent("Model", value: pair.name)
                        LabeledContent("Vision", value: pair.supportsVision ? "yes (projector loaded)" : "no projector")
                    } else {
                        LabeledContent("Model") {
                            Text("none loaded").foregroundStyle(.secondary)
                        }
                    }
                    LabeledContent("Context", value: "\(controller.contextLength) tokens")
                    LabeledContent("Queue", value: "\(controller.queueDepth) in flight or waiting")
                    LabeledContent("Served", value: "\(controller.log.servedCount) ok, \(controller.log.failedCount) failed")
                }

                Section("Address") {
                    if controller.baseURLs.isEmpty {
                        Text("Start the server to see its address.")
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(controller.baseURLs, id: \.self) { url in
                            Button {
                                UIPasteboard.general.string = url
                                copied = url
                            } label: {
                                HStack {
                                    Text(url).font(.system(.body, design: .monospaced))
                                    Spacer()
                                    Image(systemName: copied == url ? "checkmark" : "doc.on.doc")
                                        .foregroundStyle(.secondary)
                                }
                            }
                            .buttonStyle(.plain)
                        }
                        Text("Point the client's base URL at one of these.")
                            .font(.footnote).foregroundStyle(.secondary)
                    }
                }

                Section("Memory") {
                    LabeledContent("Footprint", value: MemoryProbe.format(controller.footprintBytes))
                    LabeledContent("Peak", value: MemoryProbe.format(controller.log.peakFootprintBytes))
                    LabeledContent("Available", value: MemoryProbe.format(controller.availableBytes))
                    LabeledContent("Thermal", value: controller.thermalState)
                    if controller.thermalState == "serious" || controller.thermalState == "critical" {
                        Text("The device is throttling. Generation speed measured now is not representative.")
                            .font(.footnote).foregroundStyle(.orange)
                    }
                }

                if !controller.log.decodeRateHistory.isEmpty {
                    Section("Generation speed") {
                        Sparkline(values: controller.log.decodeRateHistory)
                            .frame(height: 44)
                        Text("tok/s per request, oldest to newest. A downward slope with a steady workload is thermal throttling.")
                            .font(.footnote).foregroundStyle(.secondary)
                    }
                }

                Section("Settings") {
                    LabeledContent("Port") {
                        TextField("8080", value: $controller.port, format: .number.grouping(.never))
                            .multilineTextAlignment(.trailing)
                            .keyboardType(.numberPad)
                            .disabled(controller.isRunning)
                    }
                    // These take effect at the next load, not immediately, so there is nothing
                    // to protect by locking them while the server runs.
                    Stepper("Context: \(controller.contextLength)",
                            value: $controller.contextLength, in: 2048...131072, step: 2048)
                    Stepper("Batch: \(controller.batchSize)",
                            value: $controller.batchSize, in: 64...4096, step: 64)
                    Toggle("Flash attention", isOn: $controller.flashAttention)
                    Toggle("Memory mapping", isOn: $controller.useMemoryMapping)
                    Toggle("Reuse KV cache between requests", isOn: $controller.reuseKVCache)
                    Text("Context, batch and the toggles apply at the next load. Reuse makes repeated prompts much faster and makes measured prefill depend on what ran before: off for measurement, on for a realistic deployment.")
                        .font(.footnote).foregroundStyle(.secondary)
                }

                Section {
                    Button(controller.isRunning ? "Stop server" : "Start server") {
                        if controller.isRunning { controller.stopServer() } else { controller.startServer() }
                    }
                    .foregroundStyle(controller.isRunning ? .red : .accentColor)
                } footer: {
                    Text("The server runs only while this app is in the foreground and the screen is on; iOS suspends background apps. The screen is kept awake while the server runs.")
                }
            }
            .navigationTitle("LocalLLM Server")
        }
    }

    private var statusText: String {
        switch controller.serverState {
        case .stopped: return "stopped"
        case .starting: return "starting…"
        case .running(let port): return "listening on 0.0.0.0:\(port)"
        case .failed(let why): return "failed: \(why)"
        }
    }
}

/// Minimal line chart. A dependency-free sparkline is enough to show a trend, and a trend
/// is all that is being asked of it.
struct Sparkline: View {
    let values: [Double]

    var body: some View {
        GeometryReader { geometry in
            let maximum = values.max() ?? 1
            let minimum = values.min() ?? 0
            let span = max(maximum - minimum, 0.0001)
            Path { path in
                for (index, value) in values.enumerated() {
                    let x = geometry.size.width * (values.count > 1
                                                   ? Double(index) / Double(values.count - 1) : 0)
                    let y = geometry.size.height * (1 - (value - minimum) / span)
                    if index == 0 { path.move(to: CGPoint(x: x, y: y)) }
                    else { path.addLine(to: CGPoint(x: x, y: y)) }
                }
            }
            .stroke(.tint, lineWidth: 1.5)
            .overlay(alignment: .topLeading) {
                Text(String(format: "%.1f", maximum)).font(.caption2).foregroundStyle(.secondary)
            }
            .overlay(alignment: .bottomLeading) {
                Text(String(format: "%.1f", minimum)).font(.caption2).foregroundStyle(.secondary)
            }
        }
    }
}
