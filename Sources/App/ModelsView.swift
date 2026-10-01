import SwiftUI

/// Model management: what is on the device, what can be downloaded, and what is loaded.
struct ModelsView: View {
    @Environment(ServerController.self) private var controller
    @State private var customURL = ""
    @State private var showToken = false

    var body: some View {
        // The downloader is a `let` on the controller, so the token field binds to the
        // downloader directly; a binding through the controller would need a writable path.
        @Bindable var downloader = controller.downloader

        NavigationStack {
            List {
                Section("On this device") {
                    if controller.store.pairs.isEmpty {
                        Text("No models yet. Download one below, or copy a .gguf file into the app over Files.")
                            .foregroundStyle(.secondary)
                    }
                    ForEach(controller.store.pairs) { pair in
                        ModelRow(pair: pair)
                    }
                    ForEach(controller.store.files.filter(\.isProjector)) { projector in
                        LabeledContent(projector.name) {
                            Text(MemoryProbe.format(projector.sizeBytes)).foregroundStyle(.secondary)
                        }
                        .swipeActions {
                            Button("Delete", role: .destructive) { controller.store.delete(projector) }
                        }
                    }
                    Button("Rescan") { controller.store.refresh() }
                }

                if let error = controller.loadError {
                    Section("Load failed") {
                        Text(error).font(.callout).foregroundStyle(.red)
                    }
                }

                Section("Downloads") {
                    ForEach(controller.downloader.jobs) { job in
                        DownloadRow(job: job)
                    }
                    if !controller.downloader.jobs.isEmpty {
                        Button("Clear finished") { controller.downloader.clearFinished() }
                    }
                }

                Section("Get a model") {
                    ForEach(ModelCatalog.entries) { entry in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(entry.title).font(.headline)
                            Text(entry.detail).font(.footnote).foregroundStyle(.secondary)
                            HStack {
                                Text(MemoryProbe.format(entry.approximateBytes))
                                    .font(.caption).foregroundStyle(.secondary)
                                Spacer()
                                Button("Download") {
                                    controller.downloader.enqueue(entry)
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                        .padding(.vertical, 2)
                    }
                }

                Section("Any Hugging Face URL") {
                    TextField("https://huggingface.co/…/model.gguf", text: $customURL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .font(.system(.footnote, design: .monospaced))
                    Button("Download") {
                        if let url = URL(string: customURL.trimmingCharacters(in: .whitespaces)) {
                            controller.downloader.enqueue(url: url)
                            customURL = ""
                        }
                    }
                    .disabled(URL(string: customURL.trimmingCharacters(in: .whitespaces)) == nil)

                    DisclosureGroup("Token for gated repositories", isExpanded: $showToken) {
                        SecureField("hf_…", text: $downloader.huggingFaceToken)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                        Text("Held in memory only, never written to disk. The catalogue entries above do not need it.")
                            .font(.footnote).foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Models")
        }
    }
}

private struct ModelRow: View {
    @Environment(ServerController.self) private var controller
    let pair: ModelPair

    var body: some View {
        let isLoaded = controller.selectedPair?.id == pair.id

        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(pair.name).font(.headline)
                Spacer()
                if isLoaded {
                    Text("loaded").font(.caption).foregroundStyle(.green)
                }
            }
            Text(MemoryProbe.format(pair.totalBytes) +
                 (pair.supportsVision ? " · with projector" : " · text only, no projector"))
                .font(.footnote)
                .foregroundStyle(pair.supportsVision ? .secondary : .orange)

            HStack {
                Button(isLoaded ? "Unload" : "Load") {
                    Task {
                        if isLoaded { await controller.unloadModel() }
                        else { await controller.load(pair) }
                    }
                }
                .buttonStyle(.bordered)
                .disabled(controller.isLoadingModel || controller.isRunning)

                if controller.isLoadingModel {
                    ProgressView().controlSize(.small)
                }
            }
        }
        .padding(.vertical, 2)
        .swipeActions {
            Button("Delete", role: .destructive) { controller.store.delete(pair.model) }
                .disabled(isLoaded)
        }
    }
}

private struct DownloadRow: View {
    @Environment(ServerController.self) private var controller
    let job: DownloadJob

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(job.destinationName).font(.footnote).lineLimit(1).truncationMode(.middle)
            ProgressView(value: job.fractionComplete)
            HStack {
                Text(job.statusText).font(.caption).foregroundStyle(.secondary)
                Spacer()
                switch job.state {
                case .running:
                    Button("Pause") { controller.downloader.pause(job.id) }
                case .paused, .failed:
                    Button("Resume") { controller.downloader.resume(job.id) }
                case .finished:
                    Button("Use") { controller.store.refresh() }
                case .waiting:
                    EmptyView()
                }
                Button(role: .destructive) { controller.downloader.cancel(job.id) } label: {
                    Image(systemName: "xmark.circle")
                }
            }
            .buttonStyle(.borderless)
            .font(.caption)
        }
        .padding(.vertical, 2)
    }
}
