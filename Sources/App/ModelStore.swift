import Foundation

/// A GGUF file on the device.
struct ModelFile: Identifiable, Sendable, Equatable {
    let url: URL
    let sizeBytes: UInt64
    var id: String { url.lastPathComponent }
    var name: String { url.lastPathComponent }

    /// Projectors are recognised by name, which is how every publisher labels them
    /// (`mmproj-model-f16.gguf`, `mmproj-BF16.gguf`). A projector loaded as a language
    /// model fails with a message about tensor names; the name check avoids ever asking.
    var isProjector: Bool { name.lowercased().contains("mmproj") }
}

/// A language model with the projector that belongs to it, which is what the engine loads.
///
/// llama.cpp needs the pair. The runtime the quality numbers came from keeps both halves in
/// one file, so the two-file split is a property of this port and the first thing to get
/// wrong: a model loaded without its projector answers vision requests from the text alone,
/// which reads as a quality regression rather than as a missing file.
struct ModelPair: Identifiable, Sendable, Equatable {
    let model: ModelFile
    let projector: ModelFile?
    var id: String { model.id }
    var name: String { model.url.deletingPathExtension().lastPathComponent }
    var totalBytes: UInt64 { model.sizeBytes + (projector?.sizeBytes ?? 0) }
    var supportsVision: Bool { projector != nil }
}

/// Models in the app's Documents directory.
///
/// Documents, not the bundle: a 3 GB model inside the binary would make every build and
/// every sideload carry it, and `UIFileSharingEnabled` lets a file be dropped in over USB
/// when the network is the slow part.
@MainActor
@Observable
final class ModelStore {

    private(set) var files: [ModelFile] = []
    private(set) var pairs: [ModelPair] = []

    let directory: URL

    init() {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        directory = documents.appendingPathComponent("models", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        refresh()
    }

    func refresh() {
        let contents = (try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles])) ?? []

        // Models dropped straight into Documents by Files are picked up too: telling
        // somebody their file is in the wrong subdirectory is a worse experience than
        // looking in both places.
        let documents = directory.deletingLastPathComponent()
        let loose = (try? FileManager.default.contentsOfDirectory(
            at: documents,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles])) ?? []

        files = (contents + loose)
            .filter { $0.pathExtension.lowercased() == "gguf" }
            .map { url in
                let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
                return ModelFile(url: url, sizeBytes: UInt64(size))
            }
            .sorted { $0.name < $1.name }

        let projectors = files.filter(\.isProjector)
        pairs = files.filter { !$0.isProjector }.map { model in
            ModelPair(model: model, projector: Self.bestProjector(for: model, among: projectors))
        }
    }

    func delete(_ file: ModelFile) {
        try? FileManager.default.removeItem(at: file.url)
        refresh()
    }

    /// Picks the projector most likely to belong to a model.
    ///
    /// Longest shared filename prefix, because that is what distinguishes
    /// `gemma-3-4b-it-mmproj` from `gemma-3-12b-it-mmproj` sitting in the same directory. A
    /// lone projector is used by default: with one of each, the pairing is not in doubt.
    static func bestProjector(for model: ModelFile, among projectors: [ModelFile]) -> ModelFile? {
        guard !projectors.isEmpty else { return nil }
        if projectors.count == 1 { return projectors[0] }
        let modelName = model.name.lowercased()
        return projectors.max { a, b in
            sharedPrefixLength(modelName, a.name.lowercased()) <
            sharedPrefixLength(modelName, b.name.lowercased())
        }
    }

    static func sharedPrefixLength(_ a: String, _ b: String) -> Int {
        zip(a, b).prefix { $0 == $1 }.count
    }
}

/// Known-good downloads, offered as one tap each.
///
/// Every entry is a pair. The default is the model the measurements are about, from a
/// mirror that needs no account: the upstream repository is gated, and a download that
/// stops to ask for a licence acceptance is not something to discover on the device.
enum ModelCatalog {

    struct Entry: Identifiable, Sendable {
        let id: String
        let title: String
        let detail: String
        let modelURL: URL
        let projectorURL: URL?
        let approximateBytes: UInt64
    }

    static let entries: [Entry] = [
        Entry(id: "gemma-3-4b-it-q4km",
              title: "gemma-3-4b-it · Q4_K_M + mmproj",
              detail: "The reference pair. 2.49 GB model, 851 MB projector.",
              modelURL: URL(string: "https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf")!,
              projectorURL: URL(string: "https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/mmproj-model-f16.gguf")!,
              approximateBytes: 3_340_000_000),

        Entry(id: "gemma-3-4b-it-q8",
              title: "gemma-3-4b-it · Q8_0 + mmproj",
              detail: "Same weights at a larger quantisation. For comparing quantisation cost, not for matching the reference numbers.",
              modelURL: URL(string: "https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q8_0.gguf")!,
              projectorURL: URL(string: "https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/mmproj-model-f16.gguf")!,
              approximateBytes: 5_200_000_000),

        Entry(id: "gemma-3-1b-it-q4km",
              title: "gemma-3-1b-it · Q4_K_M (text only)",
              detail: "Small and quick to download. Useful for proving the server end to end before committing to a 3 GB transfer.",
              modelURL: URL(string: "https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf")!,
              projectorURL: nil,
              approximateBytes: 806_000_000),
    ]
}
