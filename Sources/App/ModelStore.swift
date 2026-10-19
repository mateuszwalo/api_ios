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
    /// True when the projector was paired by elimination rather than by evidence.
    ///
    /// Published projectors are called `mmproj-model-f16.gguf` and share nothing with the
    /// model's filename, so with one projector and several models there is no way to tell
    /// from names alone which it belongs to. Loading the wrong one does not fail politely:
    /// mtmd aborts and takes the app with it. When the pairing is a guess, vision starts
    /// switched off and the person loading the model decides.
    var projectorIsAmbiguous = false
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
        let models = files.filter { !$0.isProjector }
        pairs = models.map { model in
            // The catalogue knows what it published, so a model it lists is paired by fact:
            // the 1B entry has no projector at all, and the 4B entry names its own. That
            // settles the case that terminated the app — a 4B vision tower attached to the 1B
            // text model — and it does so for files downloaded before this rule existed,
            // because it reads names the catalogue already holds.
            if let entry = ModelCatalog.entry(forModelFile: model.name) {
                guard let projectorURL = entry.projectorURL else {
                    return ModelPair(model: model, projector: nil)
                }
                // The stored name, or for the original entries the bare one that builds before
                // 1.2 saved it under. Not for newer entries: a bare `mmproj-F16.gguf` could be
                // any model's, and a mismatched projector aborts the app.
                var names = [ModelCatalog.localFileName(for: projectorURL)]
                if entry.acceptsBareProjectorName { names.append(projectorURL.lastPathComponent) }
                if let exact = names.lazy.compactMap({ name in projectors.first { $0.name == name } }).first {
                    return ModelPair(model: model, projector: exact, projectorIsAmbiguous: false)
                }
                // Not the catalogue's own projector file, but one that names this model: the
                // BF16 variant, or the same projector from another publisher. Only a match by
                // name; guessing is what the catalogue rule exists to prevent.
                return ModelPair(model: model, projector: Self.namedProjector(for: model, among: projectors),
                                 projectorIsAmbiguous: false)
            }

            // A file from outside the catalogue: pair by name, and say when it is a guess.
            if let named = Self.namedProjector(for: model, among: projectors) {
                return ModelPair(model: model, projector: named, projectorIsAmbiguous: false)
            }
            let projector = Self.bestProjector(for: model, among: projectors)
            // One projector and one model is unambiguous whatever they are called. More than
            // one model and nothing in the names to go on is a guess, and says so.
            let ambiguous = projector != nil && models.count > 1
            return ModelPair(model: model, projector: projector, projectorIsAmbiguous: ambiguous)
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

    /// The projector whose name says it belongs to this model, or nil when none clearly does.
    ///
    /// Names are compared by letters and digits only, with "mmproj" taken out, so
    /// `mmproj-gemma-4-E2B-it-Q8_0`, `gemma-4-E2B-it-mmproj-F16` and `gemma-4-E2B-it-Q4_K_M` all
    /// start `gemma4e2bit`. Ten characters in common is the bar: `gemma34bit` passes, while a
    /// 4B model and a 12B projector share only `gemma3`. A tie between two projectors is no
    /// answer either.
    static func namedProjector(for model: ModelFile, among projectors: [ModelFile]) -> ModelFile? {
        let modelKey = nameKey(model.name)
        let scored = projectors.map { ($0, sharedPrefixLength(modelKey, nameKey($0.name))) }
        guard let best = scored.max(by: { $0.1 < $1.1 }), best.1 >= 10,
              scored.filter({ $0.1 == best.1 }).count == 1 else { return nil }
        return best.0
    }

    /// A file name reduced to what identifies the model: lowercase letters and digits, without
    /// the extension and without "mmproj".
    static func nameKey(_ fileName: String) -> String {
        var name = fileName.lowercased()
        if name.hasSuffix(".gguf") { name.removeLast(5) }
        name = name.replacingOccurrences(of: "mmproj", with: "")
        return String(name.unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) }.map(Character.init))
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
        /// Entries from before 1.2, whose projectors may be on the device under the bare name.
        var acceptsBareProjectorName = false
    }

    /// The name a download is saved under.
    ///
    /// The file's own name, except for a projector whose name does not say which model it
    /// belongs to. Publishers call them `mmproj-F16.gguf` in every repository, so a second
    /// download would overwrite the first, and pairing by name would be a guess. Those are
    /// prefixed with the repository's name: `unsloth/gemma-4-E2B-it-GGUF/…/mmproj-F16.gguf`
    /// is saved as `gemma-4-E2B-it-mmproj-F16.gguf`, next to `gemma-4-E2B-it-Q4_K_M.gguf`.
    /// Hugging Face's page link for a file (`/blob/`) serves HTML, not the file. Pasting it is
    /// the usual mistake, so it is turned into the download link (`/resolve/`).
    static func downloadURL(for url: URL) -> URL {
        guard url.host?.lowercased().hasSuffix("huggingface.co") == true,
              url.path.contains("/blob/"),
              var parts = URLComponents(url: url, resolvingAgainstBaseURL: false) else { return url }
        parts.path = parts.path.replacingOccurrences(of: "/blob/", with: "/resolve/")
        return parts.url ?? url
    }

    static func localFileName(for url: URL) -> String {
        let file = url.lastPathComponent
        let parts = url.pathComponents   // ["/", org, repo, "resolve", revision, …, file]
        guard file.lowercased().contains("mmproj"),
              url.host?.lowercased().hasSuffix("huggingface.co") == true,
              parts.count >= 6, parts[3] == "resolve" else { return file }
        var stem = parts[2]
        for suffix in ["-GGUF", "-gguf", "_GGUF", "_gguf"] where stem.hasSuffix(suffix) {
            stem = String(stem.dropLast(suffix.count))
        }
        guard !stem.isEmpty, !file.lowercased().contains(stem.lowercased()) else { return file }
        return "\(stem)-\(file)"
    }

    /// The catalogue entry a downloaded model file came from, matched by file name.
    static func entry(forModelFile name: String) -> Entry? {
        entries.first { $0.modelURL.lastPathComponent.caseInsensitiveCompare(name) == .orderedSame }
    }

    static let entries: [Entry] = [
        Entry(id: "gemma-3-4b-it-q4km",
              title: "gemma-3-4b-it · Q4_K_M + mmproj",
              detail: "The reference pair. 2.49 GB model, 851 MB projector.",
              modelURL: URL(string: "https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf")!,
              projectorURL: URL(string: "https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/mmproj-model-f16.gguf")!,
              approximateBytes: 3_340_000_000,
              acceptsBareProjectorName: true),

        Entry(id: "gemma-3-4b-it-q8",
              title: "gemma-3-4b-it · Q8_0 + mmproj",
              detail: "Same weights at a larger quantisation. For comparing quantisation cost, not for matching the reference numbers.",
              modelURL: URL(string: "https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q8_0.gguf")!,
              projectorURL: URL(string: "https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/mmproj-model-f16.gguf")!,
              approximateBytes: 5_200_000_000,
              acceptsBareProjectorName: true),

        Entry(id: "gemma-4-e2b-it-q4km",
              title: "gemma-4-E2B-it · Q4_K_M + mmproj",
              detail: "Gemma 4, prompted through its own Jinja template. 3.11 GB model, 986 MB projector.",
              modelURL: URL(string: "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf")!,
              projectorURL: URL(string: "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/mmproj-F16.gguf")!,
              approximateBytes: 4_092_000_000),

        Entry(id: "gemma-3-1b-it-q4km",
              title: "gemma-3-1b-it · Q4_K_M (text only)",
              detail: "Small and quick to download. Useful for proving the server end to end before committing to a 3 GB transfer.",
              modelURL: URL(string: "https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf")!,
              projectorURL: nil,
              approximateBytes: 806_000_000),
    ]
}
