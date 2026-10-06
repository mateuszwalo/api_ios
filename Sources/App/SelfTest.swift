import Foundation
import UIKit
import LlamaBridge

/// On-device diagnostics, reachable as `GET /v1/selftest`.
///
/// It exists because of the shape of this project: whoever writes the code cannot run it,
/// and the round trip from a suspected problem to an answer is a CI build, a sideload and
/// somebody picking up the iPad. The checks here are the ones that would otherwise consume
/// that round trip — the chat template, the image token count, and whether each schema in
/// use compiles to a grammar — and they answer in one request with one JSON to paste back.
struct SelfTest: Sendable {

    let engine: LlamaInferenceEngine
    let grammar: GrammarCompiler

    func run() async -> Data {
        // `?? NSNull()` would not type-check against a `String?`: the two sides of the
        // operator must agree, and `Any` is not inferred through it.
        let modelName: Any = await engine.loadedModelName.map { $0 as Any } ?? NSNull()

        var report: [String: Any] = [
            "generated_at": ISO8601DateFormatter().string(from: Date()),
            "model_loaded": await engine.isLoaded,
            "model": modelName,
            "context_length": await engine.contextLength,
            "supports_images": await engine.supportsImages,
            "footprint_bytes": MemoryProbe.footprintBytes(),
            "available_memory_bytes": MemoryProbe.availableBytes(),
            "thermal_state": MemoryProbe.thermalStateName(),
        ]

        // The engine's own words, last lines first in usefulness: when a load fails this is
        // where the reason is, and it is the only part of this report that needs no model.
        report["engine_log"] = LLMBridge.recentEngineLog().suffix(60)

        report["chat_template"] = await checkChatTemplate()
        report["image_tokens"] = await checkImageTokens()
        report["grammar"] = checkGrammars()
        report["constrained_generation"] = await checkConstrainedGeneration()

        let options: JSONSerialization.WritingOptions = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return (try? JSONSerialization.data(withJSONObject: report, options: options))
            ?? Data("{\"error\":\"could not serialise the report\"}".utf8)
    }

    // MARK: Checks

    /// The template is the quietest way for this port to diverge from the runtime the
    /// quality numbers came from, so the rendered text is reported verbatim rather than
    /// summarised into a verdict.
    private func checkChatTemplate() async -> [String: Any] {
        let prompt = EnginePrompt(turns: [
            .init(role: "system", text: "You are a helpful assistant.", images: []),
            .init(role: "user", text: "Say OK.", images: []),
        ])
        guard let rendered = await engine.renderedPromptText(prompt) else {
            return ["ok": false, "detail": "no model is loaded"]
        }
        let expected = ["<start_of_turn>", "<end_of_turn>"]
        let present = expected.filter { rendered.contains($0) }
        return [
            "ok": present.count == expected.count,
            "rendered": rendered,
            "expected_markers": expected,
            "markers_found": present,
            "note": "Gemma 3 turn markers. Their absence means the GGUF carries a different template than the reference runtime applies.",
        ]
    }

    /// 256 tokens per image means one tile and no pan & scan — the configuration the
    /// reference measurements were taken in. A larger count is the single clearest sign
    /// that the vision path has been configured differently.
    private func checkImageTokens() async -> [String: Any] {
        guard await engine.supportsImages else {
            return ["ok": false, "detail": "no projector is loaded; vision requests will be refused"]
        }
        guard let jpeg = Self.probeImage() else {
            return ["ok": false, "detail": "could not synthesise a probe image"]
        }
        let withImage = EnginePrompt(turns: [
            .init(role: "user", text: "What is in this image?",
                  images: [ImagePayload(data: jpeg, mimeType: "image/jpeg")]),
        ])
        let withoutImage = EnginePrompt(turns: [
            .init(role: "user", text: "What is in this image?", images: []),
        ])
        do {
            let total = try await engine.measurePrompt(withImage)
            let text = try await engine.measurePrompt(withoutImage)
            let perImage = total - text
            return [
                "ok": perImage == 256,
                "prompt_tokens_with_image": total,
                "prompt_tokens_text_only": text,
                "tokens_attributable_to_image": perImage,
                "expected": 256,
                "note": "256 means one tile, pan & scan off. A multiple of 256 means pan & scan is active and token counts will not match the reference.",
            ]
        } catch {
            return ["ok": false, "detail": "\(error)"]
        }
    }

    /// Compiles every schema the device has: the two bundled fixtures, plus anything
    /// dropped into `Documents/selftest/` over Files. That last part is the point — the
    /// schemas that matter are not in this repository, and this is how they get tested
    /// against the converter without ever being committed.
    private func checkGrammars() -> [String: Any] {
        var results: [[String: Any]] = []

        for name in ["schema_complex", "schema_simple"] {
            if let url = Bundle.main.url(forResource: name, withExtension: "json"),
               let text = try? String(contentsOf: url, encoding: .utf8) {
                results.append(compile(name: name + ".json", schema: text))
            }
        }

        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let directory = documents.appendingPathComponent("selftest", isDirectory: true)
        let extras = (try? FileManager.default.contentsOfDirectory(at: directory,
                                                                   includingPropertiesForKeys: nil)) ?? []
        for url in extras where url.pathExtension.lowercased() == "json" {
            if let text = try? String(contentsOf: url, encoding: .utf8) {
                results.append(compile(name: url.lastPathComponent, schema: text))
            }
        }

        return [
            "ok": results.allSatisfy { ($0["ok"] as? Bool) == true },
            "schemas": results,
            "hint": "Drop production schemas into the app's Documents/selftest/ folder over Files to have them checked here.",
        ]
    }

    private func compile(name: String, schema: String) -> [String: Any] {
        do {
            let gbnf = try grammar.compile(schema)
            return ["name": name, "ok": true, "grammar_bytes": gbnf.utf8.count,
                    "grammar_head": String(gbnf.prefix(400))]
        } catch {
            return ["name": name, "ok": false, "error": "\(error)"]
        }
    }

    /// Generates a few tokens under a grammar and reports what came out.
    ///
    /// Short on purpose: this runs on the same single slot as real traffic, and a self-test
    /// that occupies the model for a minute is one nobody will run while it matters.
    private func checkConstrainedGeneration() async -> [String: Any] {
        guard await engine.isLoaded else { return ["ok": false, "detail": "no model is loaded"] }
        guard let url = Bundle.main.url(forResource: "schema_simple", withExtension: "json"),
              let schema = try? String(contentsOf: url, encoding: .utf8),
              let gbnf = try? grammar.compile(schema) else {
            return ["ok": false, "detail": "the fixture schema did not compile"]
        }

        let request = EngineRequest(
            prompt: EnginePrompt(turns: [
                .init(role: "user",
                      text: "Return a JSON object with a \"matches\" array holding one entry with item_id 1.",
                      images: []),
            ]),
            grammar: gbnf, maxTokens: 128, temperature: 0, stopSequences: [], seed: nil)

        do {
            let result = try await engine.generate(request)
            let parsed = (try? JSONSerialization.jsonObject(with: Data(result.text.utf8))) != nil
            return [
                "ok": parsed,
                "output": result.text,
                "parses_as_json": parsed,
                "prompt_tokens": result.promptTokens,
                "completion_tokens": result.completionTokens,
                "prefill_tps": result.prefillTokensPerSecond,
                "decode_tps": result.decodeTokensPerSecond,
            ]
        } catch {
            return ["ok": false, "detail": "\(error)"]
        }
    }

    /// A small solid-colour JPEG. Gemma emits the same token count for any image, so the
    /// content is irrelevant and synthesising one avoids shipping a test asset.
    static func probeImage() -> Data? {
        let size = CGSize(width: 64, height: 64)
        let renderer = UIGraphicsImageRenderer(size: size)
        let image = renderer.image { context in
            UIColor.systemTeal.setFill()
            context.fill(CGRect(origin: .zero, size: size))
        }
        return image.jpegData(compressionQuality: 0.8)
    }
}
