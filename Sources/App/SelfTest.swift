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
            "gpu_allocated_bytes": MemoryProbe.gpuAllocatedBytes(),
            "app_version": AppVersion.description,
        ]

        // The engine's own words, last lines first in usefulness: when a load fails this is
        // where the reason is, and it is the only part of this report that needs no model.
        let fullLog = LLMBridge.recentEngineLog()
        report["engine_log"] = Array(fullLog.suffix(80))
        // The milestones and crash records alone, from the whole retained log. A model load
        // prints hundreds of lines of its own, which push the few that matter — what was
        // being loaded, what was asked for, how a process died — out of any short tail.
        report["engine_events"] = Self.importantLines(fullLog)

        report["chat_template"] = await checkChatTemplate()
        report["image_tokens"] = await checkImageTokens()
        report["grammar"] = checkGrammars()
        report["constrained_generation"] = await checkConstrainedGeneration()

        // Sanitised and validated before serialising, never handed over raw.
        //
        // JSONSerialization does not throw on a value it cannot encode: it raises an
        // Objective-C exception, which `try?` does not catch, and the process dies. An
        // ArraySlice in this report did exactly that on every call — the self-test, the one
        // tool for finding out why the app crashes, was itself crashing the app.
        let safe = Self.jsonSafe(report)
        let options: JSONSerialization.WritingOptions = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        guard JSONSerialization.isValidJSONObject(safe),
              let data = try? JSONSerialization.data(withJSONObject: safe, options: options) else {
            return Data("{\"error\":\"the self-test report could not be serialised\"}".utf8)
        }
        return data
    }

    /// Application milestones, launch markers and complete crash records, in order.
    static func importantLines(_ log: [String]) -> [String] {
        var kept: [String] = []
        var inCrashRecord = false
        for line in log {
            if line.hasPrefix("FATAL") { inCrashRecord = true }
            if inCrashRecord || line.hasPrefix("APP:") || line.hasPrefix("=====") {
                kept.append(line)
            }
            if line.contains("end of crash record") { inCrashRecord = false }
        }
        return Array(kept.suffix(150))
    }

    /// Reduces any value to the handful of types JSONSerialization accepts.
    ///
    /// Anything unrecognised becomes its description rather than reaching the serialiser,
    /// and non-finite numbers become null: a NaN tokens-per-second figure is another value
    /// JSONSerialization answers with an exception rather than an error.
    static func jsonSafe(_ value: Any) -> Any {
        switch value {
        case is NSNull:                   return value
        case let v as String:             return v
        case let v as Bool:               return v
        case let v as Int:                return v
        case let v as Int64:              return NSNumber(value: v)
        case let v as UInt64:             return NSNumber(value: v)
        case let v as Double:             return v.isFinite ? v : NSNull()
        case let v as Float:              return v.isFinite ? Double(v) : NSNull()
        case let v as [String: Any]:      return v.mapValues { jsonSafe($0) }
        case let v as [Any]:              return v.map { jsonSafe($0) }
        case let v as ArraySlice<String>: return Array(v)
        case let v as NSNumber:           return v
        default:                          return String(describing: value)
        }
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
        guard await engine.isLoaded else {
            return ["ok": false, "detail": "no model is loaded"]
        }
        // Any model, not just Gemma 3: the check is that the model's template renders at all.
        // Checking for Gemma 3's turn markers made every other model fail this test while
        // working perfectly. The rendered text is still returned in full, because a template
        // that renders but renders differently from the reference is invisible otherwise.
        guard let rendered = await engine.renderedPromptText(prompt), !rendered.isEmpty else {
            // Measuring surfaces the reason the template produced nothing.
            var reason = "the chat template produced no prompt"
            do { _ = try await engine.measurePrompt(prompt) } catch { reason = "\(error)" }
            return ["ok": false, "detail": reason, "engine": await engine.chatTemplateStatus]
        }
        var thinkingOff = prompt
        thinkingOff.disableThinking = true
        let renderedThinkingOff = await engine.renderedPromptText(thinkingOff) ?? ""

        // The same conversation through both engines. Before 1.2 every prompt went through the
        // built-in formats, so this difference is what prompt_tokens shows against those builds.
        var comparison: [String: Any] = [:]
        if let c = await engine.templateComparison(prompt) {
            comparison = [
                "jinja_tokens": c.jinjaTokens,
                "builtin_tokens": c.builtinTokens,
                "same_tokens": c.jinjaTokens >= 0 && c.jinjaTokens == c.builtinTokens,
                "same_text": c.jinjaPrompt != nil && c.jinjaPrompt == c.builtinPrompt,
                "builtin_prompt": c.builtinPrompt.map { $0 as Any } ?? NSNull(),
                "jinja_error": c.jinjaError.map { $0 as Any } ?? NSNull(),
                "builtin_error": c.builtinError.map { $0 as Any } ?? NSNull(),
                "note": "builtin is how versions before 1.2 built every prompt. builtin_tokens -1 means "
                    + "the old versions could not prompt this model at all (Gemma 4, for one).",
            ]
        }
        return [
            "ok": true,
            "engine": await engine.chatTemplateStatus,
            "rendered": rendered,
            "rendered_with_thinking_off": renderedThinkingOff,
            "thinking_switch_changes_prompt": renderedThinkingOff != rendered,
            "versus_previous_versions": comparison,
            "note": "rendered is what the model receives for a system + user turn. "
                + "thinking_switch_changes_prompt is true for templates that honour enable_thinking "
                + "(reasoning models); for others it is false and reasoning_effort has no effect.",
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
            // 256 image embeddings plus the few tokens around them: Gemma wraps the image in
            // start- and end-of-image markers, and the prompt puts a newline after it. 259 was
            // measured on the device. Pan & scan would add whole tiles, 256 at a time.
            return [
                "ok": (256...264).contains(perImage),
                "prompt_tokens_with_image": total,
                "prompt_tokens_text_only": text,
                "tokens_attributable_to_image": perImage,
                "expected": "256 embeddings + a few marker tokens (259 measured)",
                "note": "Around 259 means one tile, pan & scan off — the reference configuration. Around 515 or more means pan & scan is active and token counts will not match the reference.",
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
