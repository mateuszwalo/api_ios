import XCTest
import LlamaBridge

/// Tests for the JSON Schema → GBNF conversion.
///
/// This is the component most likely to be wrong and least likely to announce it. A grammar
/// that is merely too permissive still produces well-formed JSON, so nothing fails; the
/// output simply stops being guaranteed, and every number taken afterwards is suspect.
///
/// What can be checked here and what cannot is worth being explicit about. Conversion needs
/// no model, so it runs on a simulator in CI. Deciding whether a given string is *accepted*
/// by the grammar needs a vocabulary, which needs a model file, which is not something CI
/// has — so acceptance and rejection are checked on the device through `GET /v1/selftest`,
/// against the schemas that actually matter. These tests cover the half that can be
/// automated: that real-world constructs convert at all, that the conversion is stable, and
/// that the constraints survive into the grammar text.
final class GrammarTests: XCTestCase {

    private func fixture(_ name: String) throws -> String {
        let bundle = Bundle(for: GrammarTests.self)
        guard let url = bundle.url(forResource: name, withExtension: "json")
                ?? Bundle.main.url(forResource: name, withExtension: "json") else {
            throw XCTSkip("fixture \(name).json is not in the test bundle")
        }
        return try String(contentsOf: url, encoding: .utf8)
    }

    private func compile(_ schema: String) throws -> String {
        var error: NSError?
        let grammar = LLMBridge.grammar(fromJSONSchema: schema, error: &error)
        if let error { XCTFail("conversion failed: \(error.localizedDescription)") }
        return try XCTUnwrap(grammar)
    }

    /// The whole set of constructs that arrive in practice, in one schema: `$defs`, `$ref`,
    /// `anyOf`, `enum`, `minItems`, `maxLength`, `minimum`, `default`, `required`,
    /// `additionalProperties`, nested eight deep.
    func testComplexSchemaConverts() throws {
        let grammar = try compile(try fixture("schema_complex"))
        XCTAssertFalse(grammar.isEmpty)
        XCTAssertTrue(grammar.contains("root"), "a grammar without a root rule cannot be used")
    }

    func testSimpleSchemaConverts() throws {
        let grammar = try compile(try fixture("schema_simple"))
        XCTAssertFalse(grammar.isEmpty)
    }

    /// Every enumerated value has to appear as a literal. If one is missing the model can
    /// never produce it; if the rule is absent altogether the model can produce anything,
    /// which is the failure this whole mechanism exists to prevent.
    func testEnumValuesSurviveIntoTheGrammar() throws {
        let grammar = try compile(try fixture("schema_complex"))
        for value in ["alpha", "beta", "gamma", "delta", "epsilon", "other"] {
            XCTAssertTrue(grammar.contains("\"\(value)\""),
                          "enum value \(value) is missing from the grammar")
        }
        for value in ["one", "two", "three", "four", "five"] {
            XCTAssertTrue(grammar.contains("\"\(value)\""),
                          "nested enum value \(value) is missing from the grammar")
        }
    }

    /// A value outside the enumeration must not be spellable. Checking the literal is absent
    /// is weaker than checking the grammar rejects it, but it catches the realistic failure:
    /// a converter that drops `enum` and emits a plain string rule.
    func testValuesOutsideTheEnumerationAreNotInTheGrammar() throws {
        let grammar = try compile(try fixture("schema_complex"))
        XCTAssertFalse(grammar.contains("\"omega\""))
        XCTAssertFalse(grammar.contains("\"six\""))
    }

    /// Required properties must be named in the grammar. The object rule for `Part` names
    /// only `name` as required; if that disappears, a model may emit `{}` where a product
    /// was expected, and downstream code sees an empty result rather than an error.
    func testRequiredPropertiesAppearInTheGrammar() throws {
        let grammar = try compile(try fixture("schema_complex"))
        for property in ["name", "details", "section", "tags", "parts", "amount"] {
            XCTAssertTrue(grammar.contains("\"\\\"\(property)\\\"\"") || grammar.contains(property),
                          "required property \(property) is missing from the grammar")
        }
    }

    /// Conversion is cached by schema text, so two spellings of the same schema must give
    /// the same grammar — otherwise the cache would be a source of divergence rather than a
    /// saving.
    func testConversionIsDeterministic() throws {
        let schema = try fixture("schema_complex")
        XCTAssertEqual(try compile(schema), try compile(schema))
    }

    /// The client library adds `additionalProperties: false` to every object before sending.
    /// A converter that rejected the key would fail on every real request while passing
    /// every test written against the framework's own output.
    func testAdditionalPropertiesFalseIsAccepted() throws {
        let schema = """
        {"type":"object","additionalProperties":false,
         "properties":{"a":{"type":"string"}},"required":["a"]}
        """
        XCTAssertFalse(try compile(schema).isEmpty)
    }

    /// Malformed input must fail loudly. The caller turns this into a 400; the one thing it
    /// must never do is fall back to asking the model nicely in the prompt.
    func testMalformedSchemaIsRejected() {
        var error: NSError?
        let grammar = LLMBridge.grammar(fromJSONSchema: "{not json", error: &error)
        XCTAssertNil(grammar)
        XCTAssertNotNil(error)
    }

    func testEmptySchemaIsRejected() {
        var error: NSError?
        XCTAssertNil(LLMBridge.grammar(fromJSONSchema: "", error: &error))
        XCTAssertNotNil(error)
    }

    /// The reject vectors carry a `_why` annotation describing what each one tests. With
    /// `additionalProperties: false` in the schema, that annotation is itself a violation —
    /// so a harness that forgets to strip it would reject every vector for the wrong reason
    /// and report a passing suite that checked nothing.
    func testRejectVectorsCarryAnnotationsThatMustBeStripped() throws {
        let bundle = Bundle(for: GrammarTests.self)
        guard let url = bundle.url(forResource: "schema_complex.reject", withExtension: "jsonl") else {
            throw XCTSkip("reject vectors are not in the test bundle")
        }
        let lines = try String(contentsOf: url, encoding: .utf8)
            .split(separator: "\n")
            .filter { !$0.isEmpty }
        XCTAssertFalse(lines.isEmpty)

        for line in lines {
            let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any])
            XCTAssertNotNil(object["_why"], "every reject vector should say what it tests")
            let stripped = object.filter { $0.key != "_why" }
            XCTAssertFalse(stripped.isEmpty, "a vector must carry something besides its annotation")
        }
    }
}
