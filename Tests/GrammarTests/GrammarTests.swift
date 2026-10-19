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
        // Imported as throwing: a nullable return plus an NSError out-parameter is the
        // convention Swift folds into `throws`.
        try LLMBridge.grammar(fromJSONSchema: schema)
    }

    /// The whole set of constructs that arrive in practice, in one schema: `$defs`, `$ref`,
    /// `anyOf`, `enum`, `minItems`, `maxLength`, `minimum`, `default`, `required`,
    /// `additionalProperties`, nested eight deep.
    func testComplexSchemaConverts() throws {
        let grammar = try compile(try fixture("schema_complex"))
        XCTAssertFalse(grammar.isEmpty)
        XCTAssertTrue(grammar.contains("root"), "a grammar without a root rule cannot be used")
        // Printed, not asserted on: when an expectation about the grammar's shape fails, the
        // first question is always what the converter actually emitted, and nobody running
        // this in CI can open a debugger to find out.
        print("--- GBNF (first 800 characters) ---")
        print(String(grammar.prefix(800)))
        print("--- end ---")
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
        // Matched bare, without surrounding quotes. A GBNF literal for an enumerated string
        // carries escaped quotes of its own — `"\\"alpha\\""` — so looking for `"alpha"`
        // finds nothing even when the value is right there. That is what failed here first.
        for value in ["alpha", "beta", "gamma", "delta", "epsilon", "other"] {
            XCTAssertTrue(grammar.contains(value),
                          "enum value \(value) is missing from the grammar")
        }
        for value in ["one", "two", "three", "four", "five"] {
            XCTAssertTrue(grammar.contains(value),
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
            XCTAssertTrue(grammar.contains(property),
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
        XCTAssertThrowsError(try LLMBridge.grammar(fromJSONSchema: "{not json"))
    }

    func testEmptySchemaIsRejected() {
        XCTAssertThrowsError(try LLMBridge.grammar(fromJSONSchema: ""))
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

    // MARK: - Property order

    /// The line defining `rule`, e.g. "root ::= ...".
    private func ruleLine(_ rule: String, in grammar: String) throws -> String {
        try XCTUnwrap(grammar.split(separator: "\n").first { $0.hasPrefix("\(rule) ::=") }.map(String.init),
                      "no rule named \(rule) in:\n\(grammar)")
    }

    private func assertOrder(_ first: String, before second: String, in line: String,
                             file: StaticString = #filePath, lineNumber: UInt = #line) {
        guard let a = line.range(of: first), let b = line.range(of: second) else {
            return XCTFail("\(first) or \(second) missing from: \(line)", file: file, line: lineNumber)
        }
        XCTAssertLessThan(a.lowerBound, b.lowerBound,
                          "\(first) must come before \(second) in: \(line)", file: file, line: lineNumber)
    }

    /// The grammar follows the order in which the schema lists its properties — `b` first
    /// here, not the alphabetical `a`. The client designs that order on purpose: a model that
    /// writes the name before the categories classifies what it has read, and one forced the
    /// other way round guesses. Measured by the client at around 20 points out of 100.
    func testGrammarFollowsTheSchemasPropertyOrder() throws {
        let grammar = try compile("""
        {"type":"object","properties":{"b":{"type":"string"},"a":{"type":"string"}},"required":["b","a"]}
        """)
        assertOrder("b-kv", before: "a-kv", in: try ruleLine("root", in: grammar))
    }

    /// The same inside $defs, which is where the client's real schemas keep their objects.
    func testPropertyOrderHoldsInsideDefs() throws {
        let grammar = try compile("""
        {"$defs":{"Item":{"type":"object","properties":{"name":{"type":"string"},"categories":{"type":"string"}},
                          "required":["name","categories"]}},
         "type":"object","properties":{"item":{"$ref":"#/$defs/Item"}},"required":["item"]}
        """)
        let itemRule = try XCTUnwrap(grammar.split(separator: "\n")
            .first { $0.contains("Item ::=") }.map(String.init), "no Item rule in:\n\(grammar)")
        assertOrder("name-kv", before: "categories-kv", in: itemRule)
    }

    /// Two schemas that differ only in key order constrain the output differently, so they
    /// must produce different grammars — which also means the grammar cache must not key them
    /// on a sorted form, or one would be served the other's grammar.
    func testKeyOrderAloneChangesTheGrammar() throws {
        let ba = try compile("""
        {"type":"object","properties":{"b":{"type":"string"},"a":{"type":"string"}},"required":["b","a"]}
        """)
        let ab = try compile("""
        {"type":"object","properties":{"a":{"type":"string"},"b":{"type":"string"}},"required":["a","b"]}
        """)
        XCTAssertNotEqual(ba, ab)
    }
}
