import Foundation
import XCTest

@testable import DecisionModels

final class DecisionContractTests: XCTestCase {
    private var contract: DecisionModelContract {
        DecisionModelContract(
            model: "Qwen/Qwen3.5-9B", revision: String(repeating: "a", count: 40),
            candidateCodes: ["A", "B", "C"], candidateTokenIDs: [32, 33, 34])
    }

    func testSchemaOrderAndDuplicateKeys() throws {
        let data = Data(
            #"{"model":"nimble","context":"Text","schema":{"z":{"type":"boolean","description":"Z?"},"a":{"type":"boolean","description":"A?"}}}"#
                .utf8)
        let request = try DecisionRequest.decode(data)
        XCTAssertEqual(request.names, ["z", "a"])
        XCTAssertThrowsError(try DecisionRequest.decode(Data(#"{"context":"a","context":"b","schema":{}}"#.utf8)))
        XCTAssertThrowsError(
            try DecisionRequest.decode(
                Data(#"{"context":"a","schema":{"x":{"type":"boolean","description":"a","description":"b"}}}"#.utf8)))
    }

    func testPromptMatchesPythonSpacingAndEscaping() throws {
        let request = DecisionRequest(
            model: "nimble", context: "<alert> café\n",
            schema: ["allowed": .init(type: "boolean", description: "Allowed?")])
        let text = try DecisionPrompt.text(request: request, field: "allowed", contract: contract)
        XCTAssertTrue(
            text.contains(
                #"{"context": "\u003calert\u003e café\n", "schema": [{"name": "allowed", "description": "Allowed?", "choices": [{"code": "A", "value": false}, {"code": "B", "value": true}]}]}"#
            ))
        XCTAssertTrue(
            text.hasSuffix("Requested field: \"allowed\"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"))
    }

    func testBooleanOrderingRubricAndTemperature() throws {
        let first = try DecisionScoring.result(
            logits: [1000, 1002], choices: [.boolean(true), .boolean(false)], temperature: 1)
        let second = try DecisionScoring.result(
            logits: [1000, 1002], choices: [.boolean(true), .boolean(false)], temperature: 2)
        XCTAssertEqual(first.value, .boolean(false))
        XCTAssertEqual(second.value, first.value)
        XCTAssertEqual(first.probabilities.values.reduce(0, +), 1, accuracy: 1e-12)
        XCTAssertLessThan(second.probabilities["false"]!, first.probabilities["false"]!)
        let rubric = try DecisionScoring.result(
            logits: [0, 0], choices: [.string("0"), .string("2")], temperature: 1, rubric: true)
        XCTAssertEqual(rubric.expectedScore!, 1, accuracy: 1e-12)
        XCTAssertThrowsError(try DecisionScoring.result(logits: [.nan], choices: [.string("a")], temperature: 1))
    }

    func testInvalidChoicesOrderAndContract() throws {
        let request = DecisionRequest(
            model: "nimble", context: "x",
            schema: ["a": .init(type: "enum", description: "A?", choices: [.string("x"), .string("x")])])
        XCTAssertThrowsError(try request.validate(maximumChoices: 255))
        var bad = contract
        bad.temperature = 0
        XCTAssertThrowsError(try bad.validate())
        bad = contract
        bad.candidateTokenIDs = [32, 32, 34]
        XCTAssertThrowsError(try bad.validate())
    }

    func testSourceFamilyLeakage() throws {
        let request = DecisionRequest(
            model: "nimble", context: "x", schema: ["a": .init(type: "boolean", description: "A?")])
        let first = DecisionExample(id: "1", sourceFamily: "shared", request: request, labels: ["a": .boolean(true)])
        let second = DecisionExample(id: "2", sourceFamily: "shared", request: request, labels: ["a": .boolean(false)])
        XCTAssertThrowsError(try DecisionDataset.validateSplits([[first], [second]]))
    }
}
