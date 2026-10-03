import Foundation
import XCTest

@testable import ModelEvaluation

final class EvaluationOwnerContractTests: XCTestCase {
    private struct Fixtures: Decodable {
        var reference: String
        var mutants: [String: String]
    }

    func testClarifiedOwnerContractReferenceAndDefects() async throws {
        guard let image = ProcessInfo.processInfo.environment["AFTERGLOW_TEST_DOCKER_IMAGE"] else {
            throw XCTSkip("Set AFTERGLOW_TEST_DOCKER_IMAGE for the isolated owner contract.")
        }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let decoder = JSONDecoder()
        let suite = try decoder.decode(
            EvaluationSuite.self,
            from: Data(
                contentsOf: root.appendingPathComponent("EvaluationSuites/coding-cyber-tools-v2-owner-correction.json"))
        )
        try suite.validate()
        XCTAssertEqual(suite.cases.count, 1)
        XCTAssertEqual(suite.cases[0].id, "cyber-owner-check-v2")
        let fixtures = try decoder.decode(
            Fixtures.self,
            from: Data(contentsOf: root.appendingPathComponent("Tests/Fixtures/ModelEvaluation/owner-contract-v2.json"))
        )
        let tests = try XCTUnwrap(suite.cases[0].turns.last?.pythonTests)
        let sandbox = try EvaluationDockerSandbox(image: image)
        let referencePassed = try await sandbox.check(code: fixtures.reference, tests: tests)
        XCTAssertTrue(referencePassed)
        for (name, code) in fixtures.mutants.sorted(by: { $0.key < $1.key }) {
            let defectPassed = try await sandbox.check(code: code, tests: tests)
            XCTAssertFalse(defectPassed, "Defect escaped the clarified contract: \(name)")
        }
    }
}
