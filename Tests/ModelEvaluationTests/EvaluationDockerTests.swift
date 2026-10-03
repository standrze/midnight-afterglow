import Foundation
import XCTest

@testable import ModelEvaluation

final class EvaluationDockerTests: XCTestCase, @unchecked Sendable {
    func testBoundedSandboxPassFailureAndTimeout() async throws {
        guard let image = ProcessInfo.processInfo.environment["AFTERGLOW_TEST_DOCKER_IMAGE"] else {
            throw XCTSkip("Set AFTERGLOW_TEST_DOCKER_IMAGE to an already installed digest-pinned Python image.")
        }
        let sandbox = try EvaluationDockerSandbox(image: image, timeout: 2)
        let good = try await sandbox.check(
            code: "def add(a,b): return a+b", tests: "import solution\nassert solution.add(2,3)==5")
        XCTAssertTrue(good)
        let bad = try await sandbox.check(
            code: "def add(a,b): return a-b", tests: "import solution\nassert solution.add(2,3)==5")
        XCTAssertFalse(bad)
        let start = Date()
        let hung = try await sandbox.check(code: "while True: pass", tests: "import solution")
        XCTAssertFalse(hung)
        XCTAssertLessThan(Date().timeIntervalSince(start), 10)
    }
}
