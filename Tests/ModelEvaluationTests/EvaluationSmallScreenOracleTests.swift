import Foundation
import XCTest

@testable import ModelEvaluation

final class EvaluationSmallScreenOracleTests: XCTestCase, @unchecked Sendable {
    private func suite() throws -> EvaluationSuite {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let file = root.appendingPathComponent("EvaluationSuites/small-coding-cyber-tools-v2.json")
        let suite = try JSONDecoder().decode(EvaluationSuite.self, from: Data(contentsOf: file))
        try suite.validate()
        XCTAssertEqual(suite.cases.count, 4)
        return suite
    }

    private func sandbox() throws -> EvaluationDockerSandbox {
        guard let image = ProcessInfo.processInfo.environment["AFTERGLOW_TEST_DOCKER_IMAGE"] else {
            throw XCTSkip("Set AFTERGLOW_TEST_DOCKER_IMAGE to an installed digest-pinned Python image.")
        }
        return try EvaluationDockerSandbox(image: image)
    }

    func testRecursiveDeletionOracleAcceptsCorrectCodeAndRejectsShallowNewChildren() async throws {
        let input = try suite()
        let item = try XCTUnwrap(input.cases.first { $0.id == "code-config-patch" })
        let tests = try XCTUnwrap(item.turns.first?.pythonTests)
        let correct = """
            import copy
            def patch_config(target, patch):
                result = copy.deepcopy(target)
                for key, value in patch.items():
                    if value is None:
                        result.pop(key, None)
                    elif isinstance(value, dict):
                        child = result.get(key)
                        result[key] = patch_config(child if isinstance(child, dict) else {}, value)
                    else:
                        result[key] = copy.deepcopy(value)
                return result
            """
        // This fault mirrors the retained G32 output: copying a new dictionary leaves
        // nested None deletion markers in the result instead of recursively applying them.
        let faulty = """
            import copy
            def patch_config(target, patch):
                result = copy.deepcopy(target)
                for key, value in patch.items():
                    if value is None:
                        result.pop(key, None)
                    elif isinstance(value, dict):
                        if isinstance(result.get(key), dict):
                            result[key] = patch_config(result[key], value)
                        else:
                            result[key] = copy.deepcopy(value)
                    else:
                        result[key] = copy.deepcopy(value)
                return result
            """
        let sandbox = try sandbox()
        let pass = try await sandbox.check(code: correct, tests: tests)
        let fail = try await sandbox.check(code: faulty, tests: tests)
        XCTAssertTrue(pass)
        XCTAssertFalse(fail)
    }

    func testExplicit443PromptAndOracleAgree() async throws {
        let input = try suite()
        let item = try XCTUnwrap(input.cases.first { $0.id == "cyber-redirect-destination" })
        XCTAssertTrue(item.prompt.contains("An explicit :443 port is allowed"))
        let tests = try XCTUnwrap(item.turns.first?.pythonTests)
        let correct = """
            from urllib.parse import urlsplit
            def trusted_redirect(url):
                if any(c.isspace() or ord(c) < 32 or ord(c) == 127 or c == chr(92) for c in url):
                    return False
                try:
                    parsed = urlsplit(url)
                    return (parsed.scheme.lower() == 'https'
                            and parsed.netloc.lower() in ('api.internal', 'api.internal:443')
                            and parsed.hostname.lower() == 'api.internal'
                            and parsed.port in (None, 443))
                except (ValueError, AttributeError):
                    return False
            """
        let rejectsAllowedPort = correct.replacingOccurrences(
            of: "('api.internal', 'api.internal:443')", with: "('api.internal',)")
        let sandbox = try sandbox()
        let pass = try await sandbox.check(code: correct, tests: tests)
        let fail = try await sandbox.check(code: rejectsAllowedPort, tests: tests)
        XCTAssertTrue(pass)
        XCTAssertFalse(fail)
    }
}
