import Foundation
import XCTest

@testable import ModelEvaluation

private actor Replay: EvaluationTransport {
    var responses: [EvaluationResponse]
    var requests: [[EvaluationMessage]] = []
    init(_ responses: [EvaluationResponse]) { self.responses = responses }
    func complete(messages: [EvaluationMessage], tools: [EvaluationTool]) async throws -> EvaluationResponse {
        requests.append(messages)
        guard !responses.isEmpty else { throw EvaluationFailure.transport("Replay exhausted") }
        return responses.removeFirst()
    }
    func recorded() -> [[EvaluationMessage]] { requests }
}

private struct CodeFixture: EvaluationCodeSandbox {
    let passes: Bool
    func check(code: String, tests: String) async throws -> Bool { passes }
}

final class EvaluationRunnerTests: XCTestCase, @unchecked Sendable {
    private func suite() throws -> EvaluationSuite {
        let text =
            #"{"version":1,"name":"test","split":"development","cases":[{"id":"tool","category":"tools","sourceFamily":"lookup","prompt":"Lookup item 7. Return JSON.","tools":[{"name":"lookup","description":"Lookup","parameters":{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"],"additionalProperties":false}}],"turns":[{"calls":[{"name":"lookup","arguments":{"id":7},"result":"PRIVATE_RESULT"}]},{"calls":[],"answer":{"ok":true}}]}]}"#
        return try JSONDecoder().decode(EvaluationSuite.self, from: Data(text.utf8))
    }
    private func call(_ arguments: String = #"{"id":7}"#) -> EvaluationResponse {
        .init(
            message: .init(role: "assistant", calls: [.init(id: "call1", name: "lookup", arguments: arguments)]),
            finishReason: "tool_calls")
    }
    private func answer(_ reason: String = "stop") -> EvaluationResponse {
        .init(message: .init(role: "assistant", content: #"{"ok":true}"#), finishReason: reason)
    }

    func testToolConversationAndPrivateExpectations() async throws {
        let replay = Replay([call(), answer()])
        let report = try await EvaluationRunner.run(suite: suite(), transport: replay)
        XCTAssertEqual(report.results[0].status, "passed")
        let requests = await replay.recorded()
        XCTAssertEqual(requests.count, 2)
        XCTAssertFalse(String(data: try JSONEncoder().encode(requests[0]), encoding: .utf8)!.contains("PRIVATE_RESULT"))
        XCTAssertEqual(requests[1].last?.role, "tool")
        XCTAssertEqual(requests[1].last?.toolCallID, "call1")
        XCTAssertEqual(requests[1].last?.content, "PRIVATE_RESULT")
    }

    func testWrongArgumentTypeAndExtraArgumentsFail() async throws {
        for arguments in [#"{"id":"7"}"#, #"{"id":7,"admin":true}"#, "not JSON"] {
            let report = try await EvaluationRunner.run(suite: suite(), transport: Replay([call(arguments)]))
            XCTAssertEqual(report.results[0].status, "failed")
        }
    }

    func testTruncationAndUnexpectedCallsFail() async throws {
        let report = try await EvaluationRunner.run(suite: suite(), transport: Replay([call(), answer("length")]))
        XCTAssertEqual(report.results[0].status, "failed")
        let extra = try await EvaluationRunner.run(suite: suite(), transport: Replay([call(), call()]))
        XCTAssertEqual(extra.results[0].status, "failed")
    }

    func testTransportErrorsStayInDenominatorAndRemainingCasesRun() async throws {
        var input = try suite()
        var second = input.cases[0]
        second.id = "second"
        input.cases.append(second)
        let report = try await EvaluationRunner.run(suite: input, transport: Replay([]))
        XCTAssertEqual(report.results.count, 2)
        XCTAssertEqual(report.categoryScores["tools"]?.errors, 2)
        XCTAssertEqual(report.categoryScores["tools"]?.total, 2)
    }

    func testCodeRequiresSandboxAndBehavioralPass() async throws {
        var input = try suite()
        input.cases[0].tools = []
        input.cases[0].turns = [.init(calls: [], answer: nil, pythonTests: "assert solution.add(2,3)==5")]
        let response = EvaluationResponse(
            message: .init(role: "assistant", content: "```python\ndef add(a,b): return a+b\n```"), finishReason: "stop"
        )
        let missing = try await EvaluationRunner.run(suite: input, transport: Replay([response]))
        XCTAssertEqual(missing.results[0].status, "error")
        let failed = try await EvaluationRunner.run(
            suite: input, transport: Replay([response]), sandbox: CodeFixture(passes: false))
        XCTAssertEqual(failed.results[0].status, "failed")
        let passed = try await EvaluationRunner.run(
            suite: input, transport: Replay([response]), sandbox: CodeFixture(passes: true))
        XCTAssertEqual(passed.results[0].status, "passed")
        XCTAssertEqual(EvaluationRunner.pythonCode(response.message.content!), "def add(a,b): return a+b")
    }

    func testInvalidSuiteCannotRun() throws {
        var input = try suite()
        input.cases.append(input.cases[0])
        XCTAssertThrowsError(try input.validate())
        input = try suite()
        input.cases[0].turns.removeLast()
        XCTAssertThrowsError(try input.validate())
        XCTAssertThrowsError(try EvaluationDockerSandbox(image: "python:latest"))
    }
}
