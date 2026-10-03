import Foundation

/// Runs fixed cases serially; infrastructure errors never count as passes or disappear from denominators.
public enum EvaluationRunner {
    public static func run(
        suite: EvaluationSuite, transport: any EvaluationTransport,
        sandbox: (any EvaluationCodeSandbox)? = nil,
        onResult: (@Sendable (EvaluationCaseResult) throws -> Void)? = nil
    ) async throws -> EvaluationReport {
        try suite.validate()
        var results: [EvaluationCaseResult] = []
        for item in suite.cases {
            try Task.checkCancellation()
            let result = await runCase(item, transport: transport, sandbox: sandbox)
            results.append(result)
            try onResult?(result)
        }
        var scores: [String: EvaluationReport.Score] = [:]
        for result in results {
            var score = scores[result.category] ?? .init(passed: 0, failed: 0, errors: 0)
            switch result.status {
            case "passed": score.passed += 1
            case "failed": score.failed += 1
            default: score.errors += 1
            }
            scores[result.category] = score
        }
        return EvaluationReport(suiteName: suite.name, split: suite.split, results: results, categoryScores: scores)
    }

    private static func runCase(
        _ item: EvaluationCase, transport: any EvaluationTransport, sandbox: (any EvaluationCodeSandbox)?
    ) async -> EvaluationCaseResult {
        var responses: [EvaluationResponse] = []
        func result(_ status: String, _ reason: String) -> EvaluationCaseResult {
            .init(
                id: item.id, category: item.category, sourceFamily: item.sourceFamily,
                status: status, reason: reason, responses: responses)
        }
        var messages = [EvaluationMessage(role: "user", content: item.prompt)]
        do {
            for (index, expected) in item.turns.enumerated() {
                let response = try await transport.complete(messages: messages, tools: item.tools)
                responses.append(response)
                guard response.message.role == "assistant" else { return result("failed", "Non-assistant response.") }
                let calls = response.message.toolCalls ?? []
                if !expected.calls.isEmpty {
                    guard response.finishReason == "tool_calls", calls.count == expected.calls.count,
                        Set(calls.map(\.id)).count == calls.count,
                        calls.allSatisfy({ !$0.id.isEmpty && $0.type == "function" })
                    else { return result("failed", "Turn \(index): missing, extra, invalid or truncated tool calls.") }
                    for (actual, gold) in zip(calls, expected.calls) {
                        guard actual.function.name == gold.name,
                            (try? EvaluationJSON.parse(actual.function.arguments)) == gold.arguments
                        else { return result("failed", "Turn \(index): wrong tool, argument value or argument type.") }
                    }
                    messages.append(response.message)
                    for (actual, gold) in zip(calls, expected.calls) {
                        messages.append(.init(role: "tool", content: gold.result, callID: actual.id))
                    }
                } else {
                    guard calls.isEmpty, response.finishReason == "stop", let content = response.message.content else {
                        return result("failed", "Unexpected tool call, missing answer or truncated output.")
                    }
                    if let answer = expected.answer {
                        guard (try? EvaluationJSON.parse(content)) == answer else {
                            return result("failed", "Answer does not match the exact structured contract.")
                        }
                    } else if let tests = expected.pythonTests {
                        guard let sandbox else {
                            return result("error", "Python case requires an enabled Docker sandbox.")
                        }
                        let code = pythonCode(content)
                        guard try await sandbox.check(code: code, tests: tests) else {
                            return result("failed", "Generated Python failed private behavioral tests.")
                        }
                    }
                }
            }
            return result("passed", "All turns and behavioral expectations passed.")
        } catch {
            return result("error", error.localizedDescription)
        }
    }

    public static func pythonCode(_ content: String) -> String {
        let text = content.trimmingCharacters(in: .whitespacesAndNewlines)
        let lines = text.components(separatedBy: "\n")
        if lines.count >= 3, ["```python", "```"].contains(lines[0]), lines.last == "```" {
            return lines.dropFirst().dropLast().joined(separator: "\n")
        }
        return text
    }
}
