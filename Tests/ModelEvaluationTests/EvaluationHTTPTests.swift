import Foundation
import Synchronization
import XCTest

@testable import ModelEvaluation

private struct HTTPFixtureState: Sendable {
    var cases: [EvaluationCase] = []
    var references: [String: String] = [:]
    var faultyArgument = false
    var requests = 0
}

private final class EvaluationHTTPFixture: URLProtocol, @unchecked Sendable {
    static let state = Mutex(HTTPFixtureState())
    override class func canInit(with request: URLRequest) -> Bool { request.url?.host == "afterglow-fixture.invalid" }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        do {
            let body: Data
            if let data = request.httpBody {
                body = data
            } else if let stream = request.httpBodyStream {
                stream.open()
                defer { stream.close() }
                var data = Data()
                var buffer = [UInt8](repeating: 0, count: 4096)
                while stream.hasBytesAvailable {
                    let count = stream.read(&buffer, maxLength: buffer.count)
                    guard count >= 0 else { throw EvaluationFailure.transport("Fixture request stream failed.") }
                    if count == 0 { break }
                    data.append(contentsOf: buffer.prefix(count))
                }
                body = data
            } else {
                throw EvaluationFailure.transport("Missing fixture body.")
            }
            let object = try JSONSerialization.jsonObject(with: body) as! [String: Any]
            guard object["turns"] == nil, object["pythonTests"] == nil,
                let messages = object["messages"] as? [[String: Any]],
                let prompt = messages.first?["content"] as? String
            else { throw EvaluationFailure.transport("Private expectations leaked into request.") }
            let encoded = try Self.state.withLock { fixture in
                fixture.requests += 1
                guard let item = fixture.cases.first(where: { $0.prompt == prompt }) else {
                    throw EvaluationFailure.transport("Unknown fixture prompt.")
                }
                let turn = messages.filter { $0["role"] as? String == "assistant" }.count
                let expected = item.turns[turn]
                var message: [String: Any] = ["role": "assistant"]
                var finish = "stop"
                if !expected.calls.isEmpty {
                    message["tool_calls"] = try expected.calls.enumerated().map { index, call in
                        let arguments =
                            fixture.faultyArgument && item.id == "tool-typed-lookup"
                            ? EvaluationJSON.object(["ticket_id": .string("73")]) : call.arguments
                        let text = String(data: try JSONEncoder().encode(arguments), encoding: .utf8)!
                        return [
                            "id": "call-\(turn)-\(index)", "type": "function",
                            "function": ["name": call.name, "arguments": text],
                        ] as [String: Any]
                    }
                    finish = "tool_calls"
                } else if let answer = expected.answer {
                    message["content"] = String(data: try JSONEncoder().encode(answer), encoding: .utf8)!
                } else {
                    message["content"] = fixture.references[item.id]!
                }
                return try JSONSerialization.data(withJSONObject: [
                    "model": object["model"]!, "choices": [["message": message, "finish_reason": finish]],
                    "usage": ["completion_tokens": 32],
                ])
            }
            let response = HTTPURLResponse(
                url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1",
                headerFields: ["Content-Type": "application/json"])!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: encoded)
            client?.urlProtocolDidFinishLoading(self)
        } catch { client?.urlProtocol(self, didFailWithError: error) }
    }
}

final class EvaluationHTTPTests: XCTestCase, @unchecked Sendable {
    func testCompleteFourteenCaseSuiteThroughNativeHTTPAdapter() async throws {
        guard let image = ProcessInfo.processInfo.environment["AFTERGLOW_TEST_DOCKER_IMAGE"] else {
            throw XCTSkip("Set AFTERGLOW_TEST_DOCKER_IMAGE for the isolated Python contracts.")
        }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let suite = try JSONDecoder().decode(
            EvaluationSuite.self,
            from: Data(contentsOf: root.appendingPathComponent("EvaluationSuites/coding-cyber-tools-v1.json")))
        let references = try JSONDecoder().decode(
            [String: String].self,
            from: Data(
                contentsOf: root.appendingPathComponent("Tests/Fixtures/ModelEvaluation/reference-solutions.json")))
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [EvaluationHTTPFixture.self]
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }
        let transport = EvaluationHTTPTransport(
            endpoint: URL(string: "http://afterglow-fixture.invalid/v1/chat/completions")!, model: "mock-fixture",
            session: session)
        let sandbox = try EvaluationDockerSandbox(image: image)
        for fault in [false, true] {
            EvaluationHTTPFixture.state.withLock {
                $0 = .init(cases: suite.cases, references: references, faultyArgument: fault)
            }
            let report = try await EvaluationRunner.run(suite: suite, transport: transport, sandbox: sandbox)
            XCTAssertEqual(report.results.count, 14)
            XCTAssertEqual(
                report.results.filter { $0.status != "passed" }.map(\.id), fault ? ["tool-typed-lookup"] : [])
            XCTAssertGreaterThan(EvaluationHTTPFixture.state.withLock { $0.requests }, 14)
        }
    }
}
