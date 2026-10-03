import ArgumentParser
import CryptoKit
import Foundation
import ModelEvaluation

struct EvaluateModel: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "model", abstract: "Run a fixed coding, cybersecurity and tool-use suite against a served model.")
    @Option(help: "Versioned evaluation suite JSON.") var suite: String
    @Option(help: "Model ID exposed by the server.") var model: String
    @Option(help: "Full chat-completions URL.") var endpoint = "http://127.0.0.1:8080/v1/chat/completions"
    @Option(help: "New output directory; existing runs are never overwritten.") var output: String
    @Option(help: "Source revision recorded as an operator declaration.") var revision: String
    @Option(help: "Optional checkpoint SHA-256, recorded as an operator declaration.") var artifactSHA256: String?
    @Option var maximumTokens = 2048
    @Option var timeout: Double = 120
    @Option(help: "Digest-pinned Docker Python image, required for code cases. No image is downloaded.")
    var pythonImage: String?
    @Option var codeTimeout: Double = 15

    mutating func validate() throws {
        guard let url = URL(string: endpoint), ["http", "https"].contains(url.scheme), url.host != nil,
            maximumTokens > 0, maximumTokens <= 32768, timeout > 0, timeout <= 3600, !revision.isEmpty
        else { throw ValidationError("Valid endpoint, revision, token limit and timeout are required.") }
        if let hash = artifactSHA256, hash.count != 64 || !hash.allSatisfy({ $0.isHexDigit }) {
            throw ValidationError("--artifact-sha256 must contain 64 hexadecimal characters.")
        }
    }

    mutating func run() async throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: suite))
        let cases = try JSONDecoder().decode(EvaluationSuite.self, from: data)
        try cases.validate()
        let needsCode = cases.cases.contains { $0.turns.contains { $0.pythonTests != nil } }
        if needsCode && pythonImage == nil {
            throw ValidationError("This suite requires --python-image with a pinned digest.")
        }
        let sandbox = try pythonImage.map { try EvaluationDockerSandbox(image: $0, timeout: codeTimeout) }
        let directory = URL(fileURLWithPath: output)
        guard !FileManager.default.fileExists(atPath: directory.path) else {
            throw ValidationError("Output already exists.")
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        struct Identity: Encodable {
            let format = 1
            let model: String
            let revision: String
            let artifactSHA256: String?
            let suiteSHA256: String
            let identityEvidence = "operator_declared_not_verified_against_server_weights"
            let maximumTokens: Int
            let temperature = 0
            let timeout: Double
            let pythonImage: String?
            let codeTimeout: Double
            let createdAt: Date
            let timingScope = "HTTP wall time including queue, prefill and decode; not native decode throughput"
        }
        let identity = Identity(
            model: model, revision: revision, artifactSHA256: artifactSHA256,
            suiteSHA256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(),
            maximumTokens: maximumTokens, timeout: timeout, pythonImage: pythonImage,
            codeTimeout: codeTimeout, createdAt: Date())
        try encoder.encode(identity).write(to: directory.appendingPathComponent("manifest.json"), options: .atomic)
        try data.write(to: directory.appendingPathComponent("suite.json"), options: .atomic)
        let transport = EvaluationHTTPTransport(
            endpoint: URL(string: endpoint)!, model: model,
            maximumTokens: maximumTokens, timeout: timeout,
            token: ProcessInfo.processInfo.environment["AFTERGLOW_EVAL_TOKEN"])
        let report = try await EvaluationRunner.run(suite: cases, transport: transport, sandbox: sandbox) { result in
            let file = directory.appendingPathComponent(
                "case-\(SHA256.hash(data: Data(result.id.utf8)).map { String(format: "%02x", $0) }.joined()).json")
            let writer = JSONEncoder()
            writer.outputFormatting = [.prettyPrinted, .sortedKeys]
            try writer.encode(result).write(to: file, options: .atomic)
            print("\(result.status): \(result.id) — \(result.reason)")
        }
        try encoder.encode(report).write(to: directory.appendingPathComponent("report.json"), options: .atomic)
        for category in report.categoryScores.keys.sorted() {
            let score = report.categoryScores[category]!
            print("\(category): \(score.passed)/\(score.total) passed; \(score.failed) failed; \(score.errors) errors")
        }
        print("Review \(directory.appendingPathComponent("report.json").path)")
        if report.results.contains(where: { $0.status != "passed" }) { throw ExitCode.failure }
    }
}
