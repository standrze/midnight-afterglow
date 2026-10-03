import CryptoKit
import Foundation

#if canImport(Darwin)
    import Darwin
#else
    import Glibc
#endif

/// Frozen inputs for serial model evaluation; paths never trigger downloads.
public struct EvaluationBatchPlan: Codable, Sendable {
    public var version: Int
    public var models: [Model]
    public var waitFor: [Predecessor]
    public var minimumFreeBytes: UInt64
    public var pythonImage: String
    public var pins: [String: Artifact]

    public struct Artifact: Codable, Sendable {
        public var path: String
        public var bytes: UInt64
        public var sha256: String
    }
    public struct Model: Codable, Sendable {
        public var family: String
        public var recipe: String
        public var path: String
        public var sourceRevision: String
        enum CodingKeys: String, CodingKey {
            case family, recipe, path
            case sourceRevision = "source_revision"
        }
    }
    public struct Predecessor: Codable, Sendable {
        public var progress: String
        public var pid: Int32
        public var plan: Artifact
        public var controller: Artifact
        public var terminal: [String]
    }
    enum CodingKeys: String, CodingKey {
        case version, models, pins
        case waitFor = "wait_for"
        case minimumFreeBytes = "minimum_free_bytes"
        case pythonImage = "python_image"
    }

    public func validate() throws {
        let required: Set<String> = [
            "evaluator", "evaluator_metal", "suite", "server", "server_metal", "config", "probe",
        ]
        guard version == 1, !models.isEmpty, Set(models.map(\.recipe)).count == models.count,
            minimumFreeBytes >= 100 * 1024 * 1024 * 1024, required.isSubset(of: Set(pins.keys)),
            waitFor.allSatisfy({ $0.pid > 0 && !$0.terminal.isEmpty })
        else { throw EvaluationFailure.invalidSuite("Invalid batch version, recipes, pins, reserve or predecessor.") }
        for model in models {
            guard !model.family.isEmpty, !model.sourceRevision.isEmpty,
                !model.recipe.isEmpty,
                model.recipe.utf8.allSatisfy({
                    (48...57).contains($0) || (65...90).contains($0) || (97...122).contains($0) || $0 == 45
                })
            else {
                throw EvaluationFailure.invalidSuite(
                    "Recipe names must contain only ASCII letters, digits and hyphens.")
            }
        }
        for artifact in Array(pins.values) + waitFor.flatMap({ [$0.plan, $0.controller] }) {
            guard artifact.sha256.count == 64, artifact.sha256.allSatisfy(\.isHexDigit), artifact.bytes > 0 else {
                throw EvaluationFailure.invalidSuite("Invalid pinned artifact hash or size.")
            }
        }
        _ = try EvaluationDockerSandbox(image: pythonImage)
    }
}

/// Serial native Swift orchestration, with owned server cleanup and full checkpoint identities.
public enum EvaluationBatch {
    struct FileIdentity: Codable, Equatable {
        var path: String
        var bytes: UInt64
        var sha256: String
    }
    struct Checkpoint: Codable, Equatable {
        var recipe: String
        var files: [FileIdentity]
        var weightHashPolicy = "full_sha256_before_and_after"
    }
    private struct Completion: Codable {
        var recipe: String
        var family: String
        var report: String
        var sha256: String
        var exitCode: Int32
        var categoryScores: [String: EvaluationReport.Score]
        var fullCheckpointUnchanged: Bool
    }
    private struct Progress: Codable {
        var status = "waiting_for_existing_model_campaigns"
        var pid = ProcessInfo.processInfo.processIdentifier
        var planSHA256: String
        var completed: [Completion] = []
        var activeRecipe: String?
        var serverPID: Int32?
        var clientPID: Int32?
        var probePID: Int32?
        var freeBytes: UInt64 = 0
        var error: String?
        var performanceQualified = false
        var defaultsChanged = false
        enum CodingKeys: String, CodingKey {
            case status, pid, completed, error
            case planSHA256 = "plan_sha256"
            case activeRecipe = "active_recipe"
            case serverPID = "server_pid"
            case clientPID = "client_pid"
            case probePID = "probe_pid"
            case freeBytes = "free_bytes"
            case performanceQualified = "performance_qualified"
            case defaultsChanged = "defaults_changed"
        }
    }
    private struct PriorProgress: Decodable {
        var status: String
        var pid: Int32
        var childPID: Int32?
        enum CodingKeys: String, CodingKey {
            case status, pid
            case childPID = "child_pid"
        }
    }
    private struct Discovery: Decodable {
        var data: [Model]
        struct Model: Decodable { var id: String }
    }
    private struct Pair: Codable {
        var baseline: String
        var candidate: String
        var candidateOnlyPasses: [String]
        var baselineOnlyPasses: [String]
        var casesWithInfrastructureErrors: [String]
        var scope = "authored development suite; speed, memory budgets and promotion remain separate"
    }

    public static func run(planURL: URL, output: URL) async throws {
        let planData = try Data(contentsOf: planURL)
        let plan = try JSONDecoder().decode(EvaluationBatchPlan.self, from: planData)
        try plan.validate()
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let progressURL = output.appendingPathComponent("progress.json")
        guard !FileManager.default.fileExists(atPath: progressURL.path) else {
            throw EvaluationFailure.invalidSuite("Existing batch evidence is protected.")
        }
        var state = Progress(planSHA256: digest(planData))
        var server: Process?
        var client: Process?
        var observer: Process?
        func save() throws { try write(state, to: progressURL) }
        func pins() throws {
            guard try hash(planURL) == state.planSHA256 else {
                throw EvaluationFailure.invalidSuite("Batch plan changed.")
            }
            for artifact in plan.pins.values { try verify(artifact) }
        }
        func reserve() throws {
            let attributes = try FileManager.default.attributesOfFileSystem(forPath: output.path)
            state.freeBytes = (attributes[.systemFreeSize] as? NSNumber)?.uint64Value ?? 0
            guard state.freeBytes >= plan.minimumFreeBytes + 1024 * 1024 * 1024 else {
                throw EvaluationFailure.sandbox("100 GiB reserve plus early-stop headroom reached.")
            }
        }
        do {
            try pins()
            try reserve()
            try save()
            for predecessor in plan.waitFor {
                try verify(predecessor.plan)
                try verify(predecessor.controller)
            }
            while true {
                try Task.checkCancellation()
                var finished = true
                for predecessor in plan.waitFor {
                    let prior = try JSONDecoder().decode(
                        PriorProgress.self, from: Data(contentsOf: URL(fileURLWithPath: predecessor.progress)))
                    guard prior.pid == predecessor.pid else {
                        throw EvaluationFailure.transport("Predecessor process identity changed.")
                    }
                    let live = alive(prior.pid)
                    let terminal = predecessor.terminal.contains(prior.status) || prior.status.hasPrefix("failed_")
                    guard live || terminal else {
                        throw EvaluationFailure.transport("Predecessor ended without terminal evidence.")
                    }
                    if live || prior.childPID.map(alive) == true { finished = false }
                }
                try reserve()
                try save()
                if finished { break }
                try await Task.sleep(for: .seconds(5))
            }
            let suite = try JSONDecoder().decode(
                EvaluationSuite.self, from: Data(contentsOf: URL(fileURLWithPath: plan.pins["suite"]!.path)))
            try suite.validate()
            let caseIDs = suite.cases.map(\.id)
            for model in plan.models {
                try Task.checkCancellation()
                try pins()
                try reserve()
                let directory = output.appendingPathComponent(model.recipe)
                guard !FileManager.default.fileExists(atPath: directory.path) else {
                    throw EvaluationFailure.invalidSuite("Recipe output already exists.")
                }
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
                state.status = "verifying_checkpoint_before_run"
                state.activeRecipe = model.recipe
                try save()
                let before = try checkpoint(model)
                try write(before, to: directory.appendingPathComponent("checkpoint-before.json"))
                let port = try unusedPort()
                let secret = UUID().uuidString + UUID().uuidString
                let name = "afterglow-model-evaluation"
                let base = URL(string: "http://127.0.0.1:\(port)")!
                var environment = ProcessInfo.processInfo.environment
                environment["MIDNIGHT_API_KEY"] = secret
                environment["MIDNIGHT_MODEL_AVAILABILITY_FILE"] =
                    directory.appendingPathComponent("availability.json").path
                let serverArguments = [
                    "--config", plan.pins["config"]!.path, "--host", "127.0.0.1", "--port", String(port),
                    "--engine", "metal", "--no-ui", "--no-auto-assistant", "--model", model.path, "--name", name,
                    "--max-tokens", "2048", "--context-length", "8192", "--prefill-step-size", "512",
                    "--kv-compression", "none",
                ]
                try write(serverArguments, to: directory.appendingPathComponent("server-command.json"))
                server = try launch(
                    plan.pins["server"]!.path, arguments: serverArguments, environment: environment,
                    log: directory.appendingPathComponent("server.log"))
                state.serverPID = server!.processIdentifier
                observer = try launch(
                    plan.pins["probe"]!.path, arguments: [String(state.serverPID!), "100"],
                    log: directory.appendingPathComponent("footprint.jsonl"))
                state.probePID = observer!.processIdentifier
                state.status = "loading_owned_server"
                try save()
                let readyDeadline = Date().addingTimeInterval(300)
                while true {
                    try Task.checkCancellation()
                    try reserve()
                    guard server!.isRunning, Date() < readyDeadline else {
                        throw EvaluationFailure.transport("Owned server failed readiness.")
                    }
                    var request = URLRequest(url: base.appendingPathComponent("v1/models"), timeoutInterval: 3)
                    request.setValue("Bearer \(secret)", forHTTPHeaderField: "Authorization")
                    if let (data, response) = try? await URLSession.shared.data(for: request),
                        (response as? HTTPURLResponse)?.statusCode == 200,
                        let discovery = try? JSONDecoder().decode(Discovery.self, from: data),
                        discovery.data.contains(where: { $0.id == name })
                    {
                        try data.write(to: directory.appendingPathComponent("discovery.json"), options: .atomic)
                        break
                    }
                    try await Task.sleep(for: .milliseconds(250))
                }
                var evaluatorEnvironment = ProcessInfo.processInfo.environment
                evaluatorEnvironment["AFTERGLOW_EVAL_TOKEN"] = secret
                let evaluationDirectory = directory.appendingPathComponent("evaluation")
                client = try launch(
                    plan.pins["evaluator"]!.path,
                    arguments: [
                        "evaluate", "model", "--suite", plan.pins["suite"]!.path,
                        "--model", name, "--revision", model.sourceRevision, "--endpoint",
                        base.appendingPathComponent("v1/chat/completions").absoluteString,
                        "--maximum-tokens", "2048", "--timeout", "600", "--python-image", plan.pythonImage, "--output",
                        evaluationDirectory.path,
                    ],
                    environment: evaluatorEnvironment, log: directory.appendingPathComponent("evaluation.log"))
                state.clientPID = client!.processIdentifier
                state.status = "evaluating_model"
                try save()
                let deadline = Date().addingTimeInterval(9000)
                while client!.isRunning {
                    try Task.checkCancellation()
                    try reserve()
                    try save()
                    guard server!.isRunning, Date() < deadline else {
                        throw EvaluationFailure.transport("Owned server exited or evaluator timed out.")
                    }
                    try await Task.sleep(for: .seconds(2))
                }
                // The async poll observed termination; do not block a cooperative executor to reap it again.
                guard [0, 1].contains(client!.terminationStatus) else {
                    throw EvaluationFailure.transport("Unexpected evaluator exit.")
                }
                let reportURL = evaluationDirectory.appendingPathComponent("report.json")
                let report = try JSONDecoder().decode(EvaluationReport.self, from: Data(contentsOf: reportURL))
                guard report.results.map(\.id) == caseIDs else {
                    throw EvaluationFailure.transport("Incomplete case inventory.")
                }
                let manifest =
                    try JSONSerialization.jsonObject(
                        with: Data(contentsOf: evaluationDirectory.appendingPathComponent("manifest.json")))
                    as? [String: Any]
                guard manifest?["suiteSHA256"] as? String == plan.pins["suite"]!.sha256,
                    manifest?["maximumTokens"] as? Int == 2048
                else { throw EvaluationFailure.transport("Evaluator manifest mismatch.") }
                stop(server)
                stop(observer)
                let exitCode = client!.terminationStatus
                server = nil
                observer = nil
                client = nil
                state.serverPID = nil
                state.probePID = nil
                state.clientPID = nil
                try pins()
                let after = try checkpoint(model)
                try write(after, to: directory.appendingPathComponent("checkpoint-after.json"))
                guard before == after else {
                    throw EvaluationFailure.transport("Checkpoint changed during evaluation.")
                }
                state.completed.append(
                    .init(
                        recipe: model.recipe, family: model.family, report: reportURL.path,
                        sha256: try hash(reportURL), exitCode: exitCode, categoryScores: report.categoryScores,
                        fullCheckpointUnchanged: true))
                try save()
            }
            var reports: [String: EvaluationReport] = [:]
            for entry in state.completed {
                reports[entry.recipe] = try JSONDecoder().decode(
                    EvaluationReport.self, from: Data(contentsOf: URL(fileURLWithPath: entry.report)))
            }
            var pairs: [Pair] = []
            for entry in state.completed {
                let baselineName = entry.family + "-incumbent-ls2-q4"
                guard entry.recipe != baselineName, let baseline = reports[baselineName],
                    let candidate = reports[entry.recipe]
                else { continue }
                let base = Dictionary(uniqueKeysWithValues: baseline.results.map { ($0.id, $0.status) })
                let cand = Dictionary(uniqueKeysWithValues: candidate.results.map { ($0.id, $0.status) })
                pairs.append(
                    .init(
                        baseline: baselineName, candidate: entry.recipe,
                        candidateOnlyPasses: caseIDs.filter { base[$0] == "failed" && cand[$0] == "passed" },
                        baselineOnlyPasses: caseIDs.filter { base[$0] == "passed" && cand[$0] == "failed" },
                        casesWithInfrastructureErrors: caseIDs.filter { base[$0] == "error" || cand[$0] == "error" }))
            }
            try write(pairs, to: output.appendingPathComponent("paired-comparisons.json"))
            var lines = [
                "# Fixed model evaluation batch", "",
                "Authored development suite; speed, memory budgets and promotion remain separate.", "",
                "| Recipe | Coding | Cybersecurity | Tools | Errors |", "|---|---:|---:|---:|---:|",
            ]
            for entry in state.completed {
                let cells = ["coding", "cybersecurity", "tools"].map { category in
                    guard let score = entry.categoryScores[category] else { return "0/0" }
                    return "\(score.passed)/\(score.total)"
                }
                let errors = entry.categoryScores.values.reduce(0) { $0 + $1.errors }
                lines.append("| \(entry.recipe) | \(cells.joined(separator: " | ")) | \(errors) |")
            }
            try Data((lines.joined(separator: "\n") + "\n").utf8).write(
                to: output.appendingPathComponent("RESULTS.md"), options: .atomic)
            state.status = "all_model_suite_runs_complete"
            state.activeRecipe = nil
            try save()
        } catch {
            let containerOwner = client?.processIdentifier
            stop(client)
            if let containerOwner { cleanupContainers(owner: containerOwner) }
            stop(server)
            stop(observer)
            state.status = "failed_model_suite_evidence_preserved"
            state.error = error.localizedDescription
            state.clientPID = nil
            state.serverPID = nil
            state.probePID = nil
            try? save()
            throw error
        }
    }

    private static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    static func hash(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hash = SHA256()
        while let data = try handle.read(upToCount: 8 * 1024 * 1024), !data.isEmpty { hash.update(data: data) }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
    static func verify(_ artifact: EvaluationBatchPlan.Artifact) throws {
        let url = URL(fileURLWithPath: artifact.path)
        let size =
            try FileManager.default.attributesOfItem(atPath: url.resolvingSymlinksInPath().path)[.size] as? NSNumber
        guard size?.uint64Value == artifact.bytes, try hash(url) == artifact.sha256 else {
            throw EvaluationFailure.invalidSuite("Pinned artifact changed: \(artifact.path)")
        }
    }
    static func checkpoint(_ model: EvaluationBatchPlan.Model) throws -> Checkpoint {
        let paths = try FileManager.default.contentsOfDirectory(
            at: URL(fileURLWithPath: model.path), includingPropertiesForKeys: nil
        )
        .filter { ["safetensors", "json", "model", "jinja", "tiktoken"].contains($0.pathExtension) }.sorted {
            $0.path < $1.path
        }
        guard paths.contains(where: { $0.pathExtension == "safetensors" }) else {
            throw EvaluationFailure.invalidSuite("Retained weights are missing.")
        }
        var files: [FileIdentity] = []
        for path in paths {
            let resolved = path.resolvingSymlinksInPath()
            let size = try FileManager.default.attributesOfItem(atPath: resolved.path)[.size] as? NSNumber
            files.append(.init(path: resolved.path, bytes: size?.uint64Value ?? 0, sha256: try hash(path)))
        }
        return .init(recipe: model.recipe, files: files)
    }
    static func write<T: Encodable>(_ value: T, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(value).write(to: url, options: .atomic)
    }
    private static func alive(_ pid: Int32) -> Bool { kill(pid, 0) == 0 || errno == EPERM }
    static func launch(
        _ executable: String, arguments: [String], environment: [String: String]? = nil, log: URL
    ) throws -> Process {
        guard FileManager.default.createFile(atPath: log.path, contents: nil, attributes: [.posixPermissions: 0o600])
        else {
            throw EvaluationFailure.transport("Cannot create owned process log.")
        }
        let handle = try FileHandle(forWritingTo: log)
        defer { try? handle.close() }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        process.environment = environment
        process.standardOutput = handle
        process.standardError = handle
        process.standardInput = FileHandle.nullDevice
        try process.run()
        return process
    }
    static func stop(_ process: Process?) {
        guard let process, process.isRunning else { return }
        process.terminate()
        let deadline = Date().addingTimeInterval(20)
        while process.isRunning && Date() < deadline { usleep(50_000) }
        if process.isRunning { _ = kill(process.processIdentifier, SIGKILL) }
        process.waitUntilExit()
    }
    private static func cleanupContainers(owner: Int32) {
        let list = Process()
        list.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        list.arguments = ["docker", "ps", "--quiet", "--filter", "label=afterglow.evaluation.owner=\(owner)"]
        let pipe = Pipe()
        list.standardOutput = pipe
        list.standardError = FileHandle.nullDevice
        guard (try? list.run()) != nil else { return }
        let deadline = Date().addingTimeInterval(5)
        while list.isRunning && Date() < deadline { usleep(50_000) }
        if list.isRunning {
            stop(list)
            return
        }
        let ids =
            String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?.split(separator: "\n").map(
                String.init) ?? []
        guard !ids.isEmpty else { return }
        let remove = Process()
        remove.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        remove.arguments = ["docker", "rm", "--force"] + ids
        remove.standardOutput = FileHandle.nullDevice
        remove.standardError = FileHandle.nullDevice
        guard (try? remove.run()) != nil else { return }
        let removalDeadline = Date().addingTimeInterval(5)
        while remove.isRunning && Date() < removalDeadline { usleep(50_000) }
        if remove.isRunning { stop(remove) }
    }
    private static func unusedPort() throws -> UInt16 {
        let descriptor = socket(AF_INET, SOCK_STREAM, 0)
        guard descriptor >= 0 else { throw EvaluationFailure.transport("Cannot reserve a loopback port.") }
        defer { close(descriptor) }
        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
        address.sin_port = 0
        let length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let bound = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(descriptor, $0, length) }
        }
        var actualLength = length
        let read = withUnsafeMutablePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(descriptor, $0, &actualLength) }
        }
        guard bound == 0, read == 0 else {
            throw EvaluationFailure.transport("Cannot read the reserved loopback port.")
        }
        return UInt16(bigEndian: address.sin_port)
    }
}
