import CryptoKit
import Foundation

#if canImport(Darwin)
    import Darwin
#else
    import Glibc
#endif

/// Execution inputs reference an immutable preregistration; existing campaigns must exit first.
public struct RuntimeResidencyDiagnosticPlan: Codable, Sendable {
    public var version: Int
    public var preregistration: EvaluationBatchPlan.Artifact
    public var waitFor: [EvaluationBatchPlan.Predecessor]
    public var exclusiveProcessIDs: [Int32]
}

/// Request/session residency investigation.
///
/// Its results never qualify a quantization recipe.
public enum RuntimeResidencyDiagnostic {
    struct Registration: Decodable {
        var format: String
        var model: RuntimeEvaluationPlan.Model
        var runtime: EvaluationBatchPlan.Artifact
        var metal: EvaluationBatchPlan.Artifact
        var workloads: [RuntimeEvaluationPlan.Workload]
        var minimumFreeBytes: UInt64
        var timeoutSeconds: Double
        var sourcePins: [EvaluationBatchPlan.Artifact]
        enum CodingKeys: String, CodingKey {
            case format, model, runtime, metal, workloads, minimumFreeBytes, timeoutSeconds
            case sourcePins = "source_pins"
        }
    }
    struct PriorProgress: Decodable {
        var status: String
        var pid: Int32
        var childPID: Int32?
        enum CodingKeys: String, CodingKey {
            case status, pid
            case childPID = "child_pid"
        }
    }
    struct Trial: Codable {
        var decodeTokensPerSecond: Double
        var prefillTokensPerSecond: Double
        var timeToFirstTokenMilliseconds: Double
        var promptFingerprint: String
        var contentSHA256: String
        var reasoningSHA256: String
    }
    struct Observation: Codable {
        var workload: String
        var position: Int
        var scope: String
        var nativePath: String
        var nativeSHA256: String
        var modelImplementation: String
        var warmups: [Trial]
        var trials: [Trial]
        var decodeDrift: Double
        var prefillDrift: Double
        var ttftDrift: Double
    }
    struct Progress: Codable {
        var format = "afterglow-native-residency-diagnostic-v1"
        var status = "waiting_for_existing_campaigns"
        var pid = ProcessInfo.processInfo.processIdentifier
        var planSHA256: String
        var preregistrationSHA256: String
        var childPID: Int32?
        var observations: [Observation] = []
        var performanceQualified = false
        var defaultsChanged = false
        var error: String?
    }

    public static func run(planURL: URL, output: URL, pollInterval: Duration = .seconds(2))
        async throws
    {
        let data = try Data(contentsOf: planURL)
        let plan = try JSONDecoder().decode(RuntimeResidencyDiagnosticPlan.self, from: data)
        guard plan.version == 1, plan.exclusiveProcessIDs.allSatisfy({ $0 > 0 }),
            plan.waitFor.allSatisfy({ $0.pid > 0 && !$0.terminal.isEmpty && $0.progress.hasPrefix("/") })
        else { throw EvaluationFailure.invalidSuite("Invalid diagnostic execution prerequisites.") }
        for artifact in [plan.preregistration] + plan.waitFor.flatMap({ [$0.plan, $0.controller] }) {
            guard artifact.path.hasPrefix("/"), artifact.bytes > 0, artifact.sha256.count == 64,
                artifact.sha256.allSatisfy(\.isHexDigit)
            else {
                throw EvaluationFailure.invalidSuite("Diagnostic execution requires complete absolute artifact pins.")
            }
        }
        try EvaluationBatch.verify(plan.preregistration)
        let registrationData = try Data(contentsOf: URL(fileURLWithPath: plan.preregistration.path))
        let registration = try JSONDecoder().decode(Registration.self, from: registrationData)
        try validate(registration, raw: registrationData)
        guard !FileManager.default.fileExists(atPath: output.path) else {
            throw EvaluationFailure.invalidSuite("Existing diagnostic evidence is protected.")
        }
        try FileManager.default.createDirectory(
            at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard mkdir(output.path, 0o700) == 0 else {
            throw EvaluationFailure.invalidSuite("Cannot exclusively create diagnostic output.")
        }
        var progress = Progress(
            planSHA256: digest(data), preregistrationSHA256: plan.preregistration.sha256)
        var child: Process?
        func save() throws {
            try EvaluationBatch.write(progress, to: output.appendingPathComponent("progress.json"))
        }
        func reserve() throws {
            let free =
                try FileManager.default.attributesOfFileSystem(forPath: output.path)[.systemFreeSize]
                as? NSNumber
            guard (free?.uint64Value ?? 0) >= registration.minimumFreeBytes + 1024 * 1024 * 1024 else {
                throw EvaluationFailure.transport("Disk reserve plus 1 GiB headroom reached.")
            }
        }
        func inputs() throws {
            guard try EvaluationBatch.hash(planURL) == progress.planSHA256 else {
                throw EvaluationFailure.transport("Diagnostic execution plan changed.")
            }
            for pin in [
                plan.preregistration, registration.runtime, registration.metal,
                registration.model.qualityIdentity, registration.model.qualityReport,
                registration.model.qualityManifest,
            ]
                + registration.sourcePins + plan.waitFor.flatMap({ [$0.plan, $0.controller] })
            {
                try EvaluationBatch.verify(pin)
            }
            try reserve()
        }
        func identity() throws -> EvaluationBatch.Checkpoint {
            let expected = try JSONDecoder().decode(
                EvaluationBatch.Checkpoint.self,
                from: Data(contentsOf: URL(fileURLWithPath: registration.model.qualityIdentity.path)))
            let actual = try EvaluationBatch.checkpoint(registration.model.checkpoint)
            guard actual == expected else {
                throw EvaluationFailure.transport(
                    "Diagnostic checkpoint differs from its quality-bound identity.")
            }
            return actual
        }
        do {
            try inputs()
            try data.write(to: output.appendingPathComponent("plan.json"), options: .atomic)
            try registrationData.write(
                to: output.appendingPathComponent("preregistration.json"), options: .atomic)
            try save()
            while true {
                try Task.checkCancellation()
                var ready = true
                for predecessor in plan.waitFor {
                    let prior = try JSONDecoder().decode(
                        PriorProgress.self,
                        from: Data(contentsOf: URL(fileURLWithPath: predecessor.progress)))
                    guard prior.pid == predecessor.pid else {
                        throw EvaluationFailure.transport("Predecessor process identity changed.")
                    }
                    let live = alive(prior.pid)
                    guard live || predecessor.terminal.contains(prior.status) else {
                        throw EvaluationFailure.transport("Predecessor lacks declared terminal evidence.")
                    }
                    if live || prior.childPID.map(alive) == true { ready = false }
                }
                if plan.exclusiveProcessIDs.contains(where: alive) { ready = false }
                try reserve()
                if ready { break }
                try await Task.sleep(for: pollInterval)
            }
            _ = try RuntimeEvaluation.quality(registration.model)
            for (workloadIndex, workload) in registration.workloads.enumerated() {
                var reference: [Trial]?
                var implementation: String?
                for (position, scope) in ["request", "session", "session", "request"].enumerated() {
                    try Task.checkCancellation()
                    try inputs()
                    guard !plan.exclusiveProcessIDs.contains(where: alive) else {
                        throw EvaluationFailure.transport(
                            "An exclusive process appeared during diagnostic execution.")
                    }
                    let before = try identity()
                    let stem = "w\(workloadIndex)-\(position)-\(scope)"
                    try EvaluationBatch.write(
                        before, to: output.appendingPathComponent(stem + "-before.json"))
                    let native = output.appendingPathComponent(stem + ".json")
                    var arguments = [
                        registration.model.checkpoint.path, native.path, "--engine", "metal",
                        "--tokens", "256", "--warmups", "8", "--trials", "8", "--context-length",
                        String(workload.contextLength),
                        "--prefill-step-size", "512", "--kv-compression", "none", "--disable-prompt-reuse",
                        "--temperature", "0", "--top-p", "1", "--prompt", workload.prompt,
                    ]
                    if scope == "session" { arguments.append("--wired-memory-session") }
                    try EvaluationBatch.write(
                        arguments, to: output.appendingPathComponent(stem + "-command.json"))
                    child = try EvaluationBatch.launch(
                        registration.runtime.path, arguments: arguments,
                        log: output.appendingPathComponent(stem + ".log"))
                    progress.status = "running_diagnostic"
                    progress.childPID = child!.processIdentifier
                    try save()
                    let deadline = Date().addingTimeInterval(registration.timeoutSeconds)
                    while child!.isRunning {
                        try Task.checkCancellation()
                        try reserve()
                        guard Date() < deadline else {
                            throw EvaluationFailure.transport("Diagnostic worker timeout.")
                        }
                        try await Task.sleep(for: pollInterval)
                    }
                    guard child!.terminationStatus == 0 else {
                        throw EvaluationFailure.transport("Diagnostic worker failed; evidence retained.")
                    }
                    child = nil
                    progress.childPID = nil
                    let raw = try Data(contentsOf: native)
                    let decoder = JSONDecoder()
                    decoder.keyDecodingStrategy = .convertFromSnakeCase
                    let report = try decoder.decode(RuntimeNativeReport.self, from: raw)
                    try report.validateResidency(
                        model: registration.model.checkpoint.path, workload: workload, scope: scope)
                    let all = (report.warmups + report.trials).map(trial)
                    if let reference, let implementation {
                        guard implementation == report.modelImplementation,
                            all.allSatisfy({ sameOutput($0, reference[0]) })
                        else {
                            throw EvaluationFailure.transport(
                                "Same-model diagnostic outputs or implementation changed.")
                        }
                    } else {
                        guard all.allSatisfy({ sameOutput($0, all[0]) }) else {
                            throw EvaluationFailure.transport("Diagnostic greedy output is inconsistent.")
                        }
                        reference = all
                        implementation = report.modelImplementation
                    }
                    try inputs()
                    let after = try identity()
                    guard before == after else {
                        throw EvaluationFailure.transport("Checkpoint changed during diagnostic.")
                    }
                    try EvaluationBatch.write(after, to: output.appendingPathComponent(stem + "-after.json"))
                    let trials = Array(all.suffix(8))
                    progress.observations.append(
                        .init(
                            workload: workload.name, position: position, scope: scope,
                            nativePath: native.path, nativeSHA256: digest(raw),
                            modelImplementation: report.modelImplementation,
                            warmups: Array(all.prefix(8)), trials: trials,
                            decodeDrift: drift(trials.map(\.decodeTokensPerSecond)),
                            prefillDrift: drift(trials.map(\.prefillTokensPerSecond)),
                            ttftDrift: drift(trials.map(\.timeToFirstTokenMilliseconds))))
                    try save()
                }
            }
            progress.status = "complete_diagnostic_only"
            try save()
        } catch {
            EvaluationBatch.stop(child)
            progress.childPID = nil
            progress.status = "failed_diagnostic_evidence_preserved"
            progress.error = error.localizedDescription
            try? save()
            throw error
        }
    }

    static func validate(_ registration: Registration, raw: Data) throws {
        guard registration.format == "afterglow-runtime-residency-diagnostic-preregistration-v1" else {
            throw EvaluationFailure.invalidSuite("Unknown residency preregistration format.")
        }
        let expected: [String: Any] = [
            "scope_order": ["request", "session", "session", "request"],
            "warmups_per_process": 8, "measured_trials_per_process": 8, "generated_tokens_per_trial": 256,
            "engine": "metal", "temperature": 0, "top_p": 1, "prefill_step_size": 512,
            "kv_compression": "none", "prompt_reuse": "disabled", "assistant": "none",
            "warmup_wiring": "request",
            "measured_session_flag": "--wired-memory-session", "adaptive_warmups": false,
            "selective_retry": false, "outlier_removal": false,
        ]
        let object = try JSONSerialization.jsonObject(with: raw) as? [String: Any]
        guard let actual = object?["fixed_protocol"] as? NSDictionary, actual.isEqual(to: expected)
        else {
            throw EvaluationFailure.invalidSuite(
                "Diagnostic fixed protocol differs from preregistered version one.")
        }
        try RuntimeEvaluationPlan(
            version: 1, runtime: registration.runtime, metal: registration.metal,
            baseline: registration.model, candidate: registration.model,
            workloads: registration.workloads,
            minimumFreeBytes: registration.minimumFreeBytes, exclusiveProcessIDs: [],
            timeoutSeconds: registration.timeoutSeconds
        ).validate()
    }
    static func trial(_ trial: RuntimeNativeReport.Trial) -> Trial {
        .init(
            decodeTokensPerSecond: trial.metrics.tokensPerSecond,
            prefillTokensPerSecond: trial.metrics.promptTokensPerSecond,
            timeToFirstTokenMilliseconds: trial.timeToFirstTokenMilliseconds,
            promptFingerprint: trial.promptTokenIdFingerprint,
            contentSHA256: digest(Data(trial.content.utf8)),
            reasoningSHA256: digest(Data(trial.reasoning.utf8)))
    }
    static func sameOutput(_ a: Trial, _ b: Trial) -> Bool {
        a.promptFingerprint == b.promptFingerprint && a.contentSHA256 == b.contentSHA256
            && a.reasoningSHA256 == b.reasoningSHA256
    }
    static func drift(_ values: [Double]) -> Double {
        func median(_ values: [Double]) -> Double {
            let sorted = values.sorted()
            return (sorted[1] + sorted[2]) / 2
        }
        return abs(median(Array(values.suffix(4))) / median(Array(values.prefix(4))) - 1)
    }
    static func alive(_ pid: Int32) -> Bool { kill(pid, 0) == 0 || errno == EPERM }
    static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
