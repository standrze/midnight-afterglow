import CryptoKit
import Foundation

#if canImport(Darwin)
    import Darwin
#else
    import Glibc
#endif

/// Serial runtime campaign using an explicitly pinned native executable and quality-bound checkpoints.
public enum RuntimeEvaluation {
    struct Arm: Codable {
        var phase: String
        var workload: String
        var pair: Int
        var position: Int
        var role: String
        var nativePath: String
        var nativeSHA256: String
        var modelImplementation: String
        var promptFingerprint: String
        var contentSHA256: String
        var reasoningSHA256: String
        var decodeTokensPerSecond: Double
        var prefillTokensPerSecond: Double
        var timeToFirstTokenMilliseconds: Double
    }
    struct Comparison: Codable {
        var phase: String
        var workload: String
        var decode: RuntimeMetricSummary
        var prefill: RuntimeMetricSummary
        var ttft: RuntimeMetricSummary
        var qualified: Bool { decode.qualified && prefill.qualified && ttft.qualified }
    }
    struct ScreeningComparison: Codable {
        var workload: String
        var baselineDecodeTokensPerSecond: [Double]
        var candidateDecodeTokensPerSecond: [Double]
        var pairedDecodeRatios: [Double]
        var pairedPrefillRatios: [Double]
        var pairedTTFTSpeedRatios: [Double]
        var limitation = "Two pairs are a screening observation; no confidence interval or performance qualification."
    }
    struct Progress: Codable {
        var format = "afterglow-native-runtime-v1"
        var status = "validating_inputs"
        var pid = ProcessInfo.processInfo.processIdentifier
        var planSHA256: String
        var childPID: Int32?
        var arms: [Arm] = []
        var comparisons: [Comparison] = []
        var screeningComparisons: [ScreeningComparison] = []
        var performanceQualified = false
        var normalSpeedBudgetPassed = false
        var defaultsChanged = false
        var error: String?
    }

    public static func run(planURL: URL, output: URL, pollInterval: Duration = .seconds(2)) async throws -> Bool {
        let data = try Data(contentsOf: planURL)
        let plan = try JSONDecoder().decode(RuntimeEvaluationPlan.self, from: data)
        try plan.validate()
        guard !FileManager.default.fileExists(atPath: output.path) else {
            throw EvaluationFailure.invalidSuite("Existing runtime evidence is protected.")
        }
        try FileManager.default.createDirectory(
            at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard mkdir(output.path, 0o700) == 0 else {
            throw EvaluationFailure.invalidSuite("Cannot exclusively create the new runtime output directory.")
        }
        var progress = Progress(planSHA256: digest(data))
        var child: Process?
        let progressURL = output.appendingPathComponent("progress.json")
        func save() throws { try EvaluationBatch.write(progress, to: progressURL) }
        func inputs() throws {
            guard try EvaluationBatch.hash(planURL) == progress.planSHA256 else {
                throw EvaluationFailure.invalidSuite("Runtime plan changed during the campaign.")
            }
            for pin in [
                plan.runtime, plan.metal, plan.baseline.qualityIdentity, plan.candidate.qualityIdentity,
                plan.baseline.qualityReport, plan.candidate.qualityReport, plan.baseline.qualityManifest,
                plan.candidate.qualityManifest,
            ] {
                try EvaluationBatch.verify(pin)
            }
            guard !plan.exclusiveProcessIDs.contains(where: { kill($0, 0) == 0 || errno == EPERM }) else {
                throw EvaluationFailure.transport("A declared exclusive model process is still live.")
            }
            let free =
                try FileManager.default.attributesOfFileSystem(forPath: output.path)[.systemFreeSize] as? NSNumber
            guard (free?.uint64Value ?? 0) >= plan.minimumFreeBytes + 1024 * 1024 * 1024 else {
                throw EvaluationFailure.transport("Disk reserve plus 1 GiB headroom reached.")
            }
        }
        func identity(_ model: RuntimeEvaluationPlan.Model) throws -> EvaluationBatch.Checkpoint {
            let expected = try JSONDecoder().decode(
                EvaluationBatch.Checkpoint.self,
                from: Data(contentsOf: URL(fileURLWithPath: model.qualityIdentity.path)))
            let actual = try EvaluationBatch.checkpoint(model.checkpoint)
            guard actual == expected else {
                throw EvaluationFailure.transport("Checkpoint differs from its pinned successful quality-run identity.")
            }
            return actual
        }
        do {
            try inputs()
            try data.write(to: output.appendingPathComponent("plan.json"), options: .atomic)
            try save()
            let baselineQuality = try quality(plan.baseline)
            let candidateQuality = try quality(plan.candidate)
            guard baselineQuality.suite == candidateQuality.suite, baselineQuality.ids == candidateQuality.ids else {
                throw EvaluationFailure.transport(
                    "Quality runs do not share an identical pinned suite and case inventory.")
            }
            let baselineIdentity = try identity(plan.baseline)
            let candidateIdentity = try identity(plan.candidate)
            try EvaluationBatch.write(baselineIdentity, to: output.appendingPathComponent("baseline-before.json"))
            try EvaluationBatch.write(candidateIdentity, to: output.appendingPathComponent("candidate-before.json"))
            let phases = plan.isScreening ? ["candidate"] : ["pre-control", "candidate", "post-control"]
            let pairCount = plan.isScreening ? 2 : 8
            let warmupCount = plan.isScreening ? 1 : 8
            for phase in phases {
                for (workloadIndex, workload) in plan.workloads.enumerated() {
                    progress.status = phase
                    for pair in 0..<pairCount {
                        let order = pair.isMultiple(of: 2) ? ["baseline", "candidate"] : ["candidate", "baseline"]
                        for (position, role) in order.enumerated() {
                            try Task.checkCancellation()
                            try inputs()
                            let model = phase == "candidate" && role == "candidate" ? plan.candidate : plan.baseline
                            let stem = "\(phase)-w\(workloadIndex)-p\(pair)-\(position)"
                            let native = output.appendingPathComponent(stem + ".json")
                            let arguments = [
                                model.checkpoint.path, native.path, "--engine", "metal", "--tokens", "256",
                                "--warmups", String(warmupCount), "--trials", "1", "--context-length",
                                String(workload.contextLength),
                                "--prefill-step-size", "512", "--kv-compression", "none", "--disable-prompt-reuse",
                                "--temperature", "0", "--top-p", "1", "--prompt", workload.prompt,
                            ]
                            try EvaluationBatch.write(
                                arguments, to: output.appendingPathComponent(stem + "-command.json"))
                            child = try EvaluationBatch.launch(
                                plan.runtime.path, arguments: arguments,
                                log: output.appendingPathComponent(stem + ".log"))
                            progress.childPID = child!.processIdentifier
                            try save()
                            let deadline = Date().addingTimeInterval(plan.timeoutSeconds)
                            while child!.isRunning {
                                try Task.checkCancellation()
                                // Hashing the runtime during generation could distort timings. Reserve checks are metadata only.
                                let free =
                                    try FileManager.default.attributesOfFileSystem(forPath: output.path)[
                                        .systemFreeSize]
                                    as? NSNumber
                                guard (free?.uint64Value ?? 0) >= plan.minimumFreeBytes + 1024 * 1024 * 1024,
                                    Date() < deadline
                                else {
                                    throw EvaluationFailure.transport("Runtime worker timeout or disk reserve reached.")
                                }
                                try await Task.sleep(for: pollInterval)
                            }
                            // isRunning is already false; a blocking run-loop wait can deadlock an async executor.
                            guard child!.terminationStatus == 0 else {
                                throw EvaluationFailure.transport("Native worker failed; partial evidence retained.")
                            }
                            child = nil
                            progress.childPID = nil
                            let raw = try Data(contentsOf: native)
                            let decoder = JSONDecoder()
                            decoder.keyDecodingStrategy = .convertFromSnakeCase
                            let report = try decoder.decode(RuntimeNativeReport.self, from: raw)
                            try report.validate(
                                model: model.checkpoint.path, workload: workload, expectedWarmups: warmupCount)
                            let trial = report.trials[0]
                            progress.arms.append(
                                .init(
                                    phase: phase, workload: workload.name, pair: pair, position: position, role: role,
                                    nativePath: native.path, nativeSHA256: digest(raw),
                                    modelImplementation: report.modelImplementation,
                                    promptFingerprint: trial.promptTokenIdFingerprint,
                                    contentSHA256: digest(Data(trial.content.utf8)),
                                    reasoningSHA256: digest(Data(trial.reasoning.utf8)),
                                    decodeTokensPerSecond: trial.metrics.tokensPerSecond,
                                    prefillTokensPerSecond: trial.metrics.promptTokensPerSecond,
                                    timeToFirstTokenMilliseconds: trial.timeToFirstTokenMilliseconds))
                            try save()
                        }
                    }
                    let arms = progress.arms.filter { $0.phase == phase && $0.workload == workload.name }
                    if plan.isScreening {
                        progress.screeningComparisons.append(try screen(arms))
                    } else {
                        progress.comparisons.append(try compare(arms, control: phase != "candidate"))
                    }
                    try save()
                }
                try inputs()
                let verifiedBaseline = try identity(plan.baseline)
                let verifiedCandidate = try identity(plan.candidate)
                guard verifiedBaseline == baselineIdentity, verifiedCandidate == candidateIdentity else {
                    throw EvaluationFailure.transport("Checkpoint changed during runtime phase.")
                }
                try EvaluationBatch.write(
                    verifiedBaseline, to: output.appendingPathComponent(phase + "-baseline-after.json"))
                try EvaluationBatch.write(
                    verifiedCandidate, to: output.appendingPathComponent(phase + "-candidate-after.json"))
                if !plan.isScreening && !progress.comparisons.filter({ $0.phase == phase }).allSatisfy(\.qualified) {
                    progress.status = "held_after_failed_\(phase)"
                    try save()
                    return false
                }
            }
            try EvaluationBatch.write(
                try identity(plan.baseline), to: output.appendingPathComponent("baseline-after.json"))
            try EvaluationBatch.write(
                try identity(plan.candidate), to: output.appendingPathComponent("candidate-after.json"))
            if plan.isScreening {
                progress.status = "complete_screening_only"
                try save()
                return true
            }
            progress.performanceQualified = true
            progress.normalSpeedBudgetPassed = progress.comparisons.filter { $0.phase == "candidate" }.allSatisfy {
                $0.decode.interval95[0] >= 0.75
            }
            progress.status = "complete_qualified_runtime_comparison"
            try save()
            return true
        } catch {
            EvaluationBatch.stop(child)
            progress.childPID = nil
            progress.status = "failed_runtime_evidence_preserved"
            progress.error = error.localizedDescription
            try? save()
            throw error
        }
    }

    static func screen(_ arms: [Arm]) throws -> ScreeningComparison {
        guard arms.count == 4, Set(arms.map(\.phase)) == ["candidate"],
            Set(arms.map(\.workload)).count == 1
        else { throw EvaluationFailure.transport("Incomplete screening pair inventory.") }
        var baseline: [Arm] = []
        var candidate: [Arm] = []
        for pair in 0..<2 {
            let a = arms.filter { $0.pair == pair && $0.role == "baseline" }
            let b = arms.filter { $0.pair == pair && $0.role == "candidate" }
            guard a.count == 1, b.count == 1, a[0].position == pair, b[0].position == 1 - pair,
                a[0].promptFingerprint == b[0].promptFingerprint,
                a[0].modelImplementation == b[0].modelImplementation,
                (a + b).allSatisfy({ arm in
                    [arm.decodeTokensPerSecond, arm.prefillTokensPerSecond, arm.timeToFirstTokenMilliseconds]
                        .allSatisfy { $0.isFinite && $0 > 0 }
                })
            else { throw EvaluationFailure.transport("Unmatched, invalid or incorrectly ordered screening pair.") }
            baseline.append(a[0])
            candidate.append(b[0])
        }
        return .init(
            workload: arms[0].workload,
            baselineDecodeTokensPerSecond: baseline.map(\.decodeTokensPerSecond),
            candidateDecodeTokensPerSecond: candidate.map(\.decodeTokensPerSecond),
            pairedDecodeRatios: zip(baseline, candidate).map { $1.decodeTokensPerSecond / $0.decodeTokensPerSecond },
            pairedPrefillRatios: zip(baseline, candidate).map { $1.prefillTokensPerSecond / $0.prefillTokensPerSecond },
            pairedTTFTSpeedRatios: zip(baseline, candidate).map {
                $0.timeToFirstTokenMilliseconds / $1.timeToFirstTokenMilliseconds
            })
    }

    static func compare(_ arms: [Arm], control: Bool) throws -> Comparison {
        guard arms.count == 16, Set(arms.map(\.phase)).count == 1, Set(arms.map(\.workload)).count == 1 else {
            throw EvaluationFailure.transport("Incomplete runtime pair inventory.")
        }
        var baseline: [Arm] = []
        var candidate: [Arm] = []
        for pair in 0..<8 {
            let a = arms.filter { $0.pair == pair && $0.role == "baseline" }
            let b = arms.filter { $0.pair == pair && $0.role == "candidate" }
            guard a.count == 1, b.count == 1, a[0].position == (pair.isMultiple(of: 2) ? 0 : 1),
                b[0].position == (pair.isMultiple(of: 2) ? 1 : 0),
                a[0].promptFingerprint == b[0].promptFingerprint, a[0].modelImplementation == b[0].modelImplementation,
                !control || (a[0].contentSHA256 == b[0].contentSHA256 && a[0].reasoningSHA256 == b[0].reasoningSHA256)
            else { throw EvaluationFailure.transport("Duplicate, unmatched or incorrectly ordered runtime pair.") }
            baseline.append(a[0])
            candidate.append(b[0])
        }
        func metric(_ key: KeyPath<Arm, Double>, latency: Bool = false) throws -> RuntimeMetricSummary {
            try RuntimeStatistics.summarize(
                baseline: baseline.map { $0[keyPath: key] }, candidate: candidate.map { $0[keyPath: key] },
                latency: latency, control: control)
        }
        return try .init(
            phase: arms[0].phase, workload: arms[0].workload, decode: metric(\.decodeTokensPerSecond),
            prefill: metric(\.prefillTokensPerSecond), ttft: metric(\.timeToFirstTokenMilliseconds, latency: true))
    }

    static func quality(_ model: RuntimeEvaluationPlan.Model) throws -> (suite: String, ids: [String]) {
        struct Manifest: Decodable {
            var revision: String
            var suiteSHA256: String
            var maximumTokens: Int
        }
        let manifest = try JSONDecoder().decode(
            Manifest.self, from: Data(contentsOf: URL(fileURLWithPath: model.qualityManifest.path)))
        let report = try JSONDecoder().decode(
            EvaluationReport.self, from: Data(contentsOf: URL(fileURLWithPath: model.qualityReport.path)))
        let ids = report.results.map(\.id)
        guard manifest.revision == model.checkpoint.sourceRevision, manifest.maximumTokens == 2048,
            manifest.suiteSHA256.count == 64, manifest.suiteSHA256.allSatisfy(\.isHexDigit),
            !ids.isEmpty, Set(ids).count == ids.count,
            report.results.allSatisfy({ ["passed", "failed"].contains($0.status) }),
            report.categoryScores.values.allSatisfy({ $0.errors == 0 && $0.passed >= 0 && $0.failed >= 0 }),
            report.categoryScores.values.reduce(0, { $0 + $1.total }) == ids.count
        else {
            throw EvaluationFailure.transport(
                "Quality evidence is incomplete, inconsistent or contains infrastructure errors.")
        }
        return (manifest.suiteSHA256, ids)
    }

    private static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
