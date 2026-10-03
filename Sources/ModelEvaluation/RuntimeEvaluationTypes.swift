import Foundation

/// Immutable inputs for a serial native benchmark. This command never downloads models.
public struct RuntimeEvaluationPlan: Codable, Sendable {
    public var version: Int
    /// Omit for full qualification; screening uses two pairs and one warmup per worker.
    public var mode: String? = nil
    public var isScreening: Bool { mode == "screening" }
    public var runtime: EvaluationBatchPlan.Artifact
    public var metal: EvaluationBatchPlan.Artifact
    public var baseline: Model
    public var candidate: Model
    public var workloads: [Workload]
    public var minimumFreeBytes: UInt64
    public var exclusiveProcessIDs: [Int32]
    public var timeoutSeconds: Double

    public struct Model: Codable, Sendable {
        public var checkpoint: EvaluationBatchPlan.Model
        public var qualityIdentity: EvaluationBatchPlan.Artifact
        public var qualityReport: EvaluationBatchPlan.Artifact
        public var qualityManifest: EvaluationBatchPlan.Artifact
    }
    public struct Workload: Codable, Sendable {
        public var name: String
        public var prompt: String
        public var expectedPromptTokens: Int
        public var contextLength: Int
    }

    public func validate() throws {
        guard version == 1, mode == nil || mode == "qualification" || mode == "screening",
            minimumFreeBytes >= 100 * 1024 * 1024 * 1024,
            timeoutSeconds >= 1, timeoutSeconds <= 7200, !workloads.isEmpty,
            Set(workloads.map(\.name)).count == workloads.count,
            baseline.checkpoint.family == candidate.checkpoint.family, !baseline.checkpoint.family.isEmpty,
            !baseline.checkpoint.recipe.isEmpty, !candidate.checkpoint.recipe.isEmpty,
            baseline.checkpoint.path.hasPrefix("/"), candidate.checkpoint.path.hasPrefix("/"),
            !baseline.checkpoint.sourceRevision.isEmpty, !candidate.checkpoint.sourceRevision.isEmpty,
            exclusiveProcessIDs.allSatisfy({ $0 > 0 }),
            workloads.allSatisfy({
                !$0.prompt.isEmpty && !$0.name.isEmpty && $0.expectedPromptTokens > 0
                    && $0.contextLength >= $0.expectedPromptTokens + 256 && $0.contextLength <= 131072
            })
        else { throw EvaluationFailure.invalidSuite("Invalid native runtime plan, workload, capacity or reserve.") }
        for pin in [
            runtime, metal, baseline.qualityIdentity, candidate.qualityIdentity, baseline.qualityReport,
            candidate.qualityReport, baseline.qualityManifest, candidate.qualityManifest,
        ] {
            guard pin.path.hasPrefix("/"), pin.bytes > 0, pin.sha256.count == 64, pin.sha256.allSatisfy(\.isHexDigit)
            else {
                throw EvaluationFailure.invalidSuite("Native runtime plans require complete artifact pins.")
            }
        }
        guard
            URL(fileURLWithPath: runtime.path).deletingLastPathComponent().appendingPathComponent("mlx.metallib").path
                == URL(fileURLWithPath: metal.path).standardizedFileURL.path
        else { throw EvaluationFailure.invalidSuite("Pinned Metal library must be beside the native executable.") }
    }
}

/// Diagnostic paired rate/latency summary. Qualification requires all three phases.
public struct RuntimeMetricSummary: Codable, Sendable {
    public var baselineMedian: Double
    public var candidateMedian: Double
    public var pairedMedianRatio: Double
    public var interval95: [Double]
    public var baselineDrift: Double
    public var candidateDrift: Double
    public var orderEffect: Double
    public var rejectionReasons: [String]
    public var qualified: Bool { rejectionReasons.isEmpty }
}

/// Fixed eight-pair AB/BA analysis; no outlier removal or adaptive warmups.
public enum RuntimeStatistics {
    public static func summarize(baseline: [Double], candidate: [Double], latency: Bool, control: Bool) throws
        -> RuntimeMetricSummary
    {
        guard baseline.count == 8, candidate.count == 8,
            (baseline + candidate).allSatisfy({ $0.isFinite && $0 > 0 })
        else { throw EvaluationFailure.transport("Eight complete positive finite pairs are required.") }
        let ratios = zip(baseline, candidate).map { latency ? $0 / $1 : $1 / $0 }
        func drift(_ values: [Double]) -> Double {
            abs(median(Array(values.suffix(4))) / median(Array(values.prefix(4))) - 1)
        }
        let orderEffect = abs(median([0, 2, 4, 6].map { ratios[$0] }) / median([1, 3, 5, 7].map { ratios[$0] }) - 1)
        // Versioned deterministic PRNG: bootstrap results do not depend on the host random source.
        var state: UInt64 = 20_261_002
        var bootstrap: [Double] = []
        for _ in 0..<2000 {
            var sample: [Double] = []
            for _ in 0..<8 {
                state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                sample.append(ratios[Int((state >> 32) % 8)])
            }
            bootstrap.append(median(sample))
        }
        bootstrap.sort()
        let interval = [bootstrap[49], bootstrap[1949]]
        let ratio = median(ratios)
        var reasons: [String] = []
        if drift(baseline) > 0.10 { reasons.append("Baseline drift exceeds 10%.") }
        if drift(candidate) > 0.10 { reasons.append("Candidate drift exceeds 10%.") }
        if orderEffect > 0.10 { reasons.append("AB/BA order effect exceeds 10%.") }
        if control && (ratio < 0.9 || ratio > 1.1 || interval[0] > 1 || interval[1] < 1) {
            reasons.append("Same-model ratio or interval fails equivalence controls.")
        }
        return .init(
            baselineMedian: median(baseline), candidateMedian: median(candidate), pairedMedianRatio: ratio,
            interval95: interval, baselineDrift: drift(baseline), candidateDrift: drift(candidate),
            orderEffect: orderEffect, rejectionReasons: reasons)
    }

    private static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        return (sorted[(sorted.count - 1) / 2] + sorted[sorted.count / 2]) / 2
    }
}

/// Typed native reports enforce full work and explicit no-cache execution.
struct RuntimeNativeReport: Decodable {
    var status: String
    var prompt: String
    var engine: String
    var modelPath: String
    var modelImplementation: String
    var contextLength: Int
    var prefillStepSize: Int
    var kvCompression: String
    var promptReusePolicy: String
    var temperature: Double
    var topP: Double
    var requestedTokens: Int
    var warmupCount: Int
    var measuredTrials: Int
    var warmups: [Trial]
    var trials: [Trial]
    var wiredMemoryScope: String?
    var wiredMemorySessionEligible: Bool?
    var wiredMemorySessionFailures: [String]?
    var wiredMemorySession: RuntimeWiredSession?
    var wiredMemoryAfterMeasured: RuntimeWiredState?
    var wiredMemoryRequests: [RuntimeWiredRequest]?

    struct Trial: Decodable {
        var content: String
        var reasoning: String
        var mode: String
        var promptTokenIdFingerprint: String
        var timeToFirstTokenMilliseconds: Double
        var metrics: Metrics
    }
    struct Metrics: Decodable {
        var cachedPromptTokenCount: Int
        var generationTokenCount: Int
        var prefilledPromptTokenCount: Int
        var promptTokenCount: Int
        var promptTokensPerSecond: Double
        var tokensPerSecond: Double
        var stopReason: String
    }

    func validate(
        model: String, workload: RuntimeEvaluationPlan.Workload, expectedMeasuredTrials: Int = 1,
        expectedWarmups: Int = 8
    ) throws {
        guard status == "measured", prompt == workload.prompt, engine == "metal", !modelImplementation.isEmpty,
            URL(fileURLWithPath: modelPath).standardizedFileURL == URL(fileURLWithPath: model).standardizedFileURL,
            contextLength == workload.contextLength, prefillStepSize == 512, kvCompression == "none",
            promptReusePolicy == "disabled", temperature == 0, topP == 1,
            requestedTokens == 256, warmupCount == expectedWarmups, expectedWarmups > 0, expectedMeasuredTrials > 0,
            measuredTrials == expectedMeasuredTrials, warmups.count == expectedWarmups,
            trials.count == expectedMeasuredTrials
        else { throw EvaluationFailure.transport("Native runtime settings differ from the fixed plan.") }
        for trial in warmups + trials {
            let m = trial.metrics
            guard trial.mode == "target_only", !trial.promptTokenIdFingerprint.isEmpty,
                trial.promptTokenIdFingerprint == trials[0].promptTokenIdFingerprint,
                m.promptTokenCount == workload.expectedPromptTokens, m.prefilledPromptTokenCount == m.promptTokenCount,
                m.cachedPromptTokenCount == 0, m.generationTokenCount == 256, m.stopReason == "length",
                [m.tokensPerSecond, m.promptTokensPerSecond, trial.timeToFirstTokenMilliseconds].allSatisfy({
                    $0.isFinite && $0 > 0
                })
            else { throw EvaluationFailure.transport("Incomplete native work, cached tokens or invalid metrics.") }
        }
    }
}
