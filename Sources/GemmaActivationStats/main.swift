import ArgumentParser
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import MistralActivationScaleSearchCore
import Tokenizers
import WickModelSupport

@main
struct GemmaActivationStatsCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "wick-gemma-activation-stats",
        abstract: "Collect source-bound BF16 Gemma 4 coding/cybersecurity activation statistics, one layer at a time.")

    @Argument(help: "Local indexed, unquantized BF16 Gemma 4 source checkpoint.")
    var model: String

    @Argument(help: "JSONL rows with id, category (coding/cybersecurity), source_family and text.")
    var corpus: String

    @Argument(help: "New .safetensors output outside the source checkpoint; parent must exist.")
    var output: String

    @Option(name: .customLong("segment-tokens"), help: "Tokens per independent sample segment (1...2048).")
    var segmentTokens = 512

    @Option(name: .customLong("maximum-total-tokens"), help: "Deterministic prefix cap; zero selects all tokens.")
    var maximumTotalTokens = 0

    @Option(
        name: .customLong("minimum-expert-positions"),
        help: "Preregistered expert coverage threshold; under-covered experts retain their template.")
    var minimumExpertPositions = 32

    @Option(name: .customLong("spool-directory"), help: "Parent for the private temporary activation spool.")
    var spoolDirectory: String?

    @Option(
        name: .customLong("projection-input-plan"),
        help: "JSON plan selecting bounded layer/expert inputs for covariance research.")
    var projectionInputPlan: String?

    @Option(
        name: .customLong("projection-input-output"),
        help: "New directory for the optional source-bound projection input export.")
    var projectionInputOutput: String?

    @Flag(help: "Use CPU for dense fixtures. BF16 MoE requires the GPU backend.")
    var cpu = false

    mutating func validate() throws {
        guard (projectionInputPlan == nil) == (projectionInputOutput == nil) else {
            throw ValidationError(
                "Projection input export requires both --projection-input-plan and --projection-input-output.")
        }
        guard (1...2048).contains(segmentTokens), maximumTotalTokens >= 0,
            maximumTotalTokens <= Int(Int32.max), minimumExpertPositions > 0
        else {
            throw ValidationError(
                "Require segment tokens 1...2048, token cap 0...Int32.max and positive expert coverage.")
        }
    }

    mutating func run() async throws {
        let source = localURL(model).resolvingSymlinksInPath()
        let corpusURL = localURL(corpus)
        let destination = localURL(output)
        let parent = destination.deletingLastPathComponent().resolvingSymlinksInPath()
        guard destination.pathExtension == "safetensors",
            !FileManager.default.fileExists(atPath: destination.path),
            parent != source, !parent.path.hasPrefix(source.path + "/"),
            FileManager.default.fileExists(atPath: parent.path)
        else {
            throw ValidationError("Output must be a new .safetensors file in an existing directory outside the source.")
        }
        let configData = try Data(contentsOf: source.appendingPathComponent("config.json"))
        let configuration = try Gemma4CalibrationConfiguration(data: configData)
        guard let root = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw ValidationError("Expected a source configuration object.")
        }
        let text = root["text_config"] as? [String: Any] ?? root
        let moe = text["enable_moe_block"] as? Bool == true
        let device: Device = cpu ? .cpu : Device.defaultDevice()
        guard !moe || device.deviceType != .cpu else {
            throw ValidationError(
                "BF16 Gemma MoE calibration requires a GPU backend; --cpu supports dense fixtures only.")
        }
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            guard FileManager.default.fileExists(atPath: source.appendingPathComponent(name).path) else {
                throw ValidationError("Local calibration requires \(name); this command does not download assets.")
            }
        }
        let inputExport: GemmaProjectionInputExport?
        if let projectionInputPlan, let projectionInputOutput {
            inputExport = try GemmaProjectionInputExport(
                planURL: localURL(projectionInputPlan), destination: localURL(projectionInputOutput),
                source: source, configuration: configuration)
        } else {
            inputExport = nil
        }
        let corpusData = try Data(contentsOf: corpusURL)
        let samples = try parseCorpus(corpusData)
        print("Fingerprinting complete indexed source content before collection.")
        let identity = try GemmaActivationSourceIdentity.capture(source: source)
        let tokenizer = try await #huggingFaceTokenizerLoader().load(from: source)
        var tokenSamples = [[Int]]()
        var segments = [[Int]]()
        var families = [String]()
        var observed = 0
        for sample in samples {
            if maximumTotalTokens > 0, observed == maximumTotalTokens { break }
            let encoded = tokenizer.encode(text: sample.text, addSpecialTokens: true)
            guard !encoded.isEmpty else { throw ValidationError("Corpus sample encoded to no tokens.") }
            let remaining = maximumTotalTokens == 0 ? encoded.count : maximumTotalTokens - observed
            let selected = Array(encoded.prefix(remaining))
            guard selected.count <= Int(Int32.max) - observed else {
                throw ValidationError("Observed token count overflow.")
            }
            tokenSamples.append(selected)
            families.append(sample.sourceFamily)
            observed += selected.count
            for offset in stride(from: 0, to: selected.count, by: segmentTokens) {
                segments.append(Array(selected[offset..<min(offset + segmentTokens, selected.count)]))
            }
        }
        let provenance = try GemmaActivationProvenance(
            source: identity, corpus: corpusData, tokenSamples: tokenSamples,
            tokenSegments: segments, sourceFamilies: families)
        let spool = spoolDirectory.map(localURL) ?? FileManager.default.temporaryDirectory
        let threshold = minimumExpertPositions
        let expertsPerToken = moe ? (text["top_k_experts"] as? Int ?? 0) : 0
        let limits = try MLXResourceLimits.resolve(
            for: cpu ? .cpu : compiledEngine, physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory)
        try Device.withDefaultDevice(device) {
            try MLXResourceGuard.apply(limits)
            print(
                "Collecting \(observed) tokens in \(segments.count) independent segments; plain text, special tokens enabled."
            )
            let collected = try Gemma4ActivationCollector.collect(
                source: source, tokenSegments: segments, spoolParent: spool, minimumExpertPositions: threshold,
                projectionInputRecorder: inputExport.map { exporter in
                    { layer in try exporter.recorder(layer: layer) }
                },
                consumeProjectionInputs: inputExport.map { exporter in
                    { layer, inputs in try exporter.consume(layer: layer, inputs: inputs) }
                })
            guard collected.observedTokenCount == provenance.observedTokenCount,
                collected.segmentTokenCounts == provenance.segmentTokenCounts
            else { throw ValidationError("Collection token coverage differs from provenance.") }
            var moments = Dictionary(uniqueKeysWithValues: collected.dense.map { ($0.path, $0.secondMoments) })
            for expert in collected.experts {
                guard moments[expert.path] == nil else { throw ValidationError("Duplicate projection statistics.") }
                moments[expert.path] = expert.secondMoments
            }
            let statistics = try GemmaActivationStatistics(
                moments: moments,
                expertCounts: Dictionary(
                    uniqueKeysWithValues: collected.experts.map { ($0.path, $0.expertPositionCounts) }),
                provenance: provenance, minimumExpertPositions: threshold, expertsPerToken: expertsPerToken)
            try statistics.write(to: destination)
            try inputExport?.finish(provenance: provenance)
            print(
                "Saved \(collected.dense.count) dense and \(collected.experts.count) expert projections to \(destination.path)."
            )
            print("Statistics are calibration evidence; no model quality, speed or memory improvement is established.")
        }
    }
}

private struct CorpusSample: Decodable {
    let id: String
    let category: String
    let sourceFamily: String
    let text: String

    enum CodingKeys: String, CodingKey {
        case id, category, text
        case sourceFamily = "source_family"
    }
}

private func parseCorpus(_ data: Data) throws -> [CorpusSample] {
    guard let text = String(data: data, encoding: .utf8) else { throw ValidationError("Corpus must be UTF-8 JSONL.") }
    var seen = Set<String>()
    let samples = try text.split(whereSeparator: \.isNewline).enumerated().map { index, line in
        let sample = try JSONDecoder().decode(CorpusSample.self, from: Data(line.utf8))
        guard !sample.id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            seen.insert(sample.id).inserted, ["coding", "cybersecurity"].contains(sample.category),
            !sample.sourceFamily.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            !sample.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { throw ValidationError("Invalid or duplicate corpus sample on nonempty line \(index + 1).") }
        return sample
    }
    guard !samples.isEmpty else { throw ValidationError("Corpus must contain at least one sample.") }
    return samples
}

private func localURL(_ path: String) -> URL {
    URL(fileURLWithPath: (path as NSString).expandingTildeInPath).standardizedFileURL
}

private var compiledEngine: ModelEngine {
    #if MLX_METAL_BACKEND
        .metal
    #elseif MLX_CUDA_BACKEND
        .cuda
    #else
        .cpu
    #endif
}
