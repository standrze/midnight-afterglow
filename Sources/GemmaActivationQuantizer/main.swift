import ArgumentParser
import Foundation
import GemmaActivationQuantizerCore
import MLX
import QuantizerSupport
import WickModelSupport

@main
struct GemmaActivationQuantizerCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "wick-gemma-awss-quantize",
        abstract: "Refine a native Gemma 4 LS2 checkpoint using source-bound, independent fit/development statistics.")

    @Argument(help: "Original indexed BF16 Gemma 4 source.") var source: String
    @Argument(help: "Indexed Wick LS2 affine Q4/G64 template from this source.") var template: String
    @Argument(help: "Source-bound calibration statistics.") var calibration: String
    @Argument(help: "Independent source-bound development statistics.") var development: String
    @Argument(help: "Destination checkpoint outside source and template.") var output: String
    @Option(
        name: .customLong("module"),
        help: "Repeat for selected decoder paths; default selects all eligible LS2 decoder paths.")
    var modules: [String] = []
    @Option(name: .customLong("minimum-expert-positions")) var minimumExpertPositions = 32
    @Option(name: .customLong("max-shard-gib")) var maximumShardGiB = 1.0
    @Flag(help: "Atomically replace an existing destination after all checks pass.") var overwrite = false
    @Flag(help: "Use CPU for dense fixtures; BF16 MoE requires GPU.") var cpu = false

    mutating func run() throws {
        let bytes = try QuantizerShardSize.bytes(fromGiB: maximumShardGiB)
        guard let shardBytes = Int(exactly: bytes) else {
            throw ValidationError("Shard size exceeds host integer capacity.")
        }
        let limits = try MLXResourceLimits.resolve(
            for: cpu ? .cpu : compiledEngine, physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory)
        try Device.withDefaultDevice(cpu ? .cpu : Device.defaultDevice()) {
            try MLXResourceGuard.apply(limits)
            let report = try GemmaActivationCheckpointQuantizer.run(
                source: url(source), template: url(template), calibration: url(calibration),
                development: url(development),
                destination: url(output), modules: modules, minimumExpertPositions: minimumExpertPositions,
                maximumShardBytes: shardBytes, overwrite: overwrite)
            print(
                "Published unbenchmarked candidate: \(report.selectedModules.count) selected projections, \(report.outputTensorCount) tensors."
            )
            print("No coding/cybersecurity accuracy or runtime improvement is established by conversion.")
        }
    }
}

private func url(_ path: String) -> URL {
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
