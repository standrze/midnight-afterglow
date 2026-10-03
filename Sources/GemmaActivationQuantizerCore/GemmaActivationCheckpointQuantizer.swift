import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MistralActivationScaleSearchCore
import QuantizerSupport
import WickModelSupport

/// Candidate provenance. Proxy improvements are not generated-task accuracy or
/// runtime evidence. Source verification covers each selected LS2 projection;
/// all unselected payloads and template sidecars are preserved exactly.
public struct GemmaAWSSCheckpointReport: Encodable {
    public let format = "gemma4_awss_checkpoint_v1"
    public let status = "experimental_unbenchmarked_candidate"
    public let algorithm = "gemma4_q4_g64_activation_weighted_ls2_second_pass"
    public let objective = "group_normalized_diagonal_linear_output_squared_error_proxy"
    public let source: GemmaActivationSourceIdentity
    public let templateIdentity: [String: String]
    public let calibration: GemmaActivationProvenance
    public let development: GemmaActivationProvenance
    public let statisticsFingerprints: [String: String]
    public let minimumExpertPositions: Int
    public let selectedModules: [String]
    public let retainedExperts: [String: [Int]]
    public let retainedModuleReasons: [String: String]
    public let diagnostics: [String: [Int: ActivationWeightedScaleSearchDiagnostics]]
    public let outputWeightsFingerprint: String
    public let outputTensorCount: Int
    public let sourceVerification = "selected_projection_exact_ls2_regeneration_from_bf16"
    public let preservation = "all_unselected_tensor_payloads_and_flat_template_sidecars_exact"
}

/// Converts a complete indexed native Gemma template using one BF16 source
/// layer at a time. Uses independent native projection inventory, disjoint
/// fit/dev statistics, exact selected LS2 regeneration, and atomic publication.
public enum GemmaActivationCheckpointQuantizer {
    public static func run(
        source: URL, template: URL, calibration: URL, development: URL, destination: URL,
        modules: [String] = [], minimumExpertPositions: Int = 32,
        maximumShardBytes: Int = 1_073_741_824, overwrite: Bool = false
    ) throws -> GemmaAWSSCheckpointReport {
        guard minimumExpertPositions > 0, maximumShardBytes > 0 else {
            throw QuantizerInputError("Require positive coverage and shard byte limits.")
        }
        let source = source.standardizedFileURL.resolvingSymlinksInPath()
        let template = template.standardizedFileURL.resolvingSymlinksInPath()
        let destination = destination.standardizedFileURL
        try QuantizerOutputTransaction.validate(
            sourceDirectory: template, destinationDirectory: destination, overwrite: overwrite)
        for statistics in [calibration, development] {
            try QuantizerOutputTransaction.validate(
                sourceDirectory: statistics, destinationDirectory: destination, overwrite: overwrite)
        }
        let statisticsHashes = [
            "calibration": GemmaActivationProvenance.fingerprint(try Data(contentsOf: calibration)),
            "development": GemmaActivationProvenance.fingerprint(try Data(contentsOf: development)),
        ]
        let transaction = try QuantizerOutputTransaction(
            sourceDirectory: source, destinationDirectory: destination, overwrite: overwrite)
        defer { transaction.cleanup() }
        let configData = try Data(contentsOf: source.appendingPathComponent("config.json"))
        let config = try Gemma4CalibrationConfiguration(data: configData)
        guard let root = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw QuantizerInputError("Expected BF16 source configuration object.")
        }
        let text = root["text_config"] as? [String: Any] ?? root
        let topK = text["enable_moe_block"] as? Bool == true ? (text["top_k_experts"] as? Int ?? 0) : 0
        guard topK == 0 || Device.defaultDevice().deviceType != .cpu else {
            throw QuantizerInputError("BF16 Gemma MoE fitting requires the GPU backend.")
        }
        let baseline = try GemmaTemplate(directory: template, source: source, sourceConfig: configData)
        let sourceReader = try SelectiveSafetensorsReader(directory: source)
        let templateReader = try SelectiveSafetensorsReader(directory: template)
        var shapes = [String: [Int]]()
        // The manifest cannot nominate its own inventory. Obtain it from native
        // validated BF16 blocks, discarding each block before opening the next.
        for layer in 0..<config.layerCount {
            try pooled {
                let block = try sourceBlock(sourceReader, configuration: config, layer: layer)
                for (path, width) in block.denseProjectionWidths { shapes[path] = [width] }
                let arrays = Dictionary(uniqueKeysWithValues: block.module.parameters().flattened())
                for path in block.routedProjectionPaths {
                    let relative = String(path.dropFirst(block.modulePrefix.count + 1)) + ".weight"
                    guard let weight = arrays[relative] else {
                        throw QuantizerInputError("Missing native expert weight.")
                    }
                    shapes[path] = [weight.dim(0), weight.dim(-1)]
                }
            }
            Memory.clearCache()
        }
        let head = config.moduleRoot == "model" ? "lm_head" : "language_model.lm_head"
        shapes[head] = [config.hiddenSize]
        let fit = try GemmaActivationStatistics.load(
            from: calibration, source: source, expectedProjectionShapes: shapes,
            expectedExpertsPerToken: topK, expectedMinimumExpertPositions: minimumExpertPositions)
        let dev = try GemmaActivationStatistics.load(
            from: development, source: source, expectedProjectionShapes: shapes,
            expectedExpertsPerToken: topK, expectedMinimumExpertPositions: minimumExpertPositions)
        let fitter = try GemmaActivationWeightedScaleSearch(calibration: fit, development: dev)
        let decoderLS2 = Set(shapes.keys.filter { $0 != head && !$0.hasSuffix(".router.proj") })
            .intersection(baseline.recipe.q4ScaleSearchModules)
        let eligible = Set(decoderLS2.filter { (try? baseline.requireQ4G64($0)) != nil })
        let selected = modules.isEmpty ? eligible.sorted() : modules.sorted()
        guard !selected.isEmpty, Set(selected).count == selected.count, Set(selected).isSubset(of: eligible) else {
            throw QuantizerInputError("Require unique native LS2 decoder projections; routers and heads are protected.")
        }
        for path in selected { try baseline.requireQ4G64(path) }
        try FileManager.default.createDirectory(at: transaction.stagingDirectory, withIntermediateDirectories: false)
        try baseline.copySidecars(to: transaction.stagingDirectory)
        let writer = GemmaCheckpointWriter(directory: transaction.stagingDirectory, maximumBytes: maximumShardBytes)
        var written = Set<String>()
        var selectedPayloadFingerprints = [String: String]()
        var diagnostics = [String: [Int: ActivationWeightedScaleSearchDiagnostics]]()
        var retained = [String: [Int]]()
        var reasons = [String: String]()
        for layer in 0..<config.layerCount {
            let prefix = config.moduleRoot + ".layers.\(layer)."
            let layerSelected = selected.filter { $0.hasPrefix(prefix) }
            try pooled {
                // Only selected layers require source matrices for fitting.
                let sourceArrays: [String: MLXArray]
                if layerSelected.isEmpty {
                    sourceArrays = [:]
                } else {
                    let block = try sourceBlock(sourceReader, configuration: config, layer: layer)
                    sourceArrays = Dictionary(uniqueKeysWithValues: block.module.parameters().flattened())
                }
                for path in layerSelected {
                    try pooled {
                        let key = String(path.dropFirst(prefix.count)) + ".weight"
                        guard let original = sourceArrays[key], original.dtype == .bfloat16 else {
                            throw QuantizerInputError("Missing BF16 source projection: \(path)")
                        }
                        let packed = try templateReader.read(path + ".weight")
                        let scales = try templateReader.read(path + ".scales")
                        let biases = try templateReader.read(path + ".biases")
                        guard original.dim(-1) % 64 == 0 else { throw QuantizerInputError("Invalid G64 source width.") }
                        try MLX.checkedEval(original)
                        guard MLX.all(MLX.isFinite(original)).item(Bool.self) else {
                            throw QuantizerInputError("Non-finite BF16 source projection: \(path)")
                        }
                        let regenerated = q4AffineScaleSearchQuantized(original, rowBatchSize: 256)
                        guard try exact(regenerated.weight, packed), try exact(regenerated.scales, scales),
                            try exact(regenerated.biases, biases)
                        else {
                            throw QuantizerInputError("Template does not reproduce the BF16 source LS2 grid: \(path)")
                        }
                        let result = try fitter.rescore(
                            modulePath: path, sourceWeight: original, templateWeight: packed,
                            templateScales: scales, templateBiases: biases, bits: 4, groupSize: 64)
                        diagnostics[path] = result.diagnostics
                        retained[path] = result.retainedTemplateExperts
                        reasons[path] = result.retainedModuleReason
                        for (suffix, tensor) in [
                            ("weight", result.weight), ("scales", result.scales), ("biases", result.biases),
                        ] {
                            let name = path + "." + suffix
                            selectedPayloadFingerprints[name] = GemmaActivationProvenance.fingerprint(
                                tensor.asData(access: .noCopyIfContiguous).data)
                            try writer.append(name, tensor)
                            written.insert(name)
                        }
                    }
                }
                for name in templateReader.keys where name.hasPrefix(prefix) && !written.contains(name) {
                    try writer.append(name, templateReader.read(name))
                    written.insert(name)
                }
            }
            Memory.clearCache()
        }
        for name in templateReader.keys where !written.contains(name) {
            try pooled { try writer.append(name, templateReader.read(name)) }
            written.insert(name)
        }
        try writer.finish()
        let exported = try SelectiveSafetensorsReader(directory: transaction.stagingDirectory)
        guard exported.keys == templateReader.keys else {
            throw QuantizerInputError("Output tensor inventory changed.")
        }
        let changedNames = Set(selected.flatMap { path in ["weight", "scales", "biases"].map { path + "." + $0 } })
        for name in templateReader.keys {
            try pooled {
                let expected = try templateReader.description(for: name)
                let actual = try exported.description(for: name)
                guard expected.shape == actual.shape, expected.dtype == actual.dtype, expected.bytes == actual.bytes
                else {
                    throw QuantizerInputError("Output stored geometry changed: \(name)")
                }
                if changedNames.contains(name) {
                    let payload = try exported.read(name)
                    guard
                        GemmaActivationProvenance.fingerprint(payload.asData(access: .noCopyIfContiguous).data)
                            == selectedPayloadFingerprints[name]
                    else {
                        throw QuantizerInputError("Selected output payload does not match its fitted grid: \(name)")
                    }
                } else {
                    guard try exact(templateReader.read(name), exported.read(name)) else {
                        throw QuantizerInputError("Unselected payload changed: \(name)")
                    }
                }
            }
            Memory.clearCache()
        }
        for (name, hash) in baseline.identity where name != "weights" && name != "model.safetensors.index.json" {
            guard
                try GemmaActivationProvenance.fingerprint(
                    Data(contentsOf: transaction.stagingDirectory.appendingPathComponent(name))) == hash
            else {
                throw QuantizerInputError("Template sidecar changed: \(name)")
            }
        }
        let report = GemmaAWSSCheckpointReport(
            source: fit.provenance.source, templateIdentity: baseline.identity,
            calibration: fit.provenance, development: dev.provenance,
            statisticsFingerprints: statisticsHashes, minimumExpertPositions: minimumExpertPositions,
            selectedModules: selected, retainedExperts: retained, retainedModuleReasons: reasons,
            diagnostics: diagnostics,
            outputWeightsFingerprint: try IndexedSafetensorsFingerprint.compute(
                directory: transaction.stagingDirectory),
            outputTensorCount: exported.keys.count)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(
            to: transaction.stagingDirectory.appendingPathComponent("gemma-awss-quantization.json"))
        try fit.provenance.source.requireUnchanged()
        try baseline.requireUnchanged()
        guard
            try GemmaActivationProvenance.fingerprint(Data(contentsOf: calibration)) == statisticsHashes["calibration"],
            try GemmaActivationProvenance.fingerprint(Data(contentsOf: development)) == statisticsHashes["development"]
        else { throw QuantizerInputError("Fit/development statistics changed during conversion.") }
        try QuantizerOutputTransaction.validate(
            sourceDirectory: template, destinationDirectory: destination, overwrite: overwrite)
        if let backup = try transaction.commit() { print("Previous output retained at \(backup.path)") }
        return report
    }

    private static func sourceBlock(
        _ reader: SelectiveSafetensorsReader, configuration: Gemma4CalibrationConfiguration, layer: Int
    ) throws -> Gemma4CalibrationBlock {
        let prefixes = ["model.language_model", "language_model.model", "language_model", "model"]
            .map { $0 + ".layers.\(layer)." }
        let keys = reader.keys.filter { name in prefixes.contains { name.hasPrefix($0) } }
        return try Gemma4CalibrationBlock(
            configuration: configuration, layerIndex: layer, sourceWeights: reader.read(keys: keys))
    }

    private static func exact(_ lhs: MLXArray, _ rhs: MLXArray) throws -> Bool {
        guard lhs.shape == rhs.shape, lhs.dtype == rhs.dtype else { return false }
        try MLX.checkedEval(lhs, rhs)
        return lhs.asData().data == rhs.asData().data
    }

    private static func pooled<T>(_ body: () throws -> T) rethrows -> T {
        #if canImport(ObjectiveC)
            return try autoreleasepool(invoking: body)
        #else
            return try body()
        #endif
    }
}
