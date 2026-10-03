import ArgumentParser
import Foundation
import MLX
import MLXLMCommon
import MistralActivationScaleSearchCore

private struct SafetensorsIndex: Decodable {
    var weightMap: [String: String]

    enum CodingKeys: String, CodingKey {
        case weightMap = "weight_map"
    }
}

private struct SourceConfiguration: Decodable {
    var numberOfExperts: Int

    enum CodingKeys: String, CodingKey {
        case numberOfExperts = "num_experts"
    }
}

private struct RescoreProvenance: Encodable {
    var format = 1
    var status = "experimental_measured_candidate"
    var algorithm = "q4r8_affine_scale_search_ls2"
    var createdAt: String
    var sourceModel: String
    var templateModel: String
    var bits = 4
    var groupSize = 64
    var searchFactors: [Double] = [
        0.75, 0.8125, 0.875, 0.9375, 1,
        1.0625, 1.125, 1.1875, 1.25,
    ]
    var biasRefinementIterations = 1
    var jointAffineRefinementIterations = 2
    var q4ModulesRescored: Int
    var q8ModulesPreserved: Int
    var standardQ4EmbeddingsPreserved: Int
    var expertBatchSize: Int
    var activationStatistics: String? = nil
    var sourceWeightFingerprint: String? = nil
    var sourceWeightFingerprintMethod: String? = nil
    var validationStatistics: String? = nil
    var calibrationCorpusFingerprint: String? = nil
    var validationCorpusFingerprint: String? = nil
    var minimumExpertPositions: Int? = nil
    var retainedTemplateExperts: [String: [Int]]? = nil
    var activationDiagnostics: [String: ActivationWeightedScaleSearchDiagnostics]? = nil
    var peakMLXMemoryBytes: Int = 0
    var preflightPeakMLXMemoryBytes: Int = 0
    var sourceTensorsReleased: Int = 0
    var sourceTensorsRemaining: Int = 0
    var quantizationDevice: String = "unspecified"
    var moduleQuantizationOverrides: [String: LagunaQuantizationGeometry]? = nil
    var templateTensorBytes: Int = 0
    var outputTensorBytes: Int = 0

    enum CodingKeys: String, CodingKey {
        case format, status, algorithm, bits
        case createdAt = "created_at"
        case sourceModel = "source_model"
        case templateModel = "template_model"
        case groupSize = "group_size"
        case searchFactors = "search_factors"
        case biasRefinementIterations = "bias_refinement_iterations"
        case jointAffineRefinementIterations = "joint_affine_refinement_iterations"
        case q4ModulesRescored = "q4_modules_rescored"
        case q8ModulesPreserved = "q8_modules_preserved"
        case standardQ4EmbeddingsPreserved = "standard_q4_embeddings_preserved"
        case expertBatchSize = "expert_batch_size"
        case activationStatistics = "activation_statistics"
        case sourceWeightFingerprint = "source_weight_fingerprint"
        case sourceWeightFingerprintMethod = "source_weight_fingerprint_method"
        case validationStatistics = "validation_statistics"
        case calibrationCorpusFingerprint = "calibration_corpus_fingerprint"
        case validationCorpusFingerprint = "validation_corpus_fingerprint"
        case minimumExpertPositions = "minimum_expert_positions"
        case retainedTemplateExperts = "retained_template_experts"
        case activationDiagnostics = "activation_diagnostics"
        case peakMLXMemoryBytes = "peak_mlx_memory_bytes"
        case preflightPeakMLXMemoryBytes = "preflight_peak_mlx_memory_bytes"
        case sourceTensorsReleased = "source_tensors_released"
        case sourceTensorsRemaining = "source_tensors_remaining"
        case quantizationDevice = "quantization_device"
        case moduleQuantizationOverrides = "module_quantization_overrides"
        case templateTensorBytes = "template_tensor_bytes"
        case outputTensorBytes = "output_tensor_bytes"
    }
}

private struct QuantizedArrays {
    var weight: MLXArray
    var scales: MLXArray
    var biases: MLXArray
}

private enum RescoreError: Error, LocalizedError {
    case invalidInput(String)
    case incompatibleTemplate(String)
    case missingTensor(String)

    var errorDescription: String? {
        switch self {
        case .invalidInput(let message): "Invalid rescore input: \(message)"
        case .incompatibleTemplate(let message): "Incompatible Q4R8 template: \(message)"
        case .missingTensor(let key): "Missing source tensor: \(key)"
        }
    }
}

/// Counts consumers rather than assuming every output projection owns unique source tensors.
///
/// In particular, fused and split gate/up outputs can overlap.
struct LagunaSourceUsePlan {
    private var keysByModule: [String: [String]]
    private(set) var remainingUses: [String: Int] = [:]

    init(keysByModule: [String: [String]]) {
        self.keysByModule = keysByModule.mapValues { Array(Set($0)).sorted() }
        for keys in self.keysByModule.values {
            for key in keys { remainingUses[key, default: 0] += 1 }
        }
    }

    mutating func finish(module: String) throws -> [String] {
        guard let keys = keysByModule.removeValue(forKey: module) else {
            throw RescoreError.invalidInput("unknown or repeated source consumer \(module)")
        }
        var released = [String]()
        for key in keys {
            guard let count = remainingUses[key], count > 0 else {
                throw RescoreError.invalidInput("invalid source use count for \(key)")
            }
            if count == 1 {
                remainingUses.removeValue(forKey: key)
                released.append(key)
            } else {
                remainingUses[key] = count - 1
            }
        }
        return released
    }
}

public struct LagunaScaleSearchRescorer: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: CommandLine.arguments.first?.split(separator: "/").last.map(String.init)
            ?? "wick-laguna-q4r8-rescore",
        abstract:
            "Recompute Laguna Q4 tensors with affine ScaleSearch while streaming through a standard Q4R8 layout."
    )

    @Argument(help: "Original unquantized Laguna safetensors directory.")
    var source: String

    @Argument(help: "Read-only standard MLX Q4R8 checkpoint used for layout and preserved Q8 arrays.")
    var template: String

    @Argument(help: "New destination directory; it must not already exist.")
    var destination: String

    @Option(
        name: .customLong("expert-batch"),
        help: "Number of routed experts to quantize in each bounded GPU batch."
    )
    var expertBatch = 16

    @Option(
        name: .customLong("group-size"),
        help: "Output Q4 group size, 64 or 128. G128 preserves Q8/G64 routers and the Q4/G64 embedding.")
    var groupSize = 64

    @Flag(name: .customLong("standard-q4"), help: "Use native affine Q4 without ScaleSearch as a grouping control.")
    var standardQ4 = false

    @Option(
        name: .customLong("activation-stats"),
        help: "Laguna expert-conditional BF16 calibration safetensors; enables activation-weighted refinement.")
    var activationStats: String?

    @Option(
        name: .customLong("validation-stats"),
        help: "Disjoint Laguna dev statistics; covered candidates must not worsen its stored-grid objective.")
    var validationStats: String?

    @Flag(help: "Validate source/template identity and mappings without writing a destination.")
    var preflightOnly = false

    @Flag(help: "Run searched quantization on CPU rather than the default GPU.")
    var cpu = false

    public init() {}

    public static func rescore(
        source: String,
        template: String,
        destination: String,
        expertBatch: Int = 16,
        groupSize: Int = 64,
        standardQ4: Bool = false,
        activationStats: String? = nil,
        validationStats: String? = nil,
        preflightOnly: Bool = false,
        cpu: Bool = false
    ) throws {
        var command = Self()
        command.source = source
        command.template = template
        command.destination = destination
        command.expertBatch = expertBatch
        command.groupSize = groupSize
        command.standardQ4 = standardQ4
        command.activationStats = activationStats
        command.validationStats = validationStats
        command.preflightOnly = preflightOnly
        command.cpu = cpu
        try command.validate()
        try command.run()
    }

    public mutating func validate() throws {
        guard [64, 128].contains(groupSize) else { throw ValidationError("--group-size must be 64 or 128.") }
        guard activationStats == nil || (groupSize == 64 && !standardQ4) else {
            throw ValidationError(
                "--activation-stats currently requires searched Q4 group-64; G128 and --standard-q4 are separate challengers."
            )
        }
        guard validationStats == nil || activationStats != nil else {
            throw ValidationError("--validation-stats requires --activation-stats.")
        }
        guard expertBatch >= 1, expertBatch <= 256 else {
            throw ValidationError("--expert-batch must be in 1...256.")
        }
    }

    public mutating func run() throws {
        #if os(Linux)
            defer { clearStreams() }
        #endif

        if cpu {
            try Device.withDefaultDevice(.cpu) { try execute() }
        } else {
            try execute()
        }
    }

    private func execute() throws {
        Memory.cacheLimit = 512 * 1_024 * 1_024
        Memory.peakMemory = 0

        let sourceURL = URL(fileURLWithPath: source).standardizedFileURL
        let templateURL = URL(fileURLWithPath: template).standardizedFileURL
        let destinationURL = URL(fileURLWithPath: destination).standardizedFileURL
        try validateDirectory(sourceURL, label: "source")
        try validateDirectory(templateURL, label: "template")
        guard sourceURL != templateURL, sourceURL != destinationURL, templateURL != destinationURL else {
            throw RescoreError.invalidInput("source, template, and destination must be distinct")
        }
        guard !FileManager.default.fileExists(atPath: destinationURL.path) else {
            throw RescoreError.invalidInput("destination already exists: \(destinationURL.path)")
        }

        let sourceIndexURL = sourceURL.appendingPathComponent("model.safetensors.index.json")
        let templateIndexURL = templateURL.appendingPathComponent("model.safetensors.index.json")
        let sourceIndex = try decodeIndex(at: sourceIndexURL)
        let templateIndex = try decodeIndex(at: templateIndexURL)
        let sourceConfiguration = try JSONDecoder().decode(
            SourceConfiguration.self,
            from: Data(contentsOf: sourceURL.appendingPathComponent("config.json"))
        )
        guard sourceConfiguration.numberOfExperts > 0 else {
            throw RescoreError.invalidInput("source num_experts must be positive")
        }

        let quantizedModules = templateIndex.weightMap.keys.compactMap { key -> String? in
            guard key.hasSuffix(".weight") else { return nil }
            let module = String(key.dropLast(".weight".count))
            guard templateIndex.weightMap["\(module).scales"] != nil,
                templateIndex.weightMap["\(module).biases"] != nil
            else { return nil }
            return module
        }.sorted()
        let q8Modules = quantizedModules.filter(Self.isQ8Router)
        let embeddingModules = quantizedModules.filter(Self.isStandardEmbedding)
        let q4Modules = quantizedModules.filter {
            !Self.isQ8Router($0) && !Self.isStandardEmbedding($0)
        }
        guard !q4Modules.isEmpty else {
            throw RescoreError.incompatibleTemplate("no affine Q4 modules were found")
        }
        let availableSourceKeys = Set(sourceIndex.weightMap.keys)
        var sourceUsePlan = LagunaSourceUsePlan(
            keysByModule: try Dictionary(
                uniqueKeysWithValues:
                    q4Modules.map { module in
                        (
                            module,
                            try Self.sourceKeys(
                                for: module, availableKeys: availableSourceKeys,
                                numberOfExperts: sourceConfiguration.numberOfExperts)
                        )
                    }))

        print(
            "Laguna Q4R8 LS2 rescore preflight: \(q4Modules.count) Q4 modules, "
                + "\(q8Modules.count) Q8 routers preserved, "
                + "\(embeddingModules.count) standard Q4 embedding(s) preserved"
        )
        print(
            "source tensors=\(sourceIndex.weightMap.count) template tensors=\(templateIndex.weightMap.count) "
                + "experts=\(sourceConfiguration.numberOfExperts) device=\(Device.defaultDevice())"
        )

        let groupPolicy =
            groupSize == 64
            ? nil
            : try LagunaGroupSizePolicy(
                templateConfiguration: Data(contentsOf: templateURL.appendingPathComponent("config.json")),
                groupSize: groupSize, q4Modules: q4Modules, q8Modules: q8Modules, embeddings: embeddingModules)
        print(
            "Output Q4 policy: group_size=\(groupSize) method=\(standardQ4 ? "standard" : "LS2"); routers and embedding retain G64."
        )
        let sourceConfigData = try Data(contentsOf: sourceURL.appendingPathComponent("config.json"))
        let sourceIndexData = try Data(contentsOf: sourceIndexURL)
        let calibration = try activationStats.map {
            try LagunaActivationStatistics.load(
                from: URL(fileURLWithPath: $0).standardizedFileURL,
                sourceURL: sourceURL, sourceConfig: sourceConfigData, sourceIndex: sourceIndexData)
        }
        let validation = try validationStats.map {
            try LagunaActivationStatistics.load(
                from: URL(fileURLWithPath: $0).standardizedFileURL,
                sourceURL: sourceURL, sourceConfig: sourceConfigData, sourceIndex: sourceIndexData,
                sourceWeightFingerprint: calibration?.metadata["source_weight_fingerprint"])
        }
        if let calibration, let validation { try validation.requireDisjoint(from: calibration) }
        let searchedTemplate = try calibration != nil && templateUsesScaleSearch(templateURL)
        if let calibration {
            try validateActivationCoverage(
                calibration, validation: validation,
                q4Modules: q4Modules, q8Modules: q8Modules, expertCount: sourceConfiguration.numberOfExperts)
        }
        var sourceArrays = try loadSourceArrays(sourceURL: sourceURL, index: sourceIndex)
        if groupSize != 64 {
            for key in sourceUsePlan.remainingUses.keys {
                guard let weight = sourceArrays[key], weight.ndim == 2,
                    weight.dim(-1) > 0, weight.dim(-1) % groupSize == 0
                else {
                    throw RescoreError.invalidInput(
                        "source input width is not compatible with group-\(groupSize): \(key)")
                }
            }
        }
        if let calibration {
            try validateActivationSourceGeometry(calibration, q4Modules: q4Modules, sourceArrays: sourceArrays)
        }
        try verifyTemplateIdentity(
            sourceArrays: sourceArrays,
            templateURL: templateURL,
            templateIndex: templateIndex,
            searchedTemplate: searchedTemplate
        )
        if calibration != nil {
            try verifyAllRouters(
                q8Modules, sourceArrays: sourceArrays,
                templateURL: templateURL, templateIndex: templateIndex)
        }
        let preflightPeakMemory = Memory.peakMemory
        print("Template identity checks passed for dense Q4, routed gate/up/down Q4, and router Q8.")
        print("Preflight peak MLX memory: \(preflightPeakMemory) bytes.")
        if preflightOnly { return }
        // loadArrays creates lazy Load nodes, not memory maps. Once evaluated, their
        // payloads stay resident while retained. Keep only actual conversion inputs,
        // and release each after its final evaluated output has consumed it.
        let originalSourceTensorCount = sourceArrays.count
        sourceArrays = sourceArrays.filter { sourceUsePlan.remainingUses[$0.key] != nil }
        var sourceTensorsReleased = originalSourceTensorCount - sourceArrays.count
        Memory.clearCache()

        try FileManager.default.createDirectory(
            at: destinationURL, withIntermediateDirectories: false)
        var outputComplete = false
        defer { if !outputComplete { try? FileManager.default.removeItem(at: destinationURL) } }
        try copySidecars(from: templateURL, to: destinationURL)
        if let groupPolicy {
            try groupPolicy.configuration.write(
                to: destinationURL.appendingPathComponent("config.json"), options: .atomic)
        }
        var activationDiagnostics = [String: ActivationWeightedScaleSearchDiagnostics]()
        var retainedTemplateExperts = [String: [Int]]()

        let shardNames = Set(templateIndex.weightMap.values).sorted()
        let shardOrder = Dictionary(
            uniqueKeysWithValues: shardNames.enumerated().map { ($0.element, $0.offset) }
        )
        var modulesByShard = [String: [String]]()
        for module in q4Modules {
            let moduleShards = try ["weight", "scales", "biases"].map { suffix in
                guard let shard = templateIndex.weightMap["\(module).\(suffix)"] else {
                    throw RescoreError.incompatibleTemplate("missing \(module).\(suffix)")
                }
                return shard
            }
            guard
                let firstShard = moduleShards.min(by: {
                    shardOrder[$0, default: .max] < shardOrder[$1, default: .max]
                })
            else {
                throw RescoreError.incompatibleTemplate("missing shard placement for \(module)")
            }
            modulesByShard[firstShard, default: []].append(module)
        }
        var pendingReplacements = [String: MLXArray]()
        var completedModules = 0
        var templateTensorBytes = 0
        var outputTensorBytes = 0
        for (shardOffset, shardName) in shardNames.enumerated() {
            let sourceShardURL = templateURL.appendingPathComponent(shardName)
            var arrays = try loadArrays(url: sourceShardURL, stream: .cpu)
            templateTensorBytes += arrays.values.reduce(0) { $0 + $1.nbytes }
            let modules = (modulesByShard[shardName] ?? []).sorted()
            print("[shard \(shardOffset + 1)/\(shardNames.count)] \(shardName): \(modules.count) Q4 module(s)")

            for key in pendingReplacements.keys.sorted() where arrays[key] != nil {
                let value = pendingReplacements.removeValue(forKey: key)!
                try install(value, key: key, arrays: &arrays)
            }

            for module in modules {
                let replacement: QuantizedArrays
                if let calibration {
                    let baseline = QuantizedArrays(
                        weight: try templateArray(
                            templateURL: templateURL, index: templateIndex, key: module + ".weight", expert: nil),
                        scales: try templateArray(
                            templateURL: templateURL, index: templateIndex, key: module + ".scales", expert: nil),
                        biases: try templateArray(
                            templateURL: templateURL, index: templateIndex, key: module + ".biases", expert: nil))
                    replacement = try activationRefined(
                        module: module, baseline: baseline, sourceArrays: sourceArrays,
                        numberOfExperts: sourceConfiguration.numberOfExperts, calibration: calibration,
                        validation: validation, diagnostics: &activationDiagnostics,
                        retained: &retainedTemplateExperts)
                } else if let routed = Self.routedProjection(module) {
                    replacement = try quantizeRoutedProjection(
                        routed,
                        sourceArrays: sourceArrays,
                        numberOfExperts: sourceConfiguration.numberOfExperts
                    )
                } else {
                    replacement = searched(try directSourceWeight(module, sourceArrays: sourceArrays))
                }

                // Detach output graphs from source weights before releasing the last
                // source owner; concatenated expert batches may still be lazy here.
                try MLX.checkedEval(replacement.weight, replacement.scales, replacement.biases)
                for key in try sourceUsePlan.finish(module: module) {
                    guard sourceArrays.removeValue(forKey: key) != nil else {
                        throw RescoreError.invalidInput("source tensor released before its final consumer: \(key)")
                    }
                    sourceTensorsReleased += 1
                }
                for (key, value) in replacementEntries(replacement, module: module) {
                    if arrays[key] != nil {
                        try install(value, key: key, arrays: &arrays)
                    } else {
                        guard templateIndex.weightMap[key] != nil,
                            pendingReplacements.updateValue(value, forKey: key) == nil
                        else {
                            throw RescoreError.incompatibleTemplate(
                                "invalid or duplicate deferred replacement \(key)"
                            )
                        }
                    }
                }
                completedModules += 1
                print(
                    "  [\(completedModules)/\(q4Modules.count)] \(module) "
                        + "source_tensors_remaining=\(sourceArrays.count) mlx_active_bytes=\(Memory.activeMemory)")
                Memory.clearCache()
            }

            Stream.defaultStream(Device.defaultDevice()).synchronize()
            let temporaryURL = destinationURL.appendingPathComponent(".\(shardName).partial.safetensors")
            outputTensorBytes += arrays.values.reduce(0) { $0 + $1.nbytes }
            try save(arrays: arrays, metadata: ["format": "mlx"], url: temporaryURL)
            let outputURL = destinationURL.appendingPathComponent(shardName)
            try FileManager.default.moveItem(at: temporaryURL, to: outputURL)
            arrays.removeAll(keepingCapacity: false)
            Memory.clearCache()
        }
        guard sourceUsePlan.remainingUses.isEmpty, sourceArrays.isEmpty else {
            throw RescoreError.invalidInput("conversion retained source tensors after their final consumers")
        }
        guard pendingReplacements.isEmpty, completedModules == q4Modules.count else {
            throw RescoreError.incompatibleTemplate(
                "stream ended with \(pendingReplacements.count) deferred tensor(s) and "
                    + "\(completedModules)/\(q4Modules.count) Q4 modules"
            )
        }

        let outputIndexURL = destinationURL.appendingPathComponent("model.safetensors.index.json")
        if groupPolicy != nil {
            try LagunaGroupSizePolicy.updatedIndex(Data(contentsOf: templateIndexURL), tensorBytes: outputTensorBytes)
                .write(to: outputIndexURL, options: .atomic)
        } else {
            try FileManager.default.copyItem(at: templateIndexURL, to: outputIndexURL)
        }
        var provenance = RescoreProvenance(
            createdAt: ISO8601DateFormatter().string(from: Date()),
            sourceModel: sourceURL.path,
            templateModel: templateURL.path,
            q4ModulesRescored: q4Modules.count,
            q8ModulesPreserved: q8Modules.count,
            standardQ4EmbeddingsPreserved: embeddingModules.count,
            expertBatchSize: expertBatch
        )
        provenance.groupSize = groupSize
        provenance.templateTensorBytes = templateTensorBytes
        provenance.outputTensorBytes = outputTensorBytes
        provenance.moduleQuantizationOverrides = groupPolicy?.overrides
        if standardQ4 {
            provenance.algorithm = "q4r8_affine_standard"
            provenance.searchFactors = []
            provenance.biasRefinementIterations = 0
            provenance.jointAffineRefinementIterations = 0
        }
        provenance.peakMLXMemoryBytes = Memory.peakMemory
        provenance.preflightPeakMLXMemoryBytes = preflightPeakMemory
        provenance.sourceTensorsReleased = sourceTensorsReleased
        provenance.sourceTensorsRemaining = sourceArrays.count
        provenance.quantizationDevice = String(describing: Device.defaultDevice())
        if let calibration {
            provenance.algorithm = "laguna_q4r8_activation_weighted_scale_search"
            provenance.activationStatistics = calibration.url.path
            provenance.sourceWeightFingerprint = calibration.metadata["source_weight_fingerprint"]
            provenance.sourceWeightFingerprintMethod = IndexedSafetensorsFingerprint.method
            provenance.validationStatistics = validation?.url.path
            provenance.calibrationCorpusFingerprint = calibration.metadata["corpus_fingerprint"]
            provenance.validationCorpusFingerprint = validation?.metadata["corpus_fingerprint"]
            provenance.minimumExpertPositions = max(
                calibration.minimumExpertPositions, validation?.minimumExpertPositions ?? 0)
            provenance.retainedTemplateExperts = retainedTemplateExperts
            provenance.activationDiagnostics = activationDiagnostics
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(provenance).write(
            to: destinationURL.appendingPathComponent("q4r8-scale-search.json"),
            options: .atomic
        )
        outputComplete = true
        print(
            "Created \(destinationURL.path) from \(shardNames.count) streamed shard(s). "
                + "Peak MLX memory: \(provenance.peakMLXMemoryBytes) bytes; "
                + "released \(sourceTensorsReleased) source tensors.")
    }

    private func loadSourceArrays(
        sourceURL: URL,
        index: SafetensorsIndex
    ) throws -> [String: MLXArray] {
        let shardNames = Set(index.weightMap.values).sorted()
        var result = [String: MLXArray]()
        result.reserveCapacity(index.weightMap.count)
        for (offset, shardName) in shardNames.enumerated() {
            let shardURL = sourceURL.appendingPathComponent(shardName)
            let arrays = try loadArrays(url: shardURL, stream: .cpu)
            for (key, value) in arrays {
                guard result.updateValue(value, forKey: key) == nil else {
                    throw RescoreError.invalidInput("duplicate source tensor \(key)")
                }
            }
            print("[source \(offset + 1)/\(shardNames.count)] indexed \(shardName)")
        }
        guard result.count == index.weightMap.count else {
            throw RescoreError.invalidInput(
                "source index names \(index.weightMap.count) tensors but shards exposed \(result.count)")
        }
        return result
    }

    private func searched(_ source: MLXArray) -> QuantizedArrays {
        let result: QuantizedArrays
        if standardQ4 {
            let arrays = MLX.quantized(source, groupSize: groupSize, bits: 4, mode: .affine)
            result = .init(weight: arrays.wq, scales: arrays.scales, biases: arrays.biases!)
        } else {
            let arrays = q4AffineScaleSearchQuantized(source, groupSize: groupSize)
            result = .init(weight: arrays.weight, scales: arrays.scales, biases: arrays.biases)
        }
        MLX.eval(result.weight, result.scales, result.biases)
        Stream.defaultStream(Device.defaultDevice()).synchronize()
        return .init(weight: result.weight, scales: result.scales, biases: result.biases)
    }

    private func quantizeRoutedProjection(
        _ routed: (layer: Int, projection: String),
        sourceArrays: [String: MLXArray],
        numberOfExperts: Int
    ) throws -> QuantizedArrays {
        let prefix = "model.layers.\(routed.layer).mlp.experts"
        var weightParts = [MLXArray]()
        var scaleParts = [MLXArray]()
        var biasParts = [MLXArray]()
        weightParts.reserveCapacity((numberOfExperts + expertBatch - 1) / expertBatch)
        scaleParts.reserveCapacity(weightParts.capacity)
        biasParts.reserveCapacity(weightParts.capacity)

        for start in stride(from: 0, to: numberOfExperts, by: expertBatch) {
            let end = min(start + expertBatch, numberOfExperts)
            var batch = [MLXArray]()
            batch.reserveCapacity(end - start)
            for expert in start..<end {
                if routed.projection == "gate_up_proj" {
                    let gateKey = "\(prefix).\(expert).gate_proj.weight"
                    let upKey = "\(prefix).\(expert).up_proj.weight"
                    guard let gate = sourceArrays[gateKey] else { throw RescoreError.missingTensor(gateKey) }
                    guard let up = sourceArrays[upKey] else { throw RescoreError.missingTensor(upKey) }
                    batch.append(concatenated([gate, up], axis: -2))
                } else {
                    let key = "\(prefix).\(expert).\(routed.projection).weight"
                    guard let source = sourceArrays[key] else { throw RescoreError.missingTensor(key) }
                    batch.append(source)
                }
            }

            let sourceBatch = MLX.stacked(batch)
            MLX.eval(sourceBatch)
            let result = searched(sourceBatch)
            weightParts.append(result.weight)
            scaleParts.append(result.scales)
            biasParts.append(result.biases)
            print(
                "    layer \(routed.layer) \(routed.projection): experts \(start)...\(end - 1)"
            )
            Memory.clearCache()
        }

        let result = QuantizedArrays(
            weight: concatenated(weightParts, axis: 0),
            scales: concatenated(scaleParts, axis: 0),
            biases: concatenated(biasParts, axis: 0)
        )
        MLX.eval(result.weight, result.scales, result.biases)
        Stream.defaultStream(Device.defaultDevice()).synchronize()
        return result
    }

    private func replacementEntries(
        _ replacement: QuantizedArrays,
        module: String
    ) -> [(String, MLXArray)] {
        [
            ("\(module).weight", replacement.weight),
            ("\(module).scales", replacement.scales),
            ("\(module).biases", replacement.biases),
        ]
    }

    private func install(
        _ value: MLXArray,
        key: String,
        arrays: inout [String: MLXArray]
    ) throws {
        guard let templateValue = arrays[key] else {
            throw RescoreError.incompatibleTemplate("\(key) is not in its declared shard")
        }
        var expectedShape = templateValue.shape
        if groupSize != 64, key.hasSuffix(".scales") || key.hasSuffix(".biases") {
            guard let last = expectedShape.last, last % (groupSize / 64) == 0 else {
                throw RescoreError.incompatibleTemplate("template metadata cannot be regrouped: \(key)")
            }
            expectedShape[expectedShape.count - 1] = last / (groupSize / 64)
        }
        let expectedBytes = expectedShape.reduce(templateValue.dtype.size, *)
        guard value.shape == expectedShape,
            value.dtype == templateValue.dtype,
            value.nbytes == expectedBytes
        else {
            throw RescoreError.incompatibleTemplate(
                "\(key) expected shape=\(expectedShape) dtype=\(templateValue.dtype) bytes=\(expectedBytes), "
                    + "got shape=\(value.shape) dtype=\(value.dtype) bytes=\(value.nbytes)"
            )
        }
        arrays[key] = value
    }

    private func verifyTemplateIdentity(
        sourceArrays: [String: MLXArray],
        templateURL: URL,
        templateIndex: SafetensorsIndex,
        searchedTemplate: Bool = false
    ) throws {
        // Affine CPU/GPU conversion can differ in exact integer assignments and
        // stored scales. Verify using the same device selected for this conversion,
        // just as searched() does, and retain strict byte-for-byte identity.
        var checks: [(module: String, bits: Int, source: String, expert: Int?)] = [
            (
                "language_model.model.layers.1.mlp.shared_expert.down_proj",
                4,
                "model.layers.1.mlp.shared_expert.down_proj.weight",
                nil
            ),
            (
                "language_model.model.layers.1.mlp.gate.proj",
                8,
                "model.layers.1.mlp.gate.weight",
                nil
            ),
            (
                "language_model.model.layers.1.mlp.switch_mlp.down_proj",
                4,
                "model.layers.1.mlp.experts.0.down_proj.weight",
                0
            ),
        ]

        let fusedModule = "language_model.model.layers.1.mlp.switch_mlp.gate_up_proj"
        let hasFusedGateUp = templateIndex.weightMap[fusedModule + ".weight"] != nil
        if !hasFusedGateUp {
            for projection in ["gate_proj", "up_proj"] {
                checks.append(
                    (
                        "language_model.model.layers.1.mlp.switch_mlp." + projection, 4,
                        "model.layers.1.mlp.experts.0." + projection + ".weight", 0
                    ))
            }
        }

        for check in checks {
            let weightKey = "\(check.module).weight"
            guard templateIndex.weightMap[weightKey] != nil else {
                throw RescoreError.incompatibleTemplate("missing identity check key \(weightKey)")
            }
            guard let source = sourceArrays[check.source] else {
                throw RescoreError.missingTensor(check.source)
            }
            let standard: QuantizedArrays
            if searchedTemplate && check.bits == 4 {
                standard = searched(source)
            } else {
                let ordinary = MLX.quantized(source, groupSize: 64, bits: check.bits, mode: .affine)
                guard let biases = ordinary.biases else {
                    throw RescoreError.incompatibleTemplate("standard affine quantization omitted biases")
                }
                standard = .init(weight: ordinary.wq, scales: ordinary.scales, biases: biases)
            }
            let expectedWeight = try templateArray(
                templateURL: templateURL,
                index: templateIndex,
                key: weightKey,
                expert: check.expert
            )
            let expectedScales = try templateArray(
                templateURL: templateURL,
                index: templateIndex,
                key: "\(check.module).scales",
                expert: check.expert
            )
            let expectedBiases = try templateArray(
                templateURL: templateURL,
                index: templateIndex,
                key: "\(check.module).biases",
                expert: check.expert
            )
            guard arraysEqual(standard.weight, expectedWeight),
                arraysEqual(standard.scales, expectedScales),
                arraysEqual(standard.biases, expectedBiases)
            else {
                throw RescoreError.incompatibleTemplate(
                    "\(check.module) does not match standard \(check.bits)-bit quantization of the source"
                )
            }
            Memory.clearCache()
        }

        guard hasFusedGateUp else { return }
        let gateKey = "model.layers.1.mlp.experts.0.gate_proj.weight"
        let upKey = "model.layers.1.mlp.experts.0.up_proj.weight"
        guard let gate = sourceArrays[gateKey] else { throw RescoreError.missingTensor(gateKey) }
        guard let up = sourceArrays[upKey] else { throw RescoreError.missingTensor(upKey) }
        let standardGate = MLX.quantized(
            gate, groupSize: 64, bits: 4, mode: .affine)
        let standardUp = MLX.quantized(
            up, groupSize: 64, bits: 4, mode: .affine)
        guard let gateBiases = standardGate.biases, let upBiases = standardUp.biases else {
            throw RescoreError.incompatibleTemplate("standard fused quantization omitted biases")
        }
        var standardFused = QuantizedArrays(
            weight: concatenated([standardGate.wq, standardUp.wq], axis: -2),
            scales: concatenated([standardGate.scales, standardUp.scales], axis: -2),
            biases: concatenated([gateBiases, upBiases], axis: -2)
        )
        if searchedTemplate { standardFused = searched(concatenated([gate, up], axis: -2)) }
        guard
            arraysEqual(
                standardFused.weight,
                try templateArray(
                    templateURL: templateURL,
                    index: templateIndex,
                    key: "\(fusedModule).weight",
                    expert: 0
                )
            ),
            arraysEqual(
                standardFused.scales,
                try templateArray(
                    templateURL: templateURL,
                    index: templateIndex,
                    key: "\(fusedModule).scales",
                    expert: 0
                )
            ),
            arraysEqual(
                standardFused.biases,
                try templateArray(
                    templateURL: templateURL,
                    index: templateIndex,
                    key: "\(fusedModule).biases",
                    expert: 0
                )
            )
        else {
            throw RescoreError.incompatibleTemplate(
                "fused routed gate/up layout does not match source expert order"
            )
        }
        Memory.clearCache()
    }

    private func templateArray(
        templateURL: URL,
        index: SafetensorsIndex,
        key: String,
        expert: Int?
    ) throws -> MLXArray {
        guard let shard = index.weightMap[key] else {
            throw RescoreError.incompatibleTemplate("missing index entry for \(key)")
        }
        let arrays = try loadArrays(
            url: templateURL.appendingPathComponent(shard), stream: .cpu)
        guard let value = arrays[key] else {
            throw RescoreError.incompatibleTemplate("missing \(key) in declared shard")
        }
        if let expert { return value[expert] }
        return value
    }

    private func arraysEqual(_ lhs: MLXArray, _ rhs: MLXArray) -> Bool {
        let equal = MLX.arrayEqual(lhs, rhs)
        MLX.eval(equal)
        Stream.defaultStream(Device.defaultDevice()).synchronize()
        return equal.item(Bool.self)
    }

    private func copySidecars(from templateURL: URL, to destinationURL: URL) throws {
        for item in try FileManager.default.contentsOfDirectory(
            at: templateURL,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) {
            if item.pathExtension == "safetensors"
                || item.lastPathComponent == "model.safetensors.index.json"
                || item.lastPathComponent == "q4r8-scale-search.json"
            {
                continue
            }
            try FileManager.default.copyItem(
                at: item,
                to: destinationURL.appendingPathComponent(item.lastPathComponent)
            )
        }
    }

    static func sourceKeys(
        for module: String, availableKeys: Set<String>, numberOfExperts: Int
    ) throws -> [String] {
        if let routed = Self.routedProjection(module) {
            let prefix = "model.layers.\(routed.layer).mlp.experts"
            let projections =
                routed.projection == "gate_up_proj"
                ? ["gate_proj", "up_proj"] : [routed.projection]
            let keys = (0..<numberOfExperts).flatMap { expert in
                projections.map { "\(prefix).\(expert).\($0).weight" }
            }
            for key in keys where !availableKeys.contains(key) { throw RescoreError.missingTensor(key) }
            return keys
        }
        let key = try Self.directSourceWeightKey(for: module)
        if availableKeys.contains(key) { return [key] }
        if key.hasSuffix("gate_up_proj.weight") {
            let prefix = String(key.dropLast("gate_up_proj.weight".count))
            let keys = [prefix + "gate_proj.weight", prefix + "up_proj.weight"]
            if keys.allSatisfy(availableKeys.contains) { return keys }
        }
        throw RescoreError.missingTensor(key)
    }

    private func decodeIndex(at url: URL) throws -> SafetensorsIndex {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw RescoreError.invalidInput("missing safetensors index: \(url.path)")
        }
        return try JSONDecoder().decode(SafetensorsIndex.self, from: Data(contentsOf: url))
    }

    private func validateDirectory(_ url: URL, label: String) throws {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw RescoreError.invalidInput("\(label) directory does not exist: \(url.path)")
        }
    }

    private static func isQ8Router(_ module: String) -> Bool {
        module.hasSuffix(".mlp.gate.proj")
    }

    private static func isStandardEmbedding(_ module: String) -> Bool {
        module == "language_model.model.embed_tokens"
    }

    private static func directSourceWeightKey(for module: String) throws -> String {
        let prefix = "language_model."
        guard module.hasPrefix(prefix), routedProjection(module) == nil else {
            throw RescoreError.incompatibleTemplate("cannot map direct Q4 module \(module)")
        }
        return String(module.dropFirst(prefix.count)) + ".weight"
    }

    private static func routedProjection(
        _ module: String
    ) -> (layer: Int, projection: String)? {
        let parts = module.split(separator: ".")
        guard parts.count == 7,
            parts[0] == "language_model",
            parts[1] == "model",
            parts[2] == "layers",
            let layer = Int(parts[3]),
            parts[4] == "mlp",
            parts[5] == "switch_mlp",
            ["down_proj", "gate_up_proj", "gate_proj", "up_proj"].contains(String(parts[6]))
        else { return nil }
        return (layer, String(parts[6]))
    }
}

private extension LagunaScaleSearchRescorer {
    func templateUsesScaleSearch(_ templateURL: URL) throws -> Bool {
        for name in ["q4r8-scale-search.json", "scale-search-quantization.json"] {
            let url = templateURL.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else { continue }
            guard let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any],
                json["algorithm"] as? String == "q4r8_affine_scale_search_ls2"
            else {
                throw RescoreError.incompatibleTemplate(
                    "activation refinement requires a standard Q4R8 or verified LS2 template")
            }
            return true
        }
        return false
    }

    func validateActivationCoverage(
        _ calibration: LagunaActivationStatistics, validation: LagunaActivationStatistics?,
        q4Modules: [String], q8Modules: [String], expertCount: Int
    ) throws {
        for stats in [calibration, validation].compactMap({ $0 }) {
            var allowed = Set<String>()
            for module in q4Modules + q8Modules {
                let path = try stats.momentPath(for: module)
                allowed.insert(path)
                let moments = stats.moments[path]!
                if Self.routedProjection(module) != nil {
                    guard moments.ndim == 2, moments.dim(0) == expertCount,
                        stats.counts[path]?.count == expertCount
                    else {
                        throw LagunaActivationInputError(message: "expected expert-conditional matrix for \(module)")
                    }
                } else if moments.ndim != 1 {
                    throw LagunaActivationInputError(message: "expected dense input vector for \(module)")
                }
            }
            guard allowed == Set(stats.moments.keys) else {
                throw LagunaActivationInputError(message: "statistics include unexpected projections")
            }
            for module in q4Modules {
                guard let routed = Self.routedProjection(module), routed.projection != "down_proj" else { continue }
                let gatePath = try stats.momentPath(for: module)
                let down = "language_model.model.layers.\(routed.layer).mlp.switch_mlp.down_proj"
                let downPath = try stats.momentPath(for: down)
                guard stats.counts[gatePath] == stats.counts[downPath] else {
                    throw LagunaActivationInputError(
                        message: "gate/up and down selected-position counts disagree for \(module)")
                }
            }
        }
        if let validation {
            for module in q4Modules {
                let lhs = try calibration.momentPath(for: module)
                let rhs = try validation.momentPath(for: module)
                guard calibration.moments[lhs]?.shape == validation.moments[rhs]?.shape else {
                    throw LagunaActivationInputError(message: "calibration and dev geometry differ for \(module)")
                }
            }
        }
    }

    func verifyAllRouters(
        _ modules: [String], sourceArrays: [String: MLXArray],
        templateURL: URL, templateIndex: SafetensorsIndex
    ) throws {
        for module in modules {
            let sourceKey =
                String(module.dropFirst("language_model.".count))
                .replacingOccurrences(of: ".gate.proj", with: ".gate") + ".weight"
            guard let source = sourceArrays[sourceKey] else { throw RescoreError.missingTensor(sourceKey) }
            let result = MLX.quantized(source, groupSize: 64, bits: 8, mode: .affine)
            guard let biases = result.biases else {
                throw RescoreError.incompatibleTemplate("missing Q8 affine biases")
            }
            for (suffix, array) in [("weight", result.wq), ("scales", result.scales), ("biases", biases)] {
                guard
                    arraysEqual(
                        array,
                        try templateArray(
                            templateURL: templateURL, index: templateIndex,
                            key: module + "." + suffix, expert: nil))
                else {
                    throw RescoreError.incompatibleTemplate(
                        "router \(module) is not the source's standard Q8 quantization")
                }
            }
        }
    }

    func validateActivationSourceGeometry(
        _ stats: LagunaActivationStatistics, q4Modules: [String], sourceArrays: [String: MLXArray]
    ) throws {
        for module in q4Modules {
            let path = try stats.momentPath(for: module)
            let width = stats.moments[path]!.dim(-1)
            let source: MLXArray
            if let routed = Self.routedProjection(module) {
                let projection = routed.projection == "gate_up_proj" ? "gate_proj" : routed.projection
                let key = "model.layers.\(routed.layer).mlp.experts.0.\(projection).weight"
                guard let array = sourceArrays[key] else { throw RescoreError.missingTensor(key) }
                source = array
            } else {
                source = try directSourceWeight(module, sourceArrays: sourceArrays)
            }
            guard source.ndim == 2, source.dim(-1) == width, width % 64 == 0,
                source.dtype == .bfloat16
            else { throw LagunaActivationInputError(message: "source BF16 shape/dtype does not match \(module)") }
        }
    }

    func directSourceWeight(_ module: String, sourceArrays: [String: MLXArray]) throws -> MLXArray {
        let key = try Self.directSourceWeightKey(for: module)
        if let value = sourceArrays[key] { return value }
        if key.hasSuffix(".gate_up_proj.weight") {
            let prefix = String(key.dropLast("gate_up_proj.weight".count))
            if let gate = sourceArrays[prefix + "gate_proj.weight"],
                let up = sourceArrays[prefix + "up_proj.weight"], gate.shape == up.shape, gate.dtype == up.dtype
            {
                return concatenated([gate, up], axis: -2)
            }
        }
        throw RescoreError.missingTensor(key)
    }

    func activationRefined(
        module: String, baseline: QuantizedArrays, sourceArrays: [String: MLXArray],
        numberOfExperts: Int, calibration: LagunaActivationStatistics,
        validation: LagunaActivationStatistics?,
        diagnostics: inout [String: ActivationWeightedScaleSearchDiagnostics],
        retained: inout [String: [Int]]
    ) throws -> QuantizedArrays {
        let calibrationPath = try calibration.momentPath(for: module)
        let validationPath = try validation?.momentPath(for: module)
        let calibrationMoments = calibration.moments[calibrationPath]!
        let validationMoments = validationPath.flatMap { validation?.moments[$0] }
        guard let routed = Self.routedProjection(module) else {
            let result = try MistralActivationWeightedScaleSearch.rescore(
                sourceWeight: directSourceWeight(module, sourceArrays: sourceArrays),
                templateWeight: baseline.weight, templateScales: baseline.scales, templateBiases: baseline.biases,
                calibrationSecondMoments: calibrationMoments, validationSecondMoments: validationMoments)
            diagnostics[module] = result.diagnostics
            return .init(weight: result.weight, scales: result.scales, biases: result.biases)
        }
        guard baseline.weight.ndim == 3, baseline.weight.dim(0) == numberOfExperts,
            baseline.scales.ndim == 3, baseline.scales.dim(0) == numberOfExperts,
            baseline.biases.shape == baseline.scales.shape,
            let calibrationCounts = calibration.counts[calibrationPath]
        else { throw RescoreError.incompatibleTemplate("invalid stacked expert template for \(module)") }
        let validationCounts = validationPath.flatMap { validation?.counts[$0] }
        var weights = [MLXArray]()
        var scales = [MLXArray]()
        var biases = [MLXArray]()
        var retainedExperts = [Int]()
        for start in stride(from: 0, to: numberOfExperts, by: expertBatch) {
            let end = min(start + expertBatch, numberOfExperts)
            var sourceBatch = [MLXArray]()
            for expert in start..<end {
                let prefix = "model.layers.\(routed.layer).mlp.experts.\(expert)."
                if routed.projection == "gate_up_proj" {
                    guard let gate = sourceArrays[prefix + "gate_proj.weight"],
                        let up = sourceArrays[prefix + "up_proj.weight"]
                    else { throw RescoreError.missingTensor(prefix + "gate_proj/up_proj.weight") }
                    guard gate.dtype == .bfloat16, up.dtype == .bfloat16, gate.shape == up.shape else {
                        throw RescoreError.invalidInput("invalid BF16 fused gate/up source for \(prefix)")
                    }
                    sourceBatch.append(concatenated([gate, up], axis: -2))
                } else {
                    let key = prefix + routed.projection + ".weight"
                    guard let source = sourceArrays[key], source.dtype == .bfloat16 else {
                        throw RescoreError.missingTensor(key + " (BF16)")
                    }
                    sourceBatch.append(source)
                }
            }
            let result = try MistralActivationWeightedScaleSearch.rescoreExperts(
                sourceWeight: MLX.stacked(sourceBatch),
                templateWeight: baseline.weight[start..<end],
                templateScales: baseline.scales[start..<end], templateBiases: baseline.biases[start..<end],
                calibrationSecondMoments: calibrationMoments[start..<end],
                calibrationPositionCounts: Array(calibrationCounts[start..<end]),
                validationSecondMoments: validationMoments.map { $0[start..<end] },
                validationPositionCounts: validationCounts.map { Array($0[start..<end]) },
                minimumExpertPositions: max(calibration.minimumExpertPositions, validation?.minimumExpertPositions ?? 0)
            )
            weights.append(result.weight)
            scales.append(result.scales)
            biases.append(result.biases)
            for (expert, diagnostic) in result.diagnostics {
                diagnostics[module + ".expert.\(start + expert)"] = diagnostic
            }
            retainedExperts.append(contentsOf: result.retainedTemplateExperts.map { start + $0 })
            MLX.eval(result.weight, result.scales, result.biases)
            Memory.clearCache()
        }
        retained[module] = retainedExperts
        return .init(
            weight: concatenated(weights, axis: 0), scales: concatenated(scales, axis: 0),
            biases: concatenated(biases, axis: 0))
    }
}
