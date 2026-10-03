import ArgumentParser
import Foundation
import LagunaScaleSearchCore
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import QuantizerSupport
import WickModelSupport

private enum WickMetadata {
    static let version = "0.2.0"

    static var commandName: String {
        let executableName = URL(fileURLWithPath: CommandLine.arguments[0]).lastPathComponent
        let supportedNames = ["wick", "facet", "model-runner-quantize"]
        return supportedNames.contains(executableName) ? executableName : "wick"
    }
}

private struct IgnoredConfigurationValue: Decodable {}

private struct SourceDescriptor: Decodable {
    let modelType: String
    let architectures: [String]
    let quantization: IgnoredConfigurationValue?
    let quantizationConfig: IgnoredConfigurationValue?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case quantization
        case quantizationConfig = "quantization_config"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        architectures = try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        quantization = try container.decodeIfPresent(
            IgnoredConfigurationValue.self, forKey: .quantization)
        quantizationConfig = try container.decodeIfPresent(
            IgnoredConfigurationValue.self, forKey: .quantizationConfig)
    }

    var isQuantized: Bool {
        quantization != nil || quantizationConfig != nil
    }

    var isLagunaDFlash: Bool {
        architectures.contains("DFlashLagunaForCausalLM")
    }
}

private struct QuantizerProvenance: Encodable {
    var boundedMemory: Bool
    var format = 1
    var tool = "midnight-afterglow"
    var toolVersion = WickMetadata.version
    var status: String
    var algorithm: String
    var createdAt: String
    var sourceModel: String
    var modelType: String
    var architectureProfile: String
    var bits = 4
    var groupSize = 64
    var mode: String
    var calibration: String
    var standardModules: [String]
    var gemmaGroupPolicy = false
    var g128Modules: [String] = []
    var omittedTiedHead = false
    var tiedHeadQuantizationAliases: [String: String]? = nil
    var q8Bits = 8
    var q4ScaleSearchModules: [String]
    var q5ScaleSearchModules: [String] = []
    var standardQ4Modules: [String]
    var q8Modules: [String]
    var mandatoryQ8Modules: [String]
    var skippedModules: [String]
    var outputShards: [String]

    enum CodingKeys: String, CodingKey {
        case format, tool, status, algorithm, bits, mode, calibration
        case standardModules = "standard_modules"
        case boundedMemory = "bounded_memory"
        case toolVersion = "tool_version"
        case createdAt = "created_at"
        case sourceModel = "source_model"
        case modelType = "model_type"
        case architectureProfile = "architecture_profile"
        case groupSize = "group_size"
        case gemmaGroupPolicy = "gemma_group_policy"
        case g128Modules = "g128_modules"
        case omittedTiedHead = "omitted_tied_head"
        case tiedHeadQuantizationAliases = "tied_head_quantization_aliases"
        case q8Bits = "q8_bits"
        case q4ScaleSearchModules = "q4_scale_search_modules"
        case q5ScaleSearchModules = "q5_scale_search_modules"
        case standardQ4Modules = "standard_q4_modules"
        case q8Modules = "q8_modules"
        case mandatoryQ8Modules = "mandatory_q8_modules"
        case skippedModules = "skipped_modules"
        case outputShards = "output_shards"
    }
}

private struct ArchitectureProfile {
    let name: String
    let mandatoryQ8Suffixes: [String]
    let requiresMandatoryQ8Match: Bool

    static let lagunaDFlash = Self(
        name: "laguna-dflash-q4r8",
        // DFlash's 10,240 -> 2,048 context projection is the sole bottleneck for
        // all target features. Its per-head attention gates are tiny and directly
        // modulate every draft layer. Protecting both costs little relative to the
        // 0.5B drafter while the larger Q/K/V/O and MLP matrices receive searched Q4.
        mandatoryQ8Suffixes: ["fc", ".self_attn.g_proj"],
        requiresMandatoryQ8Match: true
    )

    static func resolve(modelType: String) -> Self {
        switch modelType {
        case "laguna":
            return .init(
                name: "laguna-q4r8",
                mandatoryQ8Suffixes: [".mlp.gate.proj"],
                requiresMandatoryQ8Match: true
            )
        case "mixtral":
            return .init(
                name: "mixtral-moe-q4r8",
                mandatoryQ8Suffixes: [".block_sparse_moe.gate"],
                requiresMandatoryQ8Match: true
            )
        case "talkie":
            // Talkie is dense. Its learned scalar/head gains stay in source precision;
            // there is no router or measured basis for a mandatory Q8 matrix policy.
            return .init(
                name: "talkie-dense-affine",
                mandatoryQ8Suffixes: [],
                requiresMandatoryQ8Match: false
            )
        case "gpt_oss":
            return .init(
                name: "gpt-oss-moe-q4r8",
                mandatoryQ8Suffixes: [".mlp.router"],
                requiresMandatoryQ8Match: true
            )
        case "qwen3_moe", "qwen3_5_moe":
            return .init(
                name: "qwen-moe-q4r8",
                mandatoryQ8Suffixes: [".mlp.gate"],
                requiresMandatoryQ8Match: true
            )
        case "qwen3_next", "qwen3_5", "qwen3_5_text":
            // These model types can describe dense or routed variants. A discovered
            // MoE gate is protected, while a genuinely dense configuration remains Q4.
            return .init(
                name: "qwen-auto-q4r8",
                mandatoryQ8Suffixes: [".mlp.gate"],
                requiresMandatoryQ8Match: false
            )
        case "phimoe":
            return .init(
                name: "phi-moe-q4r8",
                mandatoryQ8Suffixes: [".block_sparse_moe.gate"],
                requiresMandatoryQ8Match: true
            )
        case "minimax":
            return .init(
                name: "minimax-moe-q4r8",
                mandatoryQ8Suffixes: [".block_sparse_moe.gate"],
                requiresMandatoryQ8Match: true
            )
        case "jamba":
            return .init(
                name: "jamba-moe-q4r8",
                mandatoryQ8Suffixes: [".block_sparse_moe.router"],
                requiresMandatoryQ8Match: true
            )
        default:
            return .init(
                name: "generic-mlx-q4",
                mandatoryQ8Suffixes: [],
                requiresMandatoryQ8Match: false
            )
        }
    }
}

private struct ConversionPlan {
    let modelType: String
    let profile: ArchitectureProfile
    let quantizablePaths: Set<String>
    let scaleSearchPaths: Set<String>
    let standardQ4Paths: Set<String>
    let standardPaths: Set<String>
    let q8Paths: Set<String>
    let mandatoryQ8Paths: Set<String>
    let skippedPaths: Set<String>
    var gemmaGroupPolicy: GemmaGroupSizePolicy? = nil
}

private enum QuantizerError: Error, LocalizedError {
    case invalidInput(String)
    case unsupportedModelType(String)
    case unsupportedDrafterArchitecture(String)
    case invalidPolicy(String)

    var errorDescription: String? {
        switch self {
        case .invalidInput(let message):
            "Invalid quantization input: \(message)"
        case .unsupportedModelType(let modelType):
            "Unsupported MLX Swift model_type '\(modelType)'. Add it to LLMTypeRegistry before quantizing it."
        case .unsupportedDrafterArchitecture(let architecture):
            "Unsupported MLX Swift drafter architecture '\(architecture)'. Add it to MTPDrafterTypeRegistry before quantizing it."
        case .invalidPolicy(let message):
            "Invalid quantization policy: \(message)"
        }
    }
}

public struct ModelQuantizer: AsyncParsableCommand {
    public init() {}
    public static let configuration = CommandConfiguration(
        commandName: "quantize",
        abstract:
            "Convert an unquantized checkpoint to affine, MXFP4, MXFP8, or NVFP4 with architecture-aware Q8 policies.",
        discussion: """
            The source must be an unquantized safetensors checkpoint whose model_type is registered
            by MLX Swift LM. Mixtral, Laguna, GPT-OSS, and supported Qwen MoE models automatically
            keep their routing projections in standard affine Q8. Dense models such as Mistral and
            Llama use searched affine Q4 for eligible matrices by default.
            --mode or --bits selects standard MLX quantization. --calibration scale-search
            explicitly selects searched affine Q4/G64 or Q4/G128. See `wick formats` for geometry.

            Talkie uses Wick's native model adapter and folds the learned output-head
            gain into its BF16 weight before quantization. Source projection names are
            preserved on disk. --standard-q8 selects a standard affine Q8 reference for
            all eligible matrices; --standard-q4 creates the ordinary Q4 control.

            Poolside Laguna DFlash drafter checkpoints are detected by architecture and converted
            through the native drafter registry. Their context projection and per-head attention
            gates remain Q8; the larger draft projections use searched Q4.

            Laguna can additionally use --template to invoke the proven bounded-memory streaming
            converter while preserving its expert layout and exact Q8 router policy.
            """,
        version: WickMetadata.version
    )

    @Argument(help: "Unquantized local safetensors LLM or DFlash drafter directory.")
    var source: String

    @Argument(help: "New destination model directory.")
    var destination: String

    @Option(
        name: .customLong("template"),
        help:
            "Laguna-only standard Q4R8/G64 template; enables bounded shard conversion and preserves its layout."
    )
    var template: String?

    @Option(help: "MLX format: affine, mxfp4, mxfp8, or nvfp4. Explicit selection defaults to standard calibration.")
    var mode: QuantizerMode?

    @Option(
        help: "Weight bits; affine supports 2, 3, 4, 5, 6, or 8. Explicit selection defaults to standard calibration.")
    var bits: Int?

    @Option(name: .customLong("group-size"), help: "Group size: affine 32/64/128, MXFP4/MXFP8 32, NVFP4 16.")
    var requestedGroupSize: Int?

    @Option(help: "Calibration: standard or scale-search. ScaleSearch requires affine Q4/Q5 with G64 or G128.")
    var calibration: QuantizerCalibration?

    private var groupSize: Int { requestedGroupSize ?? mode?.defaultGroupSize ?? 64 }

    private func resolvedRecipe() throws -> QuantizationRecipe {
        try QuantizationRecipe.resolve(
            mode: mode, bits: bits, groupSize: requestedGroupSize, calibration: calibration,
            standardQ4: standardQ4, standardQ8: standardQ8)
    }

    @Flag(
        name: .customLong("gemma-group-policy"),
        help: "Opt-in source-based Gemma 3/4 policy; preserve tied heads and validate every output geometry.")
    var gemmaGroupPolicy = false

    @Option(
        name: .customLong("g128-module"),
        help: "Gemma affine matrix path or */? glob to use G128; requires --gemma-group-policy. Repeat as needed.")
    var g128ModulePatterns: [String] = []

    @Option(name: .customLong("activation-stats"), help: "Laguna --template expert-conditional calibration statistics.")
    var activationStats: String?

    @Option(name: .customLong("validation-stats"), help: "Disjoint Laguna dev statistics for --activation-stats.")
    var validationStats: String?

    @Option(
        name: .customLong("q8-module"),
        help: "Module path or */? glob to keep in standard affine Q8; repeat as needed."
    )
    var q8ModulePatterns: [String] = []

    @Option(
        name: .customLong("skip-module"),
        help: "Module path or */? glob to leave unquantized; repeat as needed."
    )
    var skipModulePatterns: [String] = []

    @Option(
        name: .customLong("max-shard-gib"),
        help: "Maximum output safetensors shard size in GiB for generic conversion."
    )
    var maximumShardGiB = 5.0

    @Flag(
        name: .customLong("standard-q4"),
        help:
            "Use ordinary MLX affine Q4 calibration for eligible matrices as a matched ScaleSearch control."
    )
    var standardQ4 = false

    @Flag(
        name: .customLong("standard-q8"),
        help: "Use standard MLX affine Q8 for all eligible matrices; incompatible with --standard-q4 and --template."
    )
    var standardQ8 = false

    @Option(
        name: .customLong("expert-batch"),
        help: "Laguna --template expert batch size."
    )
    var expertBatch = 16

    @Flag(help: "Resolve and print the complete module policy without writing output.")
    var dryRun = false

    @Flag(help: "Materialize one source module at a time and search Q4 in 512-row batches.")
    var boundedMemory = false

    @Flag(help: "Run conversion using CPU/system memory instead of the default accelerator.")
    var cpu = false

    @Flag(help: "Replace an existing generic-conversion destination.")
    var overwrite = false

    public mutating func validate() throws {
        let recipe = try resolvedRecipe()
        guard template == nil || (recipe.isAffineQ4 && [64, 128].contains(groupSize)) else {
            throw ValidationError("--template requires affine Q4 with group size 64 or 128.")
        }
        guard
            !gemmaGroupPolicy
                || (template == nil && recipe.mode == .affine && [4, 5, 6, 8].contains(recipe.bits)
                    && [32, 64].contains(groupSize))
        else {
            throw ValidationError(
                "--gemma-group-policy requires generic affine 4/5/6/8-bit conversion with base group size 32 or 64.")
        }
        guard g128ModulePatterns.isEmpty || gemmaGroupPolicy else {
            throw ValidationError("--g128-module requires --gemma-group-policy.")
        }
        guard activationStats == nil || (groupSize == 64 && recipe.usesScaleSearch) else {
            throw ValidationError("--activation-stats requires searched Q4 group-64.")
        }
        guard activationStats == nil || template != nil else {
            throw ValidationError("--activation-stats currently requires Laguna --template.")
        }
        guard validationStats == nil || activationStats != nil else {
            throw ValidationError("--validation-stats requires --activation-stats.")
        }
        _ = try QuantizerShardSize.bytes(fromGiB: maximumShardGiB)
        guard (1...256).contains(expertBatch) else {
            throw ValidationError("--expert-batch must be in 1...256.")
        }
        let blankPatterns = (q8ModulePatterns + skipModulePatterns + g128ModulePatterns).filter {
            $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        guard blankPatterns.isEmpty else {
            throw ValidationError("--q8-module, --skip-module and --g128-module patterns must be nonblank.")
        }
        if boundedMemory && template != nil {
            throw ValidationError(
                "--bounded-memory is for generic conversion; --template already streams Laguna shards.")
        }
        if template != nil {
            guard q8ModulePatterns.isEmpty, skipModulePatterns.isEmpty else {
                throw ValidationError(
                    "--template uses Laguna's exact Q4R8 layout; custom Q8 and skip patterns are not accepted."
                )
            }
            guard !overwrite else {
                throw ValidationError(
                    "--overwrite is not supported with --template; choose a new destination."
                )
            }
        }
    }

    public mutating func run() async throws {
        #if os(Linux)
            defer { clearStreams() }
        #endif

        if let template {
            try runLagunaTemplate(template)
            return
        }

        if dryRun || cpu {
            try await Device.withDefaultDevice(.cpu) {
                try await runGenericConversion()
            }
        } else {
            try await runGenericConversion()
        }
    }

    private func runLagunaTemplate(_ template: String) throws {
        let recipe = try resolvedRecipe()
        let standardQ4 = recipe.isStandardQ4
        let sourceURL = URL(fileURLWithPath: source).standardizedFileURL
        let descriptor = try readDescriptor(sourceURL: sourceURL)
        guard descriptor.modelType == "laguna" else {
            throw QuantizerError.invalidPolicy(
                "--template is Laguna-specific, but source model_type is '\(descriptor.modelType)'."
            )
        }
        guard !descriptor.isLagunaDFlash else {
            throw QuantizerError.invalidPolicy(
                "--template applies to the Laguna target, not its separate DFlash drafter checkpoint."
            )
        }
        print("Profile: laguna-q4r8-template-streaming")
        print(
            "Policy: \(standardQ4 ? "standard" : "searched") affine Q4 group-\(groupSize); standard affine Q8 routers preserved"
        )
        try LagunaScaleSearchRescorer.rescore(
            source: source,
            template: template,
            destination: destination,
            expertBatch: expertBatch,
            groupSize: groupSize,
            standardQ4: standardQ4,
            activationStats: activationStats,
            validationStats: validationStats,
            preflightOnly: dryRun,
            cpu: cpu
        )
    }

    private func runGenericConversion() async throws {
        let recipe = try resolvedRecipe()
        let sourceURL = URL(fileURLWithPath: source).standardizedFileURL
        let destinationURL = URL(fileURLWithPath: destination).standardizedFileURL
        let descriptor = try readDescriptor(sourceURL: sourceURL)
        guard !descriptor.isQuantized else {
            throw QuantizerError.invalidInput(
                "source config already declares quantization; requantization is intentionally unsupported"
            )
        }
        try QuantizerOutputTransaction.validate(
            sourceDirectory: sourceURL, destinationDirectory: destinationURL, overwrite: overwrite)

        let configurationData = try Data(
            contentsOf: sourceURL.appendingPathComponent("config.json"))
        if gemmaGroupPolicy && !["gemma3", "gemma3_text", "gemma4", "gemma4_text"].contains(descriptor.modelType) {
            throw QuantizerError.invalidPolicy("--gemma-group-policy supports native Gemma 3/4 text models only")
        }
        if descriptor.isLagunaDFlash {
            try await runDFlashConversion(
                sourceURL: sourceURL,
                destinationURL: destinationURL,
                descriptor: descriptor,
                configurationData: configurationData
            )
            return
        }

        var model: any BaseLanguageModel
        if descriptor.modelType == "talkie" {
            // Construct the converter explicitly: registering the runtime default would
            // fuse projection names before user Q8/skip policies can address them.
            // sanitize() also folds the canonical bare lm_head + gain before quantizing.
            model = TalkieModel(
                try JSONDecoder.json5().decode(TalkieConfiguration.self, from: configurationData),
                fuseProjections: false)
        } else {
            await LagunaModelRegistration.register()
            guard await LLMTypeRegistry.shared.contains(descriptor.modelType) else {
                throw QuantizerError.unsupportedModelType(descriptor.modelType)
            }
            model = try await LLMTypeRegistry.shared.createModel(
                configuration: configurationData,
                modelType: descriptor.modelType
            )
        }
        var omittedTiedHead = false
        if gemmaGroupPolicy {
            let sourceShards = try FileManager.default.contentsOfDirectory(
                at: sourceURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
            ).filter { $0.pathExtension == "safetensors" }.sorted { $0.path < $1.path }
            omittedTiedHead = try GemmaGroupSizePolicy.preservesGemma3TiedHead(
                sourceConfiguration: configurationData, shards: sourceShards)
            if omittedTiedHead {
                guard let native = model as? Gemma3TextModel else {
                    throw QuantizerError.invalidPolicy("tied Gemma 3 conversion requires the native text model")
                }
                model = Gemma3TiedHeadConversionModel(native)
            }
        }
        var plan = try makePlan(model: model, modelType: descriptor.modelType)
        if gemmaGroupPolicy {
            let selected = try resolvePatterns(
                g128ModulePatterns, paths: plan.quantizablePaths, optionName: "--g128-module")
            let modules = Dictionary(
                uniqueKeysWithValues: model.leafModules().flattened().compactMap {
                    path, module -> (String, GemmaGroupSizePolicy.Module)? in
                    guard plan.quantizablePaths.contains(path) else { return nil }
                    let kind: GemmaGroupSizePolicy.Module.Kind
                    if module is Linear {
                        kind = .linear
                    } else if module is SwitchLinear {
                        kind = .switchLinear
                    } else if module is Embedding {
                        kind = .embedding
                    } else {
                        kind = .unsupported
                    }
                    let shape = module.parameters().flattened().first { $0.0 == "weight" }?.1.shape ?? []
                    return (path, .init(shape: shape, kind: kind))
                })
            plan.gemmaGroupPolicy = try GemmaGroupSizePolicy(
                sourceConfiguration: configurationData, modules: modules, selectedModules: selected,
                q8Modules: plan.q8Paths, skippedModules: plan.skippedPaths, omittedTiedHead: omittedTiedHead,
                defaultBits: recipe.bits, defaultGroupSize: recipe.groupSize)
        }
        try printPlan(plan, device: dryRun ? "CPU policy inspection" : String(describing: Device.defaultDevice()))
        if dryRun { return }

        let transaction = try QuantizerOutputTransaction(
            sourceDirectory: sourceURL, destinationDirectory: destinationURL, overwrite: overwrite)
        defer { transaction.cleanup() }

        let q8 = ModelConversionQuantization(
            bits: 8,
            groupSize: 64,
            mode: .affine,
            calibration: .standard
        )
        let q8Paths = plan.q8Paths
        let skippedPaths = plan.skippedPaths
        let g128Paths = plan.gemmaGroupPolicy?.selectedModules ?? []
        let g128 = ModelConversionQuantization(
            bits: recipe.bits, groupSize: 128, mode: .affine,
            calibration: recipe.mlxCalibration)
        var options = ModelConversionOptions(
            bits: recipe.bits,
            groupSize: recipe.groupSize,
            mode: recipe.mode.mlxMode,
            calibration: recipe.mlxCalibration,
            maxShardSize: try QuantizerShardSize.bytes(fromGiB: maximumShardGiB),
            overwriteExistingOutput: false,
            quantizationPredicate: { path, _ in
                if skippedPaths.contains(path) {
                    return .skip
                }
                if q8Paths.contains(path) {
                    return .quantize(q8)
                }
                if g128Paths.contains(path) {
                    return .quantize(g128)
                }
                return .quantize()
            }
        )

        options.boundedMemory = boundedMemory

        let progressHandler: @Sendable (ModelConversionProgress) -> Void = { progress in
            let detail = progress.message.map { ": \($0)" } ?? ""
            print("[\(progress.stage.rawValue)]\(detail)")
        }
        let result: ModelConversionResult
        if descriptor.modelType == "talkie" || gemmaGroupPolicy {
            result = try MLXLMCommon.convert(
                modelDirectory: sourceURL, model: model, to: transaction.stagingDirectory,
                options: options, progressHandler: progressHandler)
        } else {
            result = try await LLMModelFactory.shared.convert(
                from: sourceURL, to: transaction.stagingDirectory,
                options: options, progressHandler: progressHandler)
        }
        if let policy = plan.gemmaGroupPolicy {
            let configurationURL = result.outputDirectory.appendingPathComponent("config.json")
            let original = try Data(contentsOf: configurationURL)
            let finalized = try policy.finalizedOutputConfiguration(original)
            if finalized != original { try finalized.write(to: configurationURL, options: .atomic) }
            try policy.validateOutputWeights(result.weightsURLs)
        }
        try writeProvenance(plan: plan, result: result, sourceURL: sourceURL)
        if let backup = try transaction.commit() {
            print("Previous output retained at \(backup.path); it could not be removed after publication.")
        }
        print(
            "Created \(destinationURL.path) with \(result.weightsURLs.count) weight shard(s)."
        )
    }

    private func runDFlashConversion(
        sourceURL: URL,
        destinationURL: URL,
        descriptor: SourceDescriptor,
        configurationData: Data
    ) async throws {
        let recipe = try resolvedRecipe()
        await LagunaDFlashRegistration.register()
        let model: any MTPDrafterModel
        do {
            model = try await MTPDrafterTypeRegistry.shared.createModel(
                configuration: configurationData,
                modelType: descriptor.modelType
            )
        } catch {
            throw QuantizerError.unsupportedDrafterArchitecture(
                "\(descriptor.modelType) / DFlashLagunaForCausalLM")
        }

        let plan = try makePlan(
            model: model,
            modelType: descriptor.modelType,
            profile: .lagunaDFlash
        )
        try printPlan(
            plan,
            device: dryRun ? "CPU policy inspection" : String(describing: Device.defaultDevice())
        )
        print("DFlash note: compare acceptance and end-to-end throughput against the BF16 drafter.")
        if dryRun { return }

        let transaction = try QuantizerOutputTransaction(
            sourceDirectory: sourceURL, destinationDirectory: destinationURL, overwrite: overwrite)
        defer { transaction.cleanup() }

        let q8 = ModelConversionQuantization(
            bits: 8,
            groupSize: 64,
            mode: .affine,
            calibration: .standard
        )
        let q8Paths = plan.q8Paths
        let skippedPaths = plan.skippedPaths
        var options = ModelConversionOptions(
            bits: recipe.bits,
            groupSize: recipe.groupSize,
            mode: recipe.mode.mlxMode,
            calibration: recipe.mlxCalibration,
            maxShardSize: try QuantizerShardSize.bytes(fromGiB: maximumShardGiB),
            overwriteExistingOutput: false,
            quantizationPredicate: { path, _ in
                if skippedPaths.contains(path) { return .skip }
                if q8Paths.contains(path) { return .quantize(q8) }
                return .quantize()
            }
        )
        options.boundedMemory = boundedMemory
        let result = try MLXLMCommon.convert(
            modelDirectory: sourceURL,
            model: model,
            to: transaction.stagingDirectory,
            options: options,
            progressHandler: { progress in
                let detail = progress.message.map { ": \($0)" } ?? ""
                print("[\(progress.stage.rawValue)]\(detail)")
            }
        )
        try writeProvenance(plan: plan, result: result, sourceURL: sourceURL)
        if let backup = try transaction.commit() {
            print("Previous output retained at \(backup.path); it could not be removed after publication.")
        }
        print(
            "Created \(destinationURL.path) with \(result.weightsURLs.count) DFlash weight shard(s)."
        )
    }

    private func readDescriptor(sourceURL: URL) throws -> SourceDescriptor {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: sourceURL.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw QuantizerError.invalidInput(
                "source directory does not exist: \(sourceURL.path)")
        }
        let configURL = sourceURL.appendingPathComponent("config.json")
        do {
            return try JSONDecoder.json5().decode(
                SourceDescriptor.self,
                from: Data(contentsOf: configURL)
            )
        } catch {
            throw QuantizerError.invalidInput(
                "cannot decode \(configURL.path): \(error.localizedDescription)"
            )
        }
    }

    private func makePlan(
        model: any BaseLanguageModel,
        modelType: String,
        profile requestedProfile: ArchitectureProfile? = nil
    ) throws -> ConversionPlan {
        let recipe = try resolvedRecipe()
        let standardQ8 = recipe.isStandardQ8
        let modules = model.leafModules().flattened()
        let quantizable = modules.filter { _, module in
            module is Quantizable && !(module is Quantized)
        }
        let quantizablePaths = Set(quantizable.map(\.0))
        guard !quantizablePaths.isEmpty else {
            throw QuantizerError.invalidPolicy(
                "model exposes no unquantized Quantizable modules")
        }

        let profile = requestedProfile ?? ArchitectureProfile.resolve(modelType: modelType)
        let mandatoryQ8Paths = Set(
            quantizablePaths.filter { path in
                profile.mandatoryQ8Suffixes.contains { path.hasSuffix($0) }
            }
        )
        if profile.requiresMandatoryQ8Match, mandatoryQ8Paths.isEmpty {
            throw QuantizerError.invalidPolicy(
                "profile \(profile.name) found no expected mandatory Q8 module matching "
                    + profile.mandatoryQ8Suffixes.joined(separator: ", ")
            )
        }

        let requestedQ8 = try resolvePatterns(
            q8ModulePatterns,
            paths: quantizablePaths,
            optionName: "--q8-module"
        )
        let requestedSkips = try resolvePatterns(
            skipModulePatterns,
            paths: quantizablePaths,
            optionName: "--skip-module"
        )
        let q8Paths =
            standardQ8
            ? quantizablePaths.subtracting(requestedSkips)
            : mandatoryQ8Paths.union(requestedQ8)
        let skippedMandatory = requestedSkips.intersection(mandatoryQ8Paths)
        guard skippedMandatory.isEmpty else {
            throw QuantizerError.invalidPolicy(
                "mandatory Q8 module(s) cannot be skipped: "
                    + skippedMandatory.sorted().joined(separator: ", ")
            )
        }
        let conflicts = requestedSkips.intersection(q8Paths.union(requestedQ8))
        guard conflicts.isEmpty else {
            throw QuantizerError.invalidPolicy(
                "modules cannot be both Q8 and skipped: "
                    + conflicts.sorted().joined(separator: ", ")
            )
        }

        let selectedQ4Paths = quantizablePaths.subtracting(q8Paths).subtracting(requestedSkips)
        var scaleSearchPaths = Set<String>()
        var standardQ4Paths = Set<String>()
        var standardPaths = Set<String>()
        var incompatiblePaths = [String]()
        for (path, module) in quantizable where selectedQ4Paths.contains(path) {
            if let width = groupWidth(module), width % groupSize != 0 {
                incompatiblePaths.append("\(path) (input width \(width), requires group-\(groupSize))")
            } else if !recipe.isAffineQ4 && !recipe.usesScaleSearch {
                standardPaths.insert(path)
            } else if module is Linear || module is SwitchLinear {
                scaleSearchPaths.insert(path)
            } else {
                // Embeddings and custom Quantizable modules retain standard MLX Q4.
                // They share the same load/runtime format but do not accept searched arrays.
                standardQ4Paths.insert(path)
            }
        }
        for (path, module) in quantizable where q8Paths.contains(path) {
            if let width = groupWidth(module), width % 64 != 0 {
                incompatiblePaths.append("\(path) (Q8 input width \(width))")
            }
        }
        guard incompatiblePaths.isEmpty else {
            throw QuantizerError.invalidPolicy(
                "selected group sizes cannot represent these modules; explicitly --skip-module them or use a compatible architecture: "
                    + incompatiblePaths.sorted().joined(separator: ", ")
            )
        }
        guard !scaleSearchPaths.isEmpty || !standardQ4Paths.isEmpty || !standardPaths.isEmpty || !q8Paths.isEmpty else {
            throw QuantizerError.invalidPolicy(
                "the selected policy skips every quantizable module"
            )
        }

        return ConversionPlan(
            modelType: modelType,
            profile: profile,
            quantizablePaths: quantizablePaths,
            scaleSearchPaths: scaleSearchPaths,
            standardQ4Paths: standardQ4Paths,
            standardPaths: standardPaths,
            q8Paths: q8Paths,
            mandatoryQ8Paths: mandatoryQ8Paths,
            skippedPaths: requestedSkips
        )
    }

    private func groupWidth(_ module: Module) -> Int? {
        if let linear = module as? Linear { return linear.weight.dim(-1) }
        if module is SwitchLinear {
            return module.parameters().flattened().first { $0.0 == "weight" }?.1.dim(-1)
        }
        if let embedding = module as? Embedding { return embedding.weight.dim(-1) }
        // Registered models may supply their own Quantizable matrix modules.
        return module.parameters().flattened().first { $0.0 == "weight" && $0.1.ndim >= 2 }?.1.dim(-1)
    }

    private func resolvePatterns(
        _ patterns: [String],
        paths: Set<String>,
        optionName: String
    ) throws -> Set<String> {
        var resolved = Set<String>()
        for rawPattern in patterns {
            var pattern = rawPattern.trimmingCharacters(in: .whitespacesAndNewlines)
            if pattern.hasSuffix(".weight") {
                pattern.removeLast(".weight".count)
            }
            let matches = paths.filter { Self.glob(pattern, matches: $0) }
            guard !matches.isEmpty else {
                throw QuantizerError.invalidPolicy(
                    "\(optionName) pattern '\(rawPattern)' matched no quantizable module"
                )
            }
            resolved.formUnion(matches)
        }
        return resolved
    }

    private static func glob(_ pattern: String, matches text: String) -> Bool {
        let pattern = Array(pattern)
        let text = Array(text)
        var patternIndex = 0
        var textIndex = 0
        var starIndex: Int?
        var retryTextIndex = 0

        while textIndex < text.count {
            if patternIndex < pattern.count,
                pattern[patternIndex] == "?" || pattern[patternIndex] == text[textIndex]
            {
                patternIndex += 1
                textIndex += 1
            } else if patternIndex < pattern.count, pattern[patternIndex] == "*" {
                starIndex = patternIndex
                patternIndex += 1
                retryTextIndex = textIndex
            } else if let starIndex {
                patternIndex = starIndex + 1
                retryTextIndex += 1
                textIndex = retryTextIndex
            } else {
                return false
            }
        }
        while patternIndex < pattern.count, pattern[patternIndex] == "*" {
            patternIndex += 1
        }
        return patternIndex == pattern.count
    }

    private func printPlan(_ plan: ConversionPlan, device: String) throws {
        let recipe = try resolvedRecipe()
        let standardQ4 = recipe.isStandardQ4
        let standardQ8 = recipe.isStandardQ8
        print("Model type: \(plan.modelType)")
        print("Profile: \(plan.profile.name)")
        print("Device: \(device)")
        if plan.modelType == "talkie" {
            print("Talkie: preserve source projection names; fold lm_head_gain into the weight before quantization.")
            print(
                "Quality status: conversion candidate; compare historical text and long-generation repetition against BF16."
            )
        }
        if standardQ8 {
            print("Policy: standard affine group-64 Q8 for every unskipped quantizable module")
        } else if standardQ4 {
            print("Policy: standard affine group-\(groupSize) Q4 with standard affine Q8/G64 overrides")
        } else if recipe.usesScaleSearch {
            print("Policy: affine group-\(groupSize) Q\(recipe.bits) ScaleSearch with standard affine Q8/G64 overrides")
        } else {
            print(
                "Policy: standard \(recipe.mode.rawValue) \(recipe.bits)-bit group-\(groupSize) with standard affine Q8/G64 overrides"
            )
        }
        print("Quantizable modules: \(plan.quantizablePaths.count)")
        if standardQ4 {
            print("  standard Q4 Linear/SwitchLinear: \(plan.scaleSearchPaths.count)")
        } else if recipe.usesScaleSearch {
            print("  Q\(recipe.bits) ScaleSearch Linear/SwitchLinear: \(plan.scaleSearchPaths.count)")
        }
        print("  standard Q\(recipe.bits) embedding/custom modules: \(plan.standardQ4Paths.count)")
        if !plan.standardPaths.isEmpty {
            print("  standard \(recipe.mode.rawValue) \(recipe.bits)-bit modules: \(plan.standardPaths.count)")
        }
        print("  standard Q8 modules: \(plan.q8Paths.count)")
        print("  skipped modules: \(plan.skippedPaths.count)")
        if let policy = plan.gemmaGroupPolicy {
            print("Gemma source policy: exact source tie preserved; head omitted on disk: \(policy.omittedTiedHead)")
            print("G128 \(recipe.bits)-bit module paths (other matrices use base G\(groupSize)/Q8/skip policy):")
            for path in policy.selectedModules.sorted() { print("  \(path)") }
            print("G128 does not use Midnight's opt-in G64 tail kernel; compare measured speed and held-out quality.")
        }
        if !plan.q8Paths.isEmpty {
            print("Q8 module paths:")
            for path in plan.q8Paths.sorted() {
                let origin = plan.mandatoryQ8Paths.contains(path) ? "mandatory" : "requested"
                print("  \(path) [\(origin)]")
            }
        }
        if !plan.skippedPaths.isEmpty {
            print("Skipped module paths:")
            for path in plan.skippedPaths.sorted() {
                print("  \(path)")
            }
        }
    }

    private func writeProvenance(
        plan: ConversionPlan,
        result: ModelConversionResult,
        sourceURL: URL
    ) throws {
        let recipe = try resolvedRecipe()
        let standardQ4 = recipe.isStandardQ4
        let standardQ8 = recipe.isStandardQ8
        let isDFlash = plan.profile.name == ArchitectureProfile.lagunaDFlash.name
        let standardQ4Paths =
            standardQ4
            ? plan.scaleSearchPaths.union(plan.standardQ4Paths).sorted()
            : plan.standardQ4Paths.sorted()
        let provenance = QuantizerProvenance(
            boundedMemory: boundedMemory,
            status:
                !recipe.usesLegacyProvenance
                ? "experimental_unbenchmarked_candidate"
                : standardQ8
                    ? "experimental_unbenchmarked_standard_q8_reference"
                    : standardQ4
                        ? (isDFlash
                            ? "experimental_unbenchmarked_dflash_standard_q4_control"
                            : "experimental_unbenchmarked_standard_q4_control")
                        : (isDFlash
                            ? "experimental_unbenchmarked_dflash_candidate"
                            : (plan.modelType == "talkie"
                                ? "experimental_unbenchmarked_talkie_candidate"
                                : "experimental_unbenchmarked_candidate")),
            algorithm: recipe.usesScaleSearch
                ? "q\(recipe.bits)r8_affine_scale_search_ls2"
                : "mlx_\(recipe.mode.rawValue)_q\(recipe.bits)_standard",
            createdAt: ISO8601DateFormatter().string(from: Date()),
            sourceModel: sourceURL.path,
            modelType: plan.modelType,
            architectureProfile: plan.profile.name,
            bits: recipe.bits,
            groupSize: groupSize,
            mode: recipe.mode.rawValue,
            calibration: recipe.calibration.rawValue,
            standardModules: plan.standardPaths.union(standardQ4Paths).sorted(),
            gemmaGroupPolicy: plan.gemmaGroupPolicy != nil,
            g128Modules: plan.gemmaGroupPolicy?.selectedModules.sorted() ?? [],
            omittedTiedHead: plan.gemmaGroupPolicy?.omittedTiedHead ?? false,
            tiedHeadQuantizationAliases: plan.gemmaGroupPolicy?.requiresTiedHeadQuantizationAlias == true
                ? ["lm_head": "model.embed_tokens"] : nil,
            q4ScaleSearchModules: recipe.usesScaleSearch && recipe.bits == 4 ? plan.scaleSearchPaths.sorted() : [],
            q5ScaleSearchModules: recipe.usesScaleSearch && recipe.bits == 5 ? plan.scaleSearchPaths.sorted() : [],
            standardQ4Modules: recipe.bits == 4 ? standardQ4Paths : [],
            q8Modules: plan.q8Paths.sorted(),
            mandatoryQ8Modules: plan.mandatoryQ8Paths.sorted(),
            skippedModules: plan.skippedPaths.sorted(),
            outputShards: result.weightsURLs.map(\.lastPathComponent).sorted()
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(provenance).write(
            to: result.outputDirectory.appendingPathComponent(
                !recipe.usesLegacyProvenance
                    ? "quantization.json"
                    : standardQ8
                        ? "standard-q8-quantization.json"
                        : standardQ4
                            ? "standard-q4-quantization.json"
                            : "scale-search-quantization.json"),
            options: .atomic
        )
    }
}
