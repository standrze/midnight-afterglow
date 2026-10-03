import CoreFoundation
import Foundation

/// A source-based Gemma policy for affine precision candidates and selectively grouped matrices.
///
/// Names and shapes are taken from the native MLXLLM model after construction.
public struct GemmaGroupSizePolicy: Sendable {
    public struct Module: Sendable {
        public enum Kind: Sendable {
            case linear, switchLinear, embedding, unsupported
        }

        public let shape: [Int]
        public let kind: Kind

        public init(shape: [Int], kind: Kind) {
            self.shape = shape
            self.kind = kind
        }
    }

    public struct InputError: Error, LocalizedError {
        public let message: String
        public var errorDescription: String? { message }
    }

    public let selectedModules: Set<String>
    public let omittedTiedHead: Bool
    public let defaultBits: Int
    public let defaultGroupSize: Int
    private let sourceConfiguration: Data
    private let modules: [String: Module]
    private let q8Modules: Set<String>
    private let skippedModules: Set<String>

    /// Native Gemma 3 recreates its head from the embedding tensor triplet on load.
    ///
    /// Its independent module path must therefore resolve the same nondefault Q8 geometry.
    public var requiresTiedHeadQuantizationAlias: Bool {
        omittedTiedHead && q8Modules.contains("model.embed_tokens")
    }

    /// Validates the complete policy before conversion creates any output directory.
    ///
    /// The source must be unquantized; selective G128 never applies to embeddings, heads or routers.
    public init(
        sourceConfiguration: Data,
        modules: [String: Module],
        selectedModules: Set<String>,
        q8Modules: Set<String> = [],
        skippedModules: Set<String> = [],
        omittedTiedHead: Bool = false,
        defaultBits: Int = 4,
        defaultGroupSize: Int = 64
    ) throws {
        guard [4, 5, 6, 8].contains(defaultBits), [32, 64].contains(defaultGroupSize) else {
            throw InputError(message: "Gemma source policy requires affine 4/5/6/8-bit weights with base G32 or G64")
        }
        let configuration = try Self.object(sourceConfiguration)
        let text = configuration["text_config"] as? [String: Any] ?? configuration
        let topType = configuration["model_type"] as? String
        let textType = text["model_type"] as? String
        let supported = ["gemma3", "gemma3_text", "gemma4", "gemma4_text"]
        guard let topType, supported.contains(topType),
            textType == nil || supported.contains(textType!)
        else {
            throw InputError(message: "--g128-module supports native Gemma 3/4 text models only")
        }
        // Presence, even null or conflicting aliases, is rejected deliberately.
        // Packed QAT/compressed-tensors sources need a separate grid-preserving importer.
        for value in [configuration, text] {
            guard value["quantization"] == nil, value["quantization_config"] == nil,
                value["compression_config"] == nil
            else {
                throw InputError(message: "Gemma G128 requires an unquantized source; requantization is unsupported")
            }
        }
        let allModules = Set(modules.keys)
        guard selectedModules.isSubset(of: allModules), q8Modules.isSubset(of: allModules),
            skippedModules.isSubset(of: allModules)
        else {
            throw InputError(message: "Gemma grouping policy contains a module absent from the native model")
        }
        guard selectedModules.isDisjoint(with: q8Modules.union(skippedModules)),
            q8Modules.isDisjoint(with: skippedModules)
        else {
            throw InputError(message: "a G128 module cannot also be Q8 or skipped")
        }
        for name in allModules.subtracting(skippedModules).sorted() {
            let module = modules[name]!
            let groupSize = selectedModules.contains(name) ? 128 : (q8Modules.contains(name) ? 64 : defaultGroupSize)
            guard (2...3).contains(module.shape.count), module.shape.allSatisfy({ $0 > 0 }),
                module.shape.last! % groupSize == 0
            else {
                throw InputError(message: "\(name): source matrix input width is incompatible with G\(groupSize)")
            }
            if selectedModules.contains(name) {
                let components = name.split(separator: ".")
                guard module.kind == .linear || module.kind == .switchLinear,
                    components.last != "lm_head", !components.contains("router"),
                    !components.contains(where: { $0.hasPrefix("embed_tokens") })
                else {
                    throw InputError(
                        message:
                            "\(name): embeddings, output heads, routers and custom modules cannot use selective G128")
                }
            }
        }
        self.sourceConfiguration = sourceConfiguration
        self.modules = modules
        self.selectedModules = selectedModules
        self.omittedTiedHead = omittedTiedHead
        self.defaultBits = defaultBits
        self.defaultGroupSize = defaultGroupSize
        self.q8Modules = q8Modules
        self.skippedModules = skippedModules
    }

    /// Adds the runtime head's Q8 alias before validation and output publication.
    ///
    /// Existing Q4 configurations are returned byte-for-byte unchanged.
    public func finalizedOutputConfiguration(_ data: Data) throws -> Data {
        guard requiresTiedHeadQuantizationAlias else {
            try validateOutputConfiguration(data)
            return data
        }
        var output = try Self.object(data)
        guard var primary = output["quantization"] as? [String: Any],
            let alias = output["quantization_config"] as? [String: Any],
            NSDictionary(dictionary: primary).isEqual(to: alias)
        else {
            throw InputError(message: "converted Gemma quantization aliases disagree")
        }
        let headGeometry: [String: Any] = ["bits": 8, "group_size": 64, "mode": "affine"]
        if let existing = primary["lm_head"], !Self.equal(existing, headGeometry) {
            throw InputError(message: "existing tied-head metadata conflicts with its Q8 embedding")
        }
        primary["lm_head"] = headGeometry
        output["quantization"] = primary
        output["quantization_config"] = primary
        let finalized = try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
        try validateOutputConfiguration(finalized)
        return finalized
    }

    /// Checks both aliases, all resolved geometries and source configuration preservation.
    ///
    /// This validates metadata; `validateOutputWeights` separately checks the packed tensors.
    public func validateOutputConfiguration(_ data: Data) throws {
        let source = try Self.object(sourceConfiguration)
        let output = try Self.object(data)
        guard let primary = output["quantization"] as? [String: Any],
            let alias = output["quantization_config"] as? [String: Any],
            NSDictionary(dictionary: primary).isEqual(to: alias),
            Self.integer(primary["bits"]) == defaultBits,
            Self.integer(primary["group_size"]) == defaultGroupSize,
            primary["mode"] as? String == "affine"
        else {
            throw InputError(message: "converted Gemma quantization aliases or default affine geometry disagree")
        }
        for (key, value) in source {
            guard let other = output[key], Self.equal(value, other) else {
                throw InputError(message: "conversion changed source configuration field '\(key)'")
            }
        }
        if requiresTiedHeadQuantizationAlias {
            guard let head = primary["lm_head"] as? [String: Any],
                Self.integer(head["bits"]) == 8, Self.integer(head["group_size"]) == 64,
                head["mode"] as? String == "affine"
            else {
                throw InputError(
                    message: "omitted Gemma 3 head must explicitly inherit its embedding's Q8/G64 geometry")
            }
        }
        for (name, value) in primary where !["bits", "group_size", "mode"].contains(name) {
            guard modules[name] != nil || (requiresTiedHeadQuantizationAlias && name == "lm_head") else {
                throw InputError(message: "converted quantization metadata names an unknown module: \(name)")
            }
            if let geometry = value as? [String: Any] {
                guard let bits = Self.integer(geometry["bits"]), [4, 5, 6, 8].contains(bits) else {
                    throw InputError(message: "converted module must retain at least four bits: \(name)")
                }
            }
        }
        for name in modules.keys.sorted() {
            if skippedModules.contains(name) {
                guard let flag = primary[name] as? NSNumber,
                    CFGetTypeID(flag) == CFBooleanGetTypeID(), !flag.boolValue
                else {
                    throw InputError(message: "skipped module is not explicitly disabled in output: \(name)")
                }
                continue
            }
            let geometry = primary[name] as? [String: Any] ?? primary
            guard Self.integer(geometry["bits"]) == (q8Modules.contains(name) ? 8 : defaultBits),
                Self.integer(geometry["group_size"])
                    == (selectedModules.contains(name) ? 128 : (q8Modules.contains(name) ? 64 : defaultGroupSize)),
                geometry["mode"] as? String == "affine",
                primary[name] == nil || primary[name] is [String: Any]
            else {
                throw InputError(message: "converted module geometry does not match the resolved policy: \(name)")
            }
        }
    }

    /// Reads only output safetensors headers before the output transaction is committed.
    ///
    /// Dimensions come from the native model; this does not establish finite values or model quality.
    public func validateOutputWeights(_ shards: [URL]) throws {
        let tensors = try Self.readHeaders(shards).mapValues(\.header)
        if omittedTiedHead, tensors.keys.contains(where: { $0.split(separator: ".").contains("lm_head") }) {
            throw InputError(message: "conversion duplicated the source's tied output head")
        }
        for (name, module) in modules {
            let weight = tensors[name + ".weight"]
            if skippedModules.contains(name) {
                guard Self.integers(weight?["shape"]) == module.shape,
                    ["BF16", "F16", "F32"].contains(weight?["dtype"] as? String ?? ""),
                    tensors[name + ".scales"] == nil, tensors[name + ".biases"] == nil
                else { throw InputError(message: "skipped matrix was altered or packed: \(name)") }
                continue
            }
            let bits = q8Modules.contains(name) ? 8 : defaultBits
            let groupSize = selectedModules.contains(name) ? 128 : (q8Modules.contains(name) ? 64 : defaultGroupSize)
            var packedShape = module.shape
            var gridShape = module.shape
            // Five- and six-bit codes span word boundaries; dividing by 32 / bits truncates.
            packedShape[packedShape.count - 1] = module.shape.last! / 32 * bits
            gridShape[gridShape.count - 1] /= groupSize
            let scales = tensors[name + ".scales"]
            let biases = tensors[name + ".biases"]
            guard Self.integers(weight?["shape"]) == packedShape, weight?["dtype"] as? String == "U32",
                Self.integers(scales?["shape"]) == gridShape, Self.integers(biases?["shape"]) == gridShape,
                let scaleType = scales?["dtype"] as? String, ["BF16", "F16", "F32"].contains(scaleType),
                biases?["dtype"] as? String == scaleType
            else {
                throw InputError(message: "packed weight/scale/bias headers do not match the resolved policy: \(name)")
            }
        }
    }

    /// Rejects packed sources and resolves Gemma 3's tied head without evaluating MLX arrays.
    ///
    /// Duplicate floating heads are tied only if their complete stored payloads match the embedding. Comparisons read at most two MiB at a time; the complete source weights are never materialized.
    public static func preservesGemma3TiedHead(sourceConfiguration: Data, shards: [URL]) throws -> Bool {
        let config = try object(sourceConfiguration)
        let text = config["text_config"] as? [String: Any] ?? config
        let tensors = try readHeaders(shards)
        guard !tensors.isEmpty else { throw InputError(message: "source has no safetensors weights") }
        for (name, tensor) in tensors {
            let dtype = tensor.header["dtype"] as? String ?? ""
            let shape = integers(tensor.header["shape"]) ?? []
            guard !name.hasSuffix(".scales"), !name.hasSuffix(".biases"),
                !name.hasSuffix(".weight_packed"),
                shape.count < 2 || ["BF16", "F16", "F32", "F64"].contains(dtype)
            else {
                throw InputError(message: "source contains packed weights or quantization metadata: \(name)")
            }
        }
        guard ["gemma3", "gemma3_text"].contains(config["model_type"] as? String ?? "") else { return false }
        let prefixes = ["", "language_model."]
        let pairs = prefixes.compactMap { prefix -> (TensorRecord, TensorRecord?)? in
            guard let embedding = tensors[prefix + "model.embed_tokens.weight"] else { return nil }
            return (embedding, tensors[prefix + "lm_head.weight"])
        }
        guard pairs.count == 1, let (embedding, head) = pairs.first else {
            throw InputError(message: "Gemma 3 requires one unambiguous native or language_model embedding")
        }
        let declaredTie = text["tie_word_embeddings"] as? Bool ?? config["tie_word_embeddings"] as? Bool
        guard let head else {
            guard declaredTie != false else {
                throw InputError(message: "Gemma 3 declares untied embeddings but has no output-head weight")
            }
            return true
        }
        if declaredTie == false { return false }
        let matches = try samePayload(embedding, head)
        guard declaredTie != true || matches else {
            throw InputError(message: "Gemma 3 declares a tied head but its stored head differs from the embedding")
        }
        return matches
    }

    private static func samePayload(_ first: TensorRecord, _ second: TensorRecord) throws -> Bool {
        guard integers(first.header["shape"]) == integers(second.header["shape"]),
            first.header["dtype"] as? String == second.header["dtype"] as? String,
            let firstOffsets = integers(first.header["data_offsets"]),
            let secondOffsets = integers(second.header["data_offsets"]),
            firstOffsets[1] - firstOffsets[0] == secondOffsets[1] - secondOffsets[0]
        else { return false }
        let firstFile = try FileHandle(forReadingFrom: first.file)
        defer { try? firstFile.close() }
        let secondFile = try FileHandle(forReadingFrom: second.file)
        defer { try? secondFile.close() }
        try firstFile.seek(toOffset: first.payloadStart + UInt64(firstOffsets[0]))
        try secondFile.seek(toOffset: second.payloadStart + UInt64(secondOffsets[0]))
        var remaining = firstOffsets[1] - firstOffsets[0]
        while remaining > 0 {
            let count = min(1_048_576, remaining)
            let firstBytes = try firstFile.read(upToCount: count) ?? Data()
            let secondBytes = try secondFile.read(upToCount: count) ?? Data()
            guard firstBytes.count == count, secondBytes.count == count else {
                throw InputError(message: "truncated source weights while comparing tied head")
            }
            if firstBytes != secondBytes { return false }
            remaining -= count
        }
        return true
    }

    private struct TensorRecord {
        let header: [String: Any]
        let file: URL
        let payloadStart: UInt64
    }

    private static func readHeaders(_ shards: [URL]) throws -> [String: TensorRecord] {
        var tensors = [String: TensorRecord]()
        for shard in shards {
            let file = try FileHandle(forReadingFrom: shard)
            defer { try? file.close() }
            let prefix = try file.read(upToCount: 8) ?? Data()
            guard prefix.count == 8 else { throw InputError(message: "truncated safetensors prefix") }
            let length = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << (8 * $1.offset) }
            let fileSize = try file.seekToEnd()
            guard length > 0, length <= 100_000_000, fileSize >= 8, length <= fileSize - 8 else {
                throw InputError(message: "invalid safetensors header length")
            }
            try file.seek(toOffset: 8)
            let data = try file.read(upToCount: Int(length)) ?? Data()
            guard data.count == Int(length) else { throw InputError(message: "truncated safetensors header") }
            let header = try Self.object(data)
            var ranges = [(Int, Int)]()
            for (name, value) in header where name != "__metadata__" {
                guard tensors[name] == nil, let tensor = value as? [String: Any],
                    let shape = Self.integers(tensor["shape"]), shape.allSatisfy({ $0 >= 0 }),
                    let offsets = Self.integers(tensor["data_offsets"]), offsets.count == 2,
                    offsets[0] >= 0, offsets[1] >= offsets[0],
                    UInt64(offsets[1]) <= fileSize - 8 - length,
                    let dtype = tensor["dtype"] as? String, let bytes = Self.dtypeBytes[dtype]
                else {
                    throw InputError(message: "invalid, duplicate or unsupported output tensor: \(name)")
                }
                var tensorBytes = bytes
                for dimension in shape {
                    let product = tensorBytes.multipliedReportingOverflow(by: dimension)
                    guard !product.overflow else { throw InputError(message: "tensor extent overflow: \(name)") }
                    tensorBytes = product.partialValue
                }
                guard offsets[1] - offsets[0] == tensorBytes else {
                    throw InputError(message: "output tensor extent disagrees with shape/dtype: \(name)")
                }
                if tensorBytes > 0 { ranges.append((offsets[0], offsets[1])) }
                tensors[name] = TensorRecord(header: tensor, file: shard, payloadStart: 8 + length)
            }
            ranges.sort { $0.0 < $1.0 }
            guard zip(ranges, ranges.dropFirst()).allSatisfy({ pair in pair.0.1 <= pair.1.0 }) else {
                throw InputError(message: "overlapping output tensor extents")
            }
        }
        return tensors
    }

    private static let dtypeBytes = [
        "BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2, "BF16": 2, "F16": 2,
        "F32": 4, "U32": 4, "I32": 4, "U64": 8, "I64": 8, "F64": 8,
    ]

    private static func object(_ data: Data) throws -> [String: Any] {
        guard let value = try JSONSerialization.jsonObject(with: data, options: [.json5Allowed]) as? [String: Any]
        else {
            throw InputError(message: "expected a JSON configuration or safetensors header object")
        }
        return value
    }

    private static func integer(_ value: Any?) -> Int? {
        guard let value = value as? NSNumber, CFGetTypeID(value) != CFBooleanGetTypeID(),
            value.doubleValue.isFinite, value.doubleValue.rounded() == value.doubleValue,
            value.doubleValue >= 0, value.doubleValue < Double(Int.max)
        else { return nil }
        return value.intValue
    }

    private static func integers(_ value: Any?) -> [Int]? {
        guard let values = value as? [Any] else { return nil }
        let result = values.compactMap(integer)
        return result.count == values.count ? result : nil
    }

    private static func equal(_ first: Any, _ second: Any) -> Bool {
        NSDictionary(dictionary: ["value": first]).isEqual(to: ["value": second])
    }
}
