import Foundation
import MLX

public enum GemmaActivationStatisticsError: Error, LocalizedError {
    case invalid(String)

    public var errorDescription: String? {
        switch self {
        case .invalid(let message): "Invalid Gemma activation statistics: \(message)"
        }
    }
}

/// Full-content reproducibility identity, not publisher authentication.
///
/// Capture before collection; publication/load recapture it to reject changed sources.
public struct GemmaActivationSourceIdentity: Codable, Equatable {
    public let directory: String
    public let configFingerprint: String
    public let indexFingerprint: String
    public let weightsFingerprint: String
    public let tokenizerFingerprints: [String: String]

    public static func capture(source: URL) throws -> Self {
        let root = source.standardizedFileURL.resolvingSymlinksInPath()
        let config = try Data(contentsOf: root.appendingPathComponent("config.json"))
        let index = try Data(contentsOf: root.appendingPathComponent("model.safetensors.index.json"))
        guard let object = try JSONSerialization.jsonObject(with: config) as? [String: Any],
            let type = object["model_type"] as? String, ["gemma4", "gemma4_text"].contains(type)
        else { throw GemmaActivationStatisticsError.invalid("expected a Gemma 4 source") }
        let text = object["text_config"] as? [String: Any] ?? object
        for config in [object, text] {
            for key in ["quantization", "quantization_config"] {
                if let value = config[key], !(value is NSNull) {
                    throw GemmaActivationStatisticsError.invalid("quantized source configuration")
                }
            }
        }
        var tokenizer = [String: String]()
        for name in [
            "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
            "chat_template.jinja",
        ] {
            let file = root.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: file.path) {
                tokenizer[name] = GemmaActivationProvenance.fingerprint(try Data(contentsOf: file))
            }
        }
        guard tokenizer["tokenizer.json"] != nil || tokenizer["tokenizer.model"] != nil else {
            throw GemmaActivationStatisticsError.invalid("missing source tokenizer assets")
        }
        return Self(
            directory: root.path,
            configFingerprint: GemmaActivationProvenance.fingerprint(config),
            indexFingerprint: GemmaActivationProvenance.fingerprint(index),
            weightsFingerprint: try IndexedSafetensorsFingerprint.compute(directory: root, indexData: index),
            tokenizerFingerprints: tokenizer)
    }

    public func requireUnchanged() throws {
        guard try Self.capture(source: URL(fileURLWithPath: directory)) == self else {
            throw GemmaActivationStatisticsError.invalid("source weights, configuration, index or tokenizer changed")
        }
    }
}

public struct GemmaActivationProvenance: Codable, Equatable {
    public let source: GemmaActivationSourceIdentity
    public let corpusFingerprint: String
    public let sampleTokenFingerprints: [String]
    public let segmentTokenFingerprints: [String]
    public let segmentTokenCounts: [Int]
    public let sourceFamilies: [String]
    public let observedTokenCount: Int

    public init(
        source: GemmaActivationSourceIdentity, corpus: Data,
        tokenSamples: [[Int]], tokenSegments: [[Int]], sourceFamilies: [String]
    ) throws {
        guard !corpus.isEmpty, !tokenSamples.isEmpty, !tokenSegments.isEmpty,
            tokenSamples.allSatisfy({ !$0.isEmpty }), tokenSegments.allSatisfy({ !$0.isEmpty }),
            !sourceFamilies.isEmpty,
            sourceFamilies.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty })
        else { throw GemmaActivationStatisticsError.invalid("empty corpus, samples, segments or source families") }
        var sampleIndex = 0
        var offset = 0
        var count = 0
        for segment in tokenSegments {
            guard sampleIndex < tokenSamples.count,
                segment.count <= tokenSamples[sampleIndex].count - offset,
                Array(tokenSamples[sampleIndex][offset..<(offset + segment.count)]) == segment,
                segment.allSatisfy({ $0 >= 0 && $0 <= Int(Int32.max) })
            else {
                throw GemmaActivationStatisticsError.invalid(
                    "segments must partition samples in order without crossing sample boundaries")
            }
            let (next, overflow) = count.addingReportingOverflow(segment.count)
            guard !overflow, next <= Int(Int32.max) else {
                throw GemmaActivationStatisticsError.invalid("observed token count overflow")
            }
            count = next
            offset += segment.count
            if offset == tokenSamples[sampleIndex].count {
                sampleIndex += 1
                offset = 0
            }
        }
        guard sampleIndex == tokenSamples.count, offset == 0 else {
            throw GemmaActivationStatisticsError.invalid("segments omit sample tokens")
        }
        self.source = source
        corpusFingerprint = Self.fingerprint(corpus)
        sampleTokenFingerprints = tokenSamples.map(Self.tokenFingerprint)
        segmentTokenFingerprints = tokenSegments.map(Self.tokenFingerprint)
        segmentTokenCounts = tokenSegments.map(\.count)
        self.sourceFamilies = Array(Set(sourceFamilies.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }))
            .sorted()
        observedTokenCount = count
    }

    public func requireDisjoint(from other: Self) throws {
        try validate()
        try other.validate()
        guard source == other.source,
            corpusFingerprint != other.corpusFingerprint,
            Set(sampleTokenFingerprints).isDisjoint(with: Set(other.sampleTokenFingerprints)),
            Set(segmentTokenFingerprints).isDisjoint(with: Set(other.segmentTokenFingerprints)),
            Set(sourceFamilies).isDisjoint(with: Set(other.sourceFamilies))
        else {
            throw GemmaActivationStatisticsError.invalid(
                "fit/development source identity or corpus/sample/segment/family separation failed")
        }
    }

    public static func fingerprint(_ data: Data) -> String {
        var value: UInt64 = 14_695_981_039_346_656_037
        for byte in data { value = (value ^ UInt64(byte)) &* 1_099_511_628_211 }
        return String(format: "fnv1a64:%016llx", value)
    }

    private static func tokenFingerprint(_ tokens: [Int]) -> String {
        var bytes = Data("gemma-token-sequence-v1\0".utf8)
        for value in [tokens.count] + tokens {
            let integer = UInt64(value)
            for shift in stride(from: 0, through: 56, by: 8) {
                bytes.append(UInt8((integer >> shift) & 255))
            }
        }
        return fingerprint(bytes)
    }

    fileprivate func validate() throws {
        func validHash(_ value: String) -> Bool {
            value.hasPrefix("fnv1a64:") && value.count == 24
                && value.dropFirst(8).allSatisfy({ "0123456789abcdef".contains($0) })
        }
        var sum = 0
        for count in segmentTokenCounts {
            let (next, overflow) = sum.addingReportingOverflow(count)
            guard count > 0, !overflow else { throw GemmaActivationStatisticsError.invalid("invalid segment counts") }
            sum = next
        }
        guard observedTokenCount > 0, observedTokenCount <= Int(Int32.max), sum == observedTokenCount,
            !sampleTokenFingerprints.isEmpty, segmentTokenCounts.count == segmentTokenFingerprints.count,
            !sourceFamilies.isEmpty, Set(sourceFamilies).count == sourceFamilies.count,
            sourceFamilies.allSatisfy({
                !$0.isEmpty && $0 == $0.trimmingCharacters(in: .whitespacesAndNewlines)
            }),
            ([corpusFingerprint, source.configFingerprint, source.indexFingerprint, source.weightsFingerprint]
                + sampleTokenFingerprints + segmentTokenFingerprints + Array(source.tokenizerFingerprints.values))
                .allSatisfy(validHash),
            !source.tokenizerFingerprints.isEmpty
        else { throw GemmaActivationStatisticsError.invalid("invalid or incomplete provenance") }
    }
}

public struct GemmaActivationStatistics {
    public static let momentSuffix = ".input_second_moment"
    public static let expertCountSuffix = ".expert_position_count"
    public let moments: [String: MLXArray]
    public let expertCounts: [String: [Int]]
    public let provenance: GemmaActivationProvenance
    public let minimumExpertPositions: Int
    public let expertsPerToken: Int

    private struct Manifest: Codable {
        let format: String
        let algorithm: String
        let sourceWeightFingerprintMethod: String
        let collectionStrategy: String
        let routerWeighting: String
        let insufficientCoveragePolicy: String
        let provenance: GemmaActivationProvenance
        let minimumExpertPositions: Int
        let expertsPerToken: Int
        let projectionShapes: [String: [Int]]
    }

    public init(
        moments: [String: MLXArray], expertCounts: [String: [Int]],
        provenance: GemmaActivationProvenance, minimumExpertPositions: Int, expertsPerToken: Int
    ) throws {
        self.moments = moments
        self.expertCounts = expertCounts
        self.provenance = provenance
        self.minimumExpertPositions = minimumExpertPositions
        self.expertsPerToken = expertsPerToken
        try validate()
    }

    /// Call with the projection inventory independently obtained from the source model, not the manifest's own inventory.
    ///
    /// All tensors must match exactly.
    public static func load(
        from url: URL, source: URL, expectedProjectionShapes: [String: [Int]],
        expectedExpertsPerToken: Int, expectedMinimumExpertPositions: Int
    ) throws -> Self {
        let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: url, stream: .cpu)
        guard let encoded = metadata["gemma_activation_manifest"], metadata.count == 1 else {
            throw GemmaActivationStatisticsError.invalid("missing or unexpected statistics metadata")
        }
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(encoded.utf8))
        guard manifest.format == "gemma4_activation_stats_v1",
            manifest.algorithm == "input_channel_second_moment_with_expert_conditioning",
            manifest.sourceWeightFingerprintMethod == IndexedSafetensorsFingerprint.method,
            manifest.collectionStrategy == "layer_major_independent_segment_spool",
            manifest.routerWeighting == "conditional_on_selection_no_routing_score_weight",
            manifest.insufficientCoveragePolicy == "retain_template_expert",
            manifest.projectionShapes == expectedProjectionShapes,
            manifest.expertsPerToken == expectedExpertsPerToken,
            manifest.minimumExpertPositions == expectedMinimumExpertPositions,
            manifest.provenance.source == (try GemmaActivationSourceIdentity.capture(source: source))
        else { throw GemmaActivationStatisticsError.invalid("source, inventory or supported objective mismatch") }
        var moments = [String: MLXArray]()
        var counts = [String: [Int]]()
        for (key, array) in arrays {
            if key.hasSuffix(momentSuffix) {
                moments[String(key.dropLast(momentSuffix.count))] = array
            } else if key.hasSuffix(expertCountSuffix) {
                guard array.dtype == .int32, array.ndim == 1 else {
                    throw GemmaActivationStatisticsError.invalid("invalid expert count tensor")
                }
                try MLX.checkedEval(array)
                counts[String(key.dropLast(expertCountSuffix.count))] = array.asArray(Int32.self).map(Int.init)
            } else {
                throw GemmaActivationStatisticsError.invalid("unexpected statistics tensor: \(key)")
            }
        }
        guard moments.mapValues(\.shape) == expectedProjectionShapes else {
            throw GemmaActivationStatisticsError.invalid("statistics projection inventory mismatch")
        }
        return try Self(
            moments: moments, expertCounts: counts, provenance: manifest.provenance,
            minimumExpertPositions: manifest.minimumExpertPositions, expertsPerToken: manifest.expertsPerToken)
    }

    /// Publish to a new file only; source changes or a failed write leave the existing destination untouched.
    ///
    /// The temporary file stays beside its target.
    public func write(to destination: URL) throws {
        try validate()
        try provenance.source.requireUnchanged()
        let destination = destination.standardizedFileURL
        let source = URL(fileURLWithPath: provenance.source.directory).standardizedFileURL.resolvingSymlinksInPath()
        let parent = destination.deletingLastPathComponent().resolvingSymlinksInPath()
        guard parent != source, !parent.path.hasPrefix(source.path + "/"),
            !FileManager.default.fileExists(atPath: destination.path)
        else { throw GemmaActivationStatisticsError.invalid("destination exists or is inside the source") }
        let temporary = parent.appendingPathComponent(".gemma-stats-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: temporary) }
        var arrays = moments.reduce(into: [String: MLXArray]()) { $0[$1.key + Self.momentSuffix] = $1.value }
        for (path, counts) in expertCounts {
            arrays[path + Self.expertCountSuffix] = MLXArray(counts.map(Int32.init))
        }
        let manifest = Manifest(
            format: "gemma4_activation_stats_v1", algorithm: "input_channel_second_moment_with_expert_conditioning",
            sourceWeightFingerprintMethod: IndexedSafetensorsFingerprint.method,
            collectionStrategy: "layer_major_independent_segment_spool",
            routerWeighting: "conditional_on_selection_no_routing_score_weight",
            insufficientCoveragePolicy: "retain_template_expert", provenance: provenance,
            minimumExpertPositions: minimumExpertPositions, expertsPerToken: expertsPerToken,
            projectionShapes: moments.mapValues(\.shape))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let metadata = ["gemma_activation_manifest": String(decoding: try encoder.encode(manifest), as: UTF8.self)]
        try MLX.save(arrays: arrays, metadata: metadata, url: temporary)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: temporary.path)
        try FileManager.default.moveItem(at: temporary, to: destination)
    }

    public func requireDisjoint(from other: Self) throws {
        try provenance.requireDisjoint(from: other.provenance)
        guard moments.mapValues(\.shape) == other.moments.mapValues(\.shape),
            minimumExpertPositions == other.minimumExpertPositions, expertsPerToken == other.expertsPerToken
        else { throw GemmaActivationStatisticsError.invalid("fit/development geometry or coverage policy mismatch") }
    }

    private func validate() throws {
        try provenance.validate()
        guard minimumExpertPositions > 0, expertsPerToken >= 0, !moments.isEmpty else {
            throw GemmaActivationStatisticsError.invalid("invalid coverage threshold or empty projection inventory")
        }
        let expertPaths = Set(moments.filter { $0.value.ndim == 2 }.keys)
        guard expertPaths == Set(expertCounts.keys), !expertPaths.isEmpty || expertsPerToken == 0 else {
            throw GemmaActivationStatisticsError.invalid("expert count inventory mismatch")
        }
        for (path, moment) in moments {
            guard !path.isEmpty, !path.hasSuffix(Self.momentSuffix), !path.hasSuffix(Self.expertCountSuffix),
                moment.dtype == .float32, [1, 2].contains(moment.ndim), moment.shape.allSatisfy({ $0 > 0 })
            else { throw GemmaActivationStatisticsError.invalid("invalid moment tensor: \(path)") }
            try MLX.checkedEval(moment)
            guard moment.asArray(Float.self).allSatisfy({ $0.isFinite && $0 >= 0 }) else {
                throw GemmaActivationStatisticsError.invalid("non-finite or negative moments: \(path)")
            }
            if moment.ndim == 2 {
                guard expertsPerToken > 0, expertsPerToken <= moment.dim(0),
                    let counts = expertCounts[path], counts.count == moment.dim(0),
                    counts.allSatisfy({ $0 >= 0 && $0 <= provenance.observedTokenCount })
                else { throw GemmaActivationStatisticsError.invalid("invalid expert coverage: \(path)") }
                let (expected, overflow) = provenance.observedTokenCount.multipliedReportingOverflow(
                    by: expertsPerToken)
                let sum = counts.reduce(Int64(0)) { $0 + Int64($1) }
                guard !overflow, sum == Int64(expected) else {
                    throw GemmaActivationStatisticsError.invalid(
                        "routed positions do not match token/top-k coverage: \(path)")
                }
                let values = moment.asArray(Float.self)
                for expert in counts.indices where counts[expert] == 0 {
                    guard values[(expert * moment.dim(1))..<((expert + 1) * moment.dim(1))].allSatisfy({ $0 == 0 })
                    else {
                        throw GemmaActivationStatisticsError.invalid("nonzero moments for unobserved expert")
                    }
                }
            }
        }
    }
}
