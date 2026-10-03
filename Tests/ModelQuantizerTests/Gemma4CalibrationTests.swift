import AfterglowModelSupport
import Foundation
import MLX
@_spi(GemmaEncoder) import MLXLLM
import MLXLMCommon
import MLXNN
import MistralActivationScaleSearchCore
import XCTest

final class Gemma4CalibrationTests: XCTestCase {
    func testDenseAndRoutedTapsPreserveNativeLayersAndConditionalMoments() throws {
        try Device.withDefaultDevice(.gpu) {
            for nested in [false, true] {
                for moe in [false, true] {
                    let data = try configuration(moe: moe, nested: nested)
                    let descriptor = try Gemma4CalibrationConfiguration(data: data)
                    let root = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
                    let text = root["text_config"] as? [String: Any] ?? root
                    let native = Gemma4TextModel(
                        try JSONDecoder().decode(
                            Gemma4TextConfiguration.self, from: JSONSerialization.data(withJSONObject: text)))
                    let values = native.parameters().flattened().map { name, tensor in
                        let raw = (0..<tensor.size).map { Float(($0 * 7 + 3) % 41 - 20) / 41 }
                        return (name, MLXArray(raw).reshaped(tensor.shape).asType(.bfloat16))
                    }
                    try native.update(parameters: ModuleParameters.unflattened(values), verify: [.all])
                    native.train(false)
                    let weights = Dictionary(uniqueKeysWithValues: values)
                    for layerIndex in 0..<2 {
                        let prefix = "model.layers.\(layerIndex)."
                        var source = weights.filter { $0.key.hasPrefix(prefix) }
                        if nested {
                            source = Dictionary(
                                uniqueKeysWithValues: source.map { key, value in
                                    (
                                        key.replacingOccurrences(
                                            of: "model.layers.", with: "model.language_model.layers."), value
                                    )
                                })
                        }
                        for tokens in [8, 40] {
                            var observed = [String: Int]()
                            let observer = try CheckingExpertObserver()
                            let inputRecorder = try Gemma4ProjectionInputRecorder(
                                targets: [
                                    .init(path: descriptor.moduleRoot + ".layers.\(layerIndex).self_attn.q_proj")
                                ],
                                maximumPositions: 3, maximumRetainedBytes: 8192)
                            let block = try Gemma4CalibrationBlock(
                                configuration: descriptor, layerIndex: layerIndex,
                                sourceWeights: source,
                                observeDense: { path, input in
                                    observed[path, default: 0] += input.size / input.dim(-1)
                                    inputRecorder.observeDense(path: path, input: input)
                                }, observeRouted: observer)
                            let raw = (0..<(tokens * 128)).map { Float($0 % 29 - 14) / 17 }
                            let hidden = MLXArray(raw).reshaped(1, tokens, 128).asType(.bfloat16)
                            let mask = createAttentionMask(
                                h: hidden, cache: nil,
                                windowSize: layerIndex == 0 ? 8 : nil)
                            let expected = native.model.layers[layerIndex](hidden, mask: mask).0
                            let actual = try block(hidden)
                            try MLX.checkedEval(expected, actual)
                            XCTAssertTrue(MLX.all(expected .== actual).item(Bool.self))
                            let captured = try XCTUnwrap(inputRecorder.finalize().first)
                            XCTAssertEqual(captured.observedPositions, tokens)
                            XCTAssertEqual(captured.inputs.shape, [3, 128])
                            XCTAssertEqual(Set(observed.keys), Set(block.denseProjectionWidths.keys))
                            XCTAssertTrue(observed.values.allSatisfy { $0 == tokens })
                            XCTAssertEqual(observed.count, moe ? 8 : 7)
                            XCTAssertEqual(block.routedProjectionPaths.count, moe ? 3 : 0)
                            if moe {
                                let results = try observer.recorder.finalize()
                                XCTAssertEqual(Set(results.map(\.path)), block.routedProjectionPaths)
                                for result in results {
                                    let reference = try XCTUnwrap(observer.expected[result.path])
                                    XCTAssertEqual(result.expertPositionCounts, reference.counts)
                                    XCTAssertEqual(result.expertPositionCounts.reduce(0, +), tokens * 2)
                                    let moments = result.secondMoments.asArray(Float.self)
                                    for expert in 0..<4 {
                                        for channel in 0..<reference.width {
                                            let index = expert * reference.width + channel
                                            let expectedMoment =
                                                reference.counts[expert] == 0
                                                ? 0
                                                : reference.sums[index] / Float(reference.counts[expert])
                                            XCTAssertEqual(moments[index], expectedMoment, accuracy: 0.00001)
                                        }
                                        XCTAssertEqual(result.eligibleExperts[expert], reference.counts[expert] >= 2)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    func testRejectsUnsupportedGeometryAndQuantizedTeachers() throws {
        let data = try configuration(moe: true, nested: false)
        let original = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        for replacement in [
            ["num_kv_shared_layers": 1], ["hidden_size_per_layer_input": 32],
            ["num_experts": 0], ["top_k_experts": 5], ["num_hidden_layers": 0],
            ["quantization": ["bits": 4]], ["layer_types": ["full_attention"]],
            ["rms_norm_eps": 0.00001], ["head_dim": 0],
        ] as [[String: Any]] {
            var changed = original
            changed.merge(replacement) { _, new in new }
            XCTAssertThrowsError(
                try Gemma4CalibrationConfiguration(
                    data: JSONSerialization.data(withJSONObject: changed)))
        }
    }

    func testOmittedNativeDefaultsDoNotPretendToDisablePLEOrKVSharing() throws {
        let data = try configuration(moe: false, nested: false)
        let root = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        for name in ["num_kv_shared_layers", "hidden_size_per_layer_input"] {
            var changed = root
            changed.removeValue(forKey: name)
            XCTAssertThrowsError(
                try Gemma4CalibrationConfiguration(
                    data: JSONSerialization.data(withJSONObject: changed)))
        }
    }

    func testPublisherFusedExpertsMatchNativeAndCPUFailsBeforeGather() throws {
        try Device.withDefaultDevice(.gpu) {
            let data = try configuration(moe: true, nested: true)
            let descriptor = try Gemma4CalibrationConfiguration(data: data)
            let root = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
            let text = try XCTUnwrap(root["text_config"] as? [String: Any])
            let native = Gemma4TextModel(
                try JSONDecoder().decode(
                    Gemma4TextConfiguration.self,
                    from: JSONSerialization.data(withJSONObject: text)))
            let parameters = native.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }
            try native.update(parameters: ModuleParameters.unflattened(parameters), verify: [.all])
            var source = Dictionary(
                uniqueKeysWithValues: parameters.filter {
                    $0.0.hasPrefix("model.layers.0.")
                })
            let prefix = "model.layers.0.experts.switch_glu."
            let gate = try XCTUnwrap(source.removeValue(forKey: prefix + "gate_proj.weight"))
            let up = try XCTUnwrap(source.removeValue(forKey: prefix + "up_proj.weight"))
            let down = try XCTUnwrap(source.removeValue(forKey: prefix + "down_proj.weight"))
            source["model.layers.0.experts.gate_up_proj"] = MLX.concatenated([gate, up], axis: -2)
            source["model.layers.0.experts.down_proj"] = down
            let observer = try CheckingExpertObserver()
            let block = try Gemma4CalibrationBlock(
                configuration: descriptor, layerIndex: 0,
                sourceWeights: source, observeRouted: observer)
            let hidden = MLX.ones([1, 8, 128], dtype: .bfloat16)
            let mask = createAttentionMask(h: hidden, cache: nil, windowSize: 8)
            let expected = native.model.layers[0](hidden, mask: mask).0
            let actual = try block(hidden)
            XCTAssertTrue(MLX.all(expected .== actual).item(Bool.self))
            XCTAssertEqual(try observer.recorder.finalize().count, 3)
            try Device.withDefaultDevice(.cpu) {
                XCTAssertThrowsError(try block(MLX.ones([1, 8, 128], dtype: .bfloat16)))
            }
        }
    }

    func testSourceAliasesAndInputFailuresAreRejected() throws {
        try Device.withDefaultDevice(.cpu) {
            let data = try configuration(moe: false, nested: false)
            let descriptor = try Gemma4CalibrationConfiguration(data: data)
            let native = Gemma4TextModel(try JSONDecoder().decode(Gemma4TextConfiguration.self, from: data))
            let source = Dictionary(
                uniqueKeysWithValues: native.parameters().flattened().filter {
                    $0.0.hasPrefix("model.layers.0.")
                }.map { ($0.0, $0.1.asType(.bfloat16)) })
            var duplicated = source
            duplicated["model.language_model.layers.0.self_attn.q_proj.weight"] =
                source["model.layers.0.self_attn.q_proj.weight"]
            XCTAssertThrowsError(
                try Gemma4CalibrationBlock(
                    configuration: descriptor,
                    layerIndex: 0, sourceWeights: duplicated))
            XCTAssertThrowsError(
                try Gemma4CalibrationBlock(
                    configuration: descriptor,
                    layerIndex: 2, sourceWeights: source))
            var wrongDType = source
            wrongDType["model.layers.0.self_attn.q_proj.weight"] = source["model.layers.0.self_attn.q_proj.weight"]?
                .asType(.float32)
            XCTAssertThrowsError(
                try Gemma4CalibrationBlock(
                    configuration: descriptor,
                    layerIndex: 0, sourceWeights: wrongDType))
            let block = try Gemma4CalibrationBlock(
                configuration: descriptor,
                layerIndex: 0, sourceWeights: source)
            XCTAssertThrowsError(try block(MLX.zeros([1, 2, 64], dtype: .bfloat16)))
            XCTAssertThrowsError(try block(MLX.zeros([1, 2, 128], dtype: .float32)))
        }
    }

    func testLayerMajorCollectorMatchesIndependentSegmentsAndCleansSpool() throws {
        try Device.withDefaultDevice(.gpu) {
            for moe in [false, true] {
                for nested in [false, true] {
                    let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
                    defer { try? FileManager.default.removeItem(at: temporary) }
                    let source = temporary.appendingPathComponent("source")
                    let spool = temporary.appendingPathComponent("spool")
                    try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
                    let data = try configuration(moe: moe, nested: nested)
                    try data.write(to: source.appendingPathComponent("config.json"))
                    let descriptor = try Gemma4CalibrationConfiguration(data: data)
                    let root = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
                    let text = root["text_config"] as? [String: Any] ?? root
                    let native = Gemma4TextModel(
                        try JSONDecoder().decode(
                            Gemma4TextConfiguration.self, from: JSONSerialization.data(withJSONObject: text)))
                    let values: [(String, MLXArray)] = native.parameters().flattened().map { name, tensor in
                        var raw = [Float]()
                        for index in 0..<tensor.size {
                            let pattern: Int = (index * 7 + 3) % 41 - 20
                            raw.append(Float(pattern) / Float(41))
                        }
                        let weight = MLXArray(raw).reshaped(tensor.shape).asType(.bfloat16)
                        return (name, weight)
                    }
                    try native.update(parameters: ModuleParameters.unflattened(values), verify: [.all])
                    native.train(false)
                    let weights = Dictionary(uniqueKeysWithValues: values)
                    let sourceWeights = Dictionary(
                        uniqueKeysWithValues: values.map { name, value in
                            (
                                nested ? name.replacingOccurrences(of: "model.", with: "model.language_model.") : name,
                                value
                            )
                        })
                    try MLX.save(arrays: sourceWeights, url: source.appendingPathComponent("model.safetensors"))
                    let index = [
                        "weight_map": Dictionary(
                            uniqueKeysWithValues: sourceWeights.keys.map {
                                ($0, "model.safetensors")
                            })
                    ]
                    try JSONSerialization.data(withJSONObject: index).write(
                        to: source.appendingPathComponent("model.safetensors.index.json"))
                    let segments = [Array(0..<5), Array(3..<18)]
                    var hidden = segments.map {
                        native.model.embedTokens(MLXArray($0).reshaped(1, $0.count)) * native.model.embedScale
                    }
                    var expectedSums = [String: [Float]]()
                    var expectedInputPrefixes = [String: [Float]]()
                    var referenceExperts = [String: CheckingExpertObserver.Reference]()
                    for layerIndex in 0..<2 {
                        let prefix = "model.layers.\(layerIndex)."
                        let observer = try CheckingExpertObserver()
                        let block = try Gemma4CalibrationBlock(
                            configuration: descriptor, layerIndex: layerIndex,
                            sourceWeights: weights.filter { $0.key.hasPrefix(prefix) },
                            observeDense: { path, input in
                                let width = input.dim(-1)
                                let values = input.asType(.float32).asArray(Float.self)
                                if path.hasSuffix("self_attn.q_proj"), expectedInputPrefixes[path] == nil {
                                    expectedInputPrefixes[path] = Array(values.prefix(2 * width))
                                }
                                var sums = expectedSums[path] ?? Array(repeating: Float(0), count: width)
                                for position in 0..<(values.count / width) {
                                    for channel in 0..<width {
                                        let value = values[position * width + channel]
                                        sums[channel] += value * value
                                    }
                                }
                                expectedSums[path] = sums
                            }, observeRouted: observer)
                        hidden = try hidden.map { try block($0) }
                        try MLX.checkedEval(hidden)
                        if moe { referenceExperts.merge(observer.expected) { existing, _ in existing } }
                    }
                    try Data("{}".utf8).write(to: source.appendingPathComponent("tokenizer.json"))
                    let provenance = try GemmaActivationProvenance(
                        source: GemmaActivationSourceIdentity.capture(source: source),
                        corpus: JSONSerialization.data(withJSONObject: segments),
                        tokenSamples: segments, tokenSegments: segments, sourceFamilies: ["gemma4-native-fixture"])
                    var projectionInputs = [Gemma4CapturedProjectionInputs]()
                    let result = try Gemma4ActivationCollector.collect(
                        source: source, tokenSegments: segments, spoolParent: spool, minimumExpertPositions: 2,
                        projectionInputRecorder: { layerIndex in
                            try Gemma4ProjectionInputRecorder(
                                targets: [
                                    .init(path: descriptor.moduleRoot + ".layers.\(layerIndex).self_attn.q_proj")
                                ],
                                maximumPositions: 2, maximumRetainedBytes: 8192)
                        },
                        consumeProjectionInputs: { _, inputs in
                            projectionInputs.append(contentsOf: inputs)
                        })
                    XCTAssertEqual(projectionInputs.count, 2)
                    for captured in projectionInputs {
                        XCTAssertEqual(captured.observedPositions, 20)
                        XCTAssertEqual(captured.inputs.shape, [2, 128])
                        XCTAssertEqual(
                            captured.inputs.asType(.float32).asArray(Float.self),
                            try XCTUnwrap(expectedInputPrefixes[captured.target.path]))
                    }
                    XCTAssertEqual(result.observedTokenCount, 20)
                    XCTAssertEqual(result.segmentTokenCounts, [5, 15])
                    XCTAssertEqual(result.dense.count, moe ? 17 : 15)
                    XCTAssertEqual(result.experts.count, moe ? 6 : 0)
                    for stat in result.dense {
                        XCTAssertEqual(stat.positionCount, 20)
                        let expected: [Float]
                        if stat.path.hasSuffix("lm_head") {
                            var sums = Array(repeating: Float(0), count: 128)
                            let tap = try GemmaHeadMomentTap(norm: native.model.norm)
                            try native.model.update(
                                modules: ModuleChildren.unflattened([("norm", tap)]), verify: [.noUnusedKeys])
                            for segment in segments {
                                let logits = native(MLXArray(segment).reshaped(1, segment.count), cache: nil)
                                try MLX.checkedEval(logits)
                                let normalized = try XCTUnwrap(tap.output)
                                let values = normalized.asType(.float32).asArray(Float.self)
                                for position in 0..<segment.count {
                                    for channel in 0..<128 {
                                        let value = values[position * 128 + channel]
                                        sums[channel] += value * value
                                    }
                                }
                            }
                            expected = sums
                        } else {
                            expected = try XCTUnwrap(expectedSums[stat.path])
                        }
                        for (actual, sum) in zip(stat.secondMoments.asArray(Float.self), expected) {
                            let reference = sum / 20
                            // FP32 tree reduction and sequential host addition have
                            // different rounding. Keep a two-ppm relative bound.
                            let tolerance = Swift.max(Float(0.000001), Swift.abs(reference) * Float(0.000002))
                            XCTAssertEqual(actual, reference, accuracy: tolerance)
                        }
                    }
                    for actual in result.experts {
                        let expected = try XCTUnwrap(referenceExperts[actual.path])
                        XCTAssertEqual(actual.expertPositionCounts, expected.counts)
                        let values = actual.secondMoments.asArray(Float.self)
                        for expert in expected.counts.indices {
                            for channel in 0..<expected.width {
                                let index = expert * expected.width + channel
                                let reference =
                                    expected.counts[expert] == 0
                                    ? Float(0) : expected.sums[index] / Float(expected.counts[expert])
                                let tolerance = Swift.max(Float(0.000001), Swift.abs(reference) * Float(0.000002))
                                XCTAssertEqual(values[index], reference, accuracy: tolerance)
                            }
                        }
                    }
                    var momentTensors = Dictionary(
                        uniqueKeysWithValues: result.dense.map { ($0.path, $0.secondMoments) })
                    momentTensors.merge(
                        Dictionary(uniqueKeysWithValues: result.experts.map { ($0.path, $0.secondMoments) })
                    ) {
                        existing, _ in existing
                    }
                    let statistics = try GemmaActivationStatistics(
                        moments: momentTensors,
                        expertCounts: Dictionary(
                            uniqueKeysWithValues: result.experts.map { ($0.path, $0.expertPositionCounts) }),
                        provenance: provenance, minimumExpertPositions: 2, expertsPerToken: moe ? 2 : 0)
                    var inventory = expectedSums.mapValues { [$0.count] }
                    for (path, reference) in referenceExperts {
                        inventory[path] = [reference.counts.count, reference.width]
                    }
                    inventory[nested ? "language_model.lm_head" : "lm_head"] = [native.model.norm.weight.size]
                    let statisticsURL = temporary.appendingPathComponent("statistics.safetensors")
                    try statistics.write(to: statisticsURL)
                    let reloaded = try GemmaActivationStatistics.load(
                        from: statisticsURL, source: source, expectedProjectionShapes: inventory,
                        expectedExpertsPerToken: moe ? 2 : 0, expectedMinimumExpertPositions: 2)
                    XCTAssertEqual(reloaded.provenance, provenance)
                    XCTAssertEqual(reloaded.expertCounts, statistics.expertCounts)
                    for (path, moment) in momentTensors {
                        XCTAssertEqual(reloaded.moments[path]?.asArray(Float.self), moment.asArray(Float.self))
                    }
                    XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: spool.path).isEmpty)
                    XCTAssertThrowsError(
                        try Gemma4ActivationCollector.collect(
                            source: source, tokenSegments: [[32]], spoolParent: spool))
                    XCTAssertThrowsError(
                        try Gemma4ActivationCollector.collect(
                            source: source, tokenSegments: segments, spoolParent: source.appendingPathComponent("spool")
                        ))
                    var invalid = sourceWeights
                    invalid[nested ? "model.language_model.norm.weight" : "model.norm.weight"] = MLX.ones(
                        [1], dtype: .bfloat16)
                    try MLX.save(arrays: invalid, url: source.appendingPathComponent("model.safetensors"))
                    XCTAssertThrowsError(
                        try Gemma4ActivationCollector.collect(
                            source: source, tokenSegments: segments, spoolParent: spool))
                    XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: spool.path).isEmpty)
                }
            }
        }
    }

    func testGemmaAWSSFitsNativeStatisticsPreservesFallbackAndReloads() throws {
        try Device.withDefaultDevice(.gpu) {
            for moe in [false, true] {
                let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
                defer { try? FileManager.default.removeItem(at: temporary) }
                let source = temporary.appendingPathComponent("source")
                try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
                let config = try configuration(moe: moe, nested: false)
                try config.write(to: source.appendingPathComponent("config.json"))
                let nativeConfig = try JSONDecoder().decode(Gemma4TextConfiguration.self, from: config)
                let native = Gemma4TextModel(nativeConfig)
                let original = Dictionary(
                    uniqueKeysWithValues: native.parameters().flattened().map { name, tensor in
                        var raw = [Float]()
                        for index in 0..<tensor.size {
                            let pattern: Int = (index * 7 + 3) % 41 - 20
                            raw.append(Float(pattern) / Float(41))
                        }
                        return (name, MLXArray(raw).reshaped(tensor.shape).asType(.bfloat16))
                    })
                let weightURL = source.appendingPathComponent("model.safetensors")
                try MLX.save(arrays: original, url: weightURL)
                let sourceBytes = try Data(contentsOf: weightURL)
                try JSONSerialization.data(withJSONObject: [
                    "weight_map": Dictionary(
                        uniqueKeysWithValues: original.keys.map { ($0, "model.safetensors") })
                ]).write(
                    to: source.appendingPathComponent("model.safetensors.index.json"))
                try Data("{}".utf8).write(to: source.appendingPathComponent("tokenizer.json"))
                let identity = try GemmaActivationSourceIdentity.capture(source: source)
                func statistics(tokens: [Int], family: String, threshold: Int = 2) throws -> GemmaActivationStatistics {
                    let collected = try Gemma4ActivationCollector.collect(
                        source: source,
                        tokenSegments: [tokens], spoolParent: temporary.appendingPathComponent("spool"),
                        minimumExpertPositions: threshold)
                    var moments = Dictionary(uniqueKeysWithValues: collected.dense.map { ($0.path, $0.secondMoments) })
                    for expert in collected.experts { moments[expert.path] = expert.secondMoments }
                    return try GemmaActivationStatistics(
                        moments: moments,
                        expertCounts: Dictionary(
                            uniqueKeysWithValues: collected.experts.map { ($0.path, $0.expertPositionCounts) }),
                        provenance: GemmaActivationProvenance(
                            source: identity, corpus: Data(family.utf8),
                            tokenSamples: [tokens], tokenSegments: [tokens], sourceFamilies: [family]),
                        minimumExpertPositions: threshold, expertsPerToken: moe ? 2 : 0)
                }
                let fit = try statistics(tokens: Array(1...8), family: "fit")
                let dev = try statistics(tokens: Array(9...16), family: "dev")
                let fitter = try GemmaActivationWeightedScaleSearch(calibration: fit, development: dev)
                XCTAssertThrowsError(try GemmaActivationWeightedScaleSearch(calibration: fit, development: fit))
                let selected =
                    ["model.layers.0.self_attn.q_proj"]
                    + (moe ? ["model.layers.0.experts.switch_glu.gate_proj", "model.layers.0.router.proj"] : [])
                var exported = original
                var quantization: [String: Any] = ["bits": 4, "group_size": 64, "mode": "affine"]
                for path in selected {
                    let matrix = try XCTUnwrap(original[path + ".weight"])
                    let bits = path.hasSuffix("router.proj") ? 8 : 4
                    let baseline: (weight: MLXArray, scales: MLXArray, biases: MLXArray)
                    if bits == 8 {
                        let standard = MLX.quantized(matrix, groupSize: 64, bits: 8)
                        baseline = (standard.wq, standard.scales, try XCTUnwrap(standard.biases))
                        quantization[path] = ["bits": 8, "group_size": 64, "mode": "affine"]
                    } else {
                        baseline = q4AffineScaleSearchQuantized(matrix)
                    }
                    let refined = try fitter.rescore(
                        modulePath: path, sourceWeight: matrix,
                        templateWeight: baseline.weight, templateScales: baseline.scales,
                        templateBiases: baseline.biases, bits: bits, groupSize: 64)
                    if bits == 4 {
                        let signed = try fitter.rescore(
                            modulePath: path, sourceWeight: matrix,
                            templateWeight: baseline.weight, templateScales: -MLX.abs(baseline.scales),
                            templateBiases: baseline.biases, bits: 4, groupSize: 64)
                        XCTAssertEqual(signed.scales.dtype, baseline.scales.dtype)
                        XCTAssertTrue(MLX.all(MLX.isFinite(signed.scales)).item(Bool.self))
                    }
                    for diagnostic in refined.diagnostics.values {
                        print(
                            "Gemma AWSS fixture: moe=\(moe) path=\(path) changed=\(diagnostic.changedGroupCount) fit=\(diagnostic.templateCalibrationWeightedMSE)->\(diagnostic.candidateCalibrationWeightedMSE) dev=\(diagnostic.templateValidationWeightedMSE ?? 0)->\(diagnostic.candidateValidationWeightedMSE ?? 0)"
                        )
                        XCTAssertLessThanOrEqual(
                            diagnostic.candidateCalibrationWeightedMSE,
                            diagnostic.templateCalibrationWeightedMSE + 1e-10)
                        XCTAssertLessThanOrEqual(
                            try XCTUnwrap(diagnostic.candidateValidationWeightedMSE),
                            try XCTUnwrap(diagnostic.templateValidationWeightedMSE) + 1e-10)
                    }
                    for expert in refined.retainedTemplateExperts {
                        for (actual, expected) in [
                            (refined.weight, baseline.weight), (refined.scales, baseline.scales),
                            (refined.biases, baseline.biases),
                        ] {
                            XCTAssertTrue(MLX.arrayEqual(actual[expert], expected[expert]).item(Bool.self))
                        }
                    }
                    if bits == 8 {
                        XCTAssertEqual(refined.retainedModuleReason, "protected_router")
                        XCTAssertTrue(MLX.arrayEqual(refined.weight, baseline.weight).item(Bool.self))
                        XCTAssertTrue(MLX.arrayEqual(refined.scales, baseline.scales).item(Bool.self))
                        XCTAssertTrue(MLX.arrayEqual(refined.biases, baseline.biases).item(Bool.self))
                    }
                    exported[path + ".weight"] = refined.weight
                    exported[path + ".scales"] = refined.scales
                    exported[path + ".biases"] = refined.biases
                    XCTAssertThrowsError(
                        try fitter.rescore(
                            modulePath: path, sourceWeight: matrix.asType(.float32),
                            templateWeight: baseline.weight, templateScales: baseline.scales,
                            templateBiases: baseline.biases, bits: bits, groupSize: 64))
                }
                if moe {
                    let highFit = try statistics(tokens: Array(1...8), family: "fit", threshold: 1000)
                    let highDev = try statistics(tokens: Array(9...16), family: "dev", threshold: 1000)
                    let guarded = try GemmaActivationWeightedScaleSearch(calibration: highFit, development: highDev)
                    let path = "model.layers.0.experts.switch_glu.gate_proj"
                    let matrix = try XCTUnwrap(original[path + ".weight"])
                    let baseline = q4AffineScaleSearchQuantized(matrix)
                    let retained = try guarded.rescore(
                        modulePath: path, sourceWeight: matrix,
                        templateWeight: baseline.weight, templateScales: baseline.scales,
                        templateBiases: baseline.biases, bits: 4, groupSize: 64)
                    XCTAssertEqual(retained.retainedTemplateExperts, [0, 1, 2, 3])
                    XCTAssertTrue(retained.diagnostics.isEmpty)
                    XCTAssertTrue(MLX.arrayEqual(retained.weight, baseline.weight).item(Bool.self))
                    XCTAssertTrue(MLX.arrayEqual(retained.scales, baseline.scales).item(Bool.self))
                    XCTAssertTrue(MLX.arrayEqual(retained.biases, baseline.biases).item(Bool.self))
                }
                let output = temporary.appendingPathComponent("candidate")
                try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
                var configObject = try XCTUnwrap(try JSONSerialization.jsonObject(with: config) as? [String: Any])
                configObject["quantization"] = quantization
                let outputConfig = try JSONSerialization.data(withJSONObject: configObject)
                try outputConfig.write(to: output.appendingPathComponent("config.json"))
                try MLX.save(arrays: exported, url: output.appendingPathComponent("model.safetensors"))
                let restored = Gemma4TextModel(nativeConfig)
                let base = try JSONDecoder().decode(BaseConfiguration.self, from: outputConfig)
                try loadWeights(
                    modelDirectory: output, model: restored, perLayerQuantization: base.perLayerQuantization)
                let logits = restored(MLXArray([Int32(1), 2]).reshaped(1, 2), cache: nil)
                try MLX.checkedEval(logits)
                XCTAssertEqual(logits.shape, [1, 2, 32])
                XCTAssertTrue(MLX.all(MLX.isFinite(logits)).item(Bool.self))
                for (key, value) in original where !selected.contains(String(key.dropLast(".weight".count))) {
                    XCTAssertTrue(MLX.arrayEqual(try XCTUnwrap(exported[key]), value).item(Bool.self))
                }
                XCTAssertEqual(try Data(contentsOf: weightURL), sourceBytes)
            }
        }
    }

    private func configuration(moe: Bool, nested: Bool) throws -> Data {
        let text: [String: Any] = [
            "model_type": "gemma4_text", "vocab_size": 32, "hidden_size": 128,
            "intermediate_size": 256, "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "num_global_key_value_heads": 2,
            "head_dim": 32, "global_head_dim": 32, "sliding_window": 8,
            "layer_types": ["sliding_attention", "full_attention"],
            "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0,
            "attention_k_eq_v": false, "use_double_wide_mlp": false,
            "enable_moe_block": moe, "num_experts": 4, "top_k_experts": 2,
            "moe_intermediate_size": 64, "tie_word_embeddings": true,
        ]
        return try JSONSerialization.data(
            withJSONObject: nested
                ? ["model_type": "gemma4", "vocab_size": 32, "text_config": text] : text)
    }
}

private final class CheckingExpertObserver: LagunaRoutedActivationObserver {
    struct Reference {
        var width: Int
        var sums: [Float]
        var counts: [Int]
    }
    let recorder: LagunaRoutedActivationRecorder
    var expected = [String: Reference]()

    init() throws { recorder = try LagunaRoutedActivationRecorder(minimumExpertPositions: 2) }

    func observeRoutedProjection(path: String, input: MLXArray, indices: MLXArray, expertCount: Int) {
        recorder.observeRoutedProjection(path: path, input: input, indices: indices, expertCount: expertCount)
        let width = input.dim(-1)
        let broadcast = MLX.broadcast(input, to: indices.shape + [1, width]).asType(.float32)
        let ids = indices.flattened().asType(.int32).asArray(Int32.self)
        let values = broadcast.asArray(Float.self)
        var reference =
            expected[path]
            ?? Reference(
                width: width,
                sums: Array(repeating: 0, count: expertCount * width), counts: Array(repeating: 0, count: expertCount))
        for (position, id) in ids.enumerated() {
            let expert = Int(id)
            reference.counts[expert] += 1
            for channel in 0..<width {
                let value = values[position * width + channel]
                reference.sums[expert * width + channel] += value * value
            }
        }
        expected[path] = reference
    }
}

private final class GemmaHeadMomentTap: RMSNorm {
    var output: MLXArray?

    init(norm: RMSNorm) throws {
        super.init(dimensions: norm.weight.size, eps: norm.eps)
        try update(parameters: ModuleParameters.unflattened([("weight", norm.weight)]), verify: [.all])
        train(false)
    }

    override func callAsFunction(_ input: MLXArray) -> MLXArray {
        let normalized = super.callAsFunction(input)
        output = normalized
        return normalized
    }
}
