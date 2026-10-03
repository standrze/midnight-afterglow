import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

final class BoundedConversionTests: XCTestCase {
    func testRowBatchesMatchWholeMatrixExactly() {
        for device in [Device.cpu, Device.gpu] {
            Device.withDefaultDevice(device) {
                for shape in [[131, 128], [3, 67, 128]] {
                    let count = shape.reduce(1, *)
                    let values = (0..<count).map { Float(($0 * 17) % 191 - 95) / 31 }
                    for dtype in [DType.float32, .bfloat16] {
                        let weight = MLXArray(values).reshaped(shape).asType(dtype)
                        let whole = q4AffineScaleSearchQuantized(weight)
                        let batched = q4AffineScaleSearchQuantized(weight, rowBatchSize: 64)
                        for (a, b) in [
                            (whole.weight, batched.weight), (whole.scales, batched.scales),
                            (whole.biases, batched.biases),
                        ] {
                            XCTAssertEqual(a.shape, b.shape)
                            XCTAssertEqual(a.dtype, b.dtype)
                            XCTAssertTrue(MLX.all(a .== b).item(Bool.self))
                        }
                    }
                }
            }
        }
    }

    func testMetalLargerBatchesMatchSmallBatches() {
        Device.withDefaultDevice(.gpu) {
            let count = 1025 * 5120
            let values: [Float] = (0..<count).map { Float(($0 * 17) % 191 - 95) / 31 }
            let weight = MLXArray(values).reshaped(1025, 5120).asType(.bfloat16)
            let started = Date()
            let small = q4AffineScaleSearchQuantized(weight, rowBatchSize: 64)
            eval(small.weight, small.scales, small.biases)
            let middle = Date()
            let large = q4AffineScaleSearchQuantized(weight, rowBatchSize: 512)
            eval(large.weight, large.scales, large.biases)
            print(
                "Metal batch timing: 64 rows \(middle.timeIntervalSince(started))s; 512 rows \(Date().timeIntervalSince(middle))s"
            )
            for (a, b) in [(small.weight, large.weight), (small.scales, large.scales), (small.biases, large.biases)] {
                XCTAssertTrue(MLX.all(a .== b).item(Bool.self))
            }
        }
    }

    func testCompleteCheckpointMatchesEagerConversion() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            let source = root.appendingPathComponent("source")
            try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: root) }
            let data = Data(
                #"{"model_type":"mixtral","vocab_size":128,"hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"num_local_experts":4,"num_experts_per_tok":2,"rms_norm_eps":0.00001,"rope_theta":1000000,"tie_word_embeddings":false}"#
                    .utf8)
            try data.write(to: source.appendingPathComponent("config.json"))
            let model = MixtralModel(try JSONDecoder().decode(MixtralConfiguration.self, from: data))
            try save(
                arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()),
                url: source.appendingPathComponent("model.safetensors"))
            var options = ModelConversionOptions(
                bits: 4, groupSize: 64, calibration: .q4AffineScaleSearch,
                quantizationPredicate: { path, _ in
                    if path.hasSuffix(".block_sparse_moe.gate") { return .quantize(.init(bits: 8, groupSize: 64)) }
                    if path.hasSuffix(".input_layernorm") { return .skip }
                    return .quantize()
                })
            let eager = try await LLMModelFactory.shared.convert(
                from: source, to: root.appendingPathComponent("eager"), options: options)
            options.boundedMemory = true
            let bounded = try await LLMModelFactory.shared.convert(
                from: source, to: root.appendingPathComponent("bounded"), options: options)
            var a = [String: MLXArray]()
            var b = [String: MLXArray]()
            for url in eager.weightsURLs { a.merge(try loadArrays(url: url)) { _, new in new } }
            for url in bounded.weightsURLs { b.merge(try loadArrays(url: url)) { _, new in new } }
            XCTAssertEqual(Set(a.keys), Set(b.keys))
            for (name, value) in a {
                let other = try XCTUnwrap(b[name])
                XCTAssertEqual(value.shape, other.shape, name)
                XCTAssertTrue(MLX.all(value .== other).item(Bool.self), name)
            }
            XCTAssertEqual(
                try Data(contentsOf: eager.outputDirectory.appendingPathComponent("config.json")),
                try Data(contentsOf: bounded.outputDirectory.appendingPathComponent("config.json")))
        }
    }
}
