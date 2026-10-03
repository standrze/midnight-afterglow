import Foundation
import QuantizerSupport
import XCTest

final class GemmaPrecisionPolicyTests: XCTestCase {
    private let source = Data(#"{"model_type":"gemma4_text","hidden_size":128}"#.utf8)
    private let body = "model.layers.0.mlp.down_proj"
    private let router = "model.layers.0.router.proj"

    func testAffinePrecisionAndGroupingValidatePhysicalPacking() throws {
        for bits in [4, 5, 6, 8] {
            for groupSize in [32, 64] {
                let policy = try GemmaGroupSizePolicy(
                    sourceConfiguration: source, modules: modules, selectedModules: [],
                    q8Modules: [router], defaultBits: bits, defaultGroupSize: groupSize)
                let geometry: [String: Any] = ["bits": bits, "group_size": groupSize, "mode": "affine"]
                var quantization = geometry
                quantization[router] = ["bits": 8, "group_size": 64, "mode": "affine"]
                let config = try configuration(quantization)
                XCTAssertNoThrow(try policy.validateOutputConfiguration(config))
                let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                defer { try? FileManager.default.removeItem(at: directory) }
                let correct = try fixture(directory, name: "correct", bits: bits, groupSize: groupSize)
                XCTAssertNoThrow(try policy.validateOutputWeights([correct]))
                if bits == 5 || bits == 6 {
                    let truncated = try fixture(
                        directory, name: "truncated", bits: bits, groupSize: groupSize, truncatedPacking: true)
                    XCTAssertThrowsError(try policy.validateOutputWeights([truncated]))
                }
                quantization["bits"] = bits == 4 ? 5 : 4
                XCTAssertThrowsError(try policy.validateOutputConfiguration(configuration(quantization)))
            }
        }
    }

    func testBaseGeometryFailsClosedAndSelectiveGroupsRetainTheirPrecision() throws {
        for (bits, groupSize) in [(3, 64), (2, 32), (7, 64), (5, 128), (4, 16)] {
            XCTAssertThrowsError(
                try GemmaGroupSizePolicy(
                    sourceConfiguration: source, modules: modules, selectedModules: [],
                    defaultBits: bits, defaultGroupSize: groupSize))
        }
        let policy = try GemmaGroupSizePolicy(
            sourceConfiguration: source, modules: modules, selectedModules: [body],
            q8Modules: [router], defaultBits: 5, defaultGroupSize: 32)
        let quantization: [String: Any] = [
            "bits": 5, "group_size": 32, "mode": "affine",
            body: ["bits": 5, "group_size": 128, "mode": "affine"],
            router: ["bits": 8, "group_size": 64, "mode": "affine"],
        ]
        XCTAssertNoThrow(try policy.validateOutputConfiguration(configuration(quantization)))
        var changed = quantization
        changed[body] = ["bits": 4, "group_size": 128, "mode": "affine"]
        XCTAssertThrowsError(try policy.validateOutputConfiguration(configuration(changed)))
    }

    private var modules: [String: GemmaGroupSizePolicy.Module] {
        [body: .init(shape: [2, 128], kind: .linear), router: .init(shape: [2, 128], kind: .linear)]
    }

    private func configuration(_ quantization: [String: Any]) throws -> Data {
        var result = try XCTUnwrap(JSONSerialization.jsonObject(with: source) as? [String: Any])
        result["quantization"] = quantization
        result["quantization_config"] = quantization
        return try JSONSerialization.data(withJSONObject: result)
    }

    private func fixture(
        _ directory: URL, name: String, bits: Int, groupSize: Int, truncatedPacking: Bool = false
    ) throws -> URL {
        var header = [String: Any]()
        var payload = Data()
        for path in [body, router] {
            let precision = path == router ? 8 : bits
            let group = path == router ? 64 : groupSize
            let packedWidth = truncatedPacking && path == body ? 128 / (32 / precision) : 128 / 32 * precision
            for (suffix, dtype, width, elementBytes) in [
                ("weight", "U32", packedWidth, 4), ("scales", "BF16", 128 / group, 2),
                ("biases", "BF16", 128 / group, 2),
            ] {
                let begin = payload.count
                payload.append(Data(repeating: 0, count: 2 * width * elementBytes))
                header[path + "." + suffix] = [
                    "dtype": dtype, "shape": [2, width], "data_offsets": [begin, payload.count],
                ]
            }
        }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while json.count % 8 != 0 { json.append(32) }
        var length = UInt64(json.count).littleEndian
        var data = withUnsafeBytes(of: &length) { Data($0) }
        data.append(json)
        data.append(payload)
        let url = directory.appendingPathComponent(name + ".safetensors")
        try data.write(to: url)
        return url
    }
}
