import Foundation
import XCTest

final class CustomQuantizationGeometryTests: XCTestCase {
    func testCustomModuleWidthIsValidatedAndCanBeExplicitlySkipped() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let executable = environment["WICK_TEST_EXECUTABLE"] ?? environment["FACET_TEST_EXECUTABLE"],
            !executable.isEmpty
        else {
            throw XCTSkip("Set WICK_TEST_EXECUTABLE to the built wick executable for CLI geometry tests.")
        }

        let manager = FileManager.default
        let root = manager.temporaryDirectory.appendingPathComponent("wick-custom-geometry-\(UUID())")
        let source = root.appendingPathComponent("source")
        let rejectedOutput = root.appendingPathComponent("rejected")
        let skippedOutput = root.appendingPathComponent("skipped")
        try manager.createDirectory(at: source, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: root) }

        // GLM4MoELite uses a custom Quantizable MultiLinear for embed_q.
        // Its input width is 24; every other quantizable input is divisible
        // by 32. Planning must inspect this custom module before conversion.
        let config = """
            {
                "model_type": "glm4_moe_lite",
                "vocab_size": 128,
                "hidden_size": 64,
                "intermediate_size": 128,
                "moe_intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "routed_scaling_factor": 1,
                "kv_lora_rank": 64,
                "qk_rope_head_dim": 8,
                "qk_nope_head_dim": 24,
                "v_head_dim": 32,
                "norm_topk_prob": true,
                "n_group": 1,
                "topk_group": 1,
                "num_experts_per_tok": 1,
                "first_k_dense_replace": 1,
                "max_position_embeddings": 128,
                "rms_norm_eps": 0.00001,
                "rope_theta": 10000,
                "attention_bias": false,
                "partial_rotary_factor": 1
            }
            """
        try Data(config.utf8).write(to: source.appendingPathComponent("config.json"))

        let path = "model.layers.0.self_attn.embed_q"
        let rejected = try runCLI(
            executable: executable,
            arguments: ["quantize", source.path, rejectedOutput.path, "--mode", "mxfp4", "--dry-run"]
        )
        XCTAssertEqual(rejected.reason, .exit, rejected.output)
        XCTAssertNotEqual(rejected.status, 0, rejected.output)
        XCTAssertTrue(rejected.output.contains(path), rejected.output)
        XCTAssertTrue(rejected.output.contains("input width 24"), rejected.output)
        XCTAssertTrue(rejected.output.contains("group-32"), rejected.output)
        XCTAssertFalse(manager.fileExists(atPath: rejectedOutput.path))

        let skipped = try runCLI(
            executable: executable,
            arguments: [
                "quantize", source.path, skippedOutput.path, "--mode", "mxfp4", "--dry-run",
                "--skip-module", path,
            ]
        )
        XCTAssertEqual(skipped.reason, .exit, skipped.output)
        XCTAssertEqual(skipped.status, 0, skipped.output)
        XCTAssertTrue(skipped.output.contains("skipped modules: 1"), skipped.output)
        XCTAssertTrue(skipped.output.contains(path), skipped.output)
        XCTAssertFalse(manager.fileExists(atPath: skippedOutput.path))
        XCTAssertEqual(try manager.contentsOfDirectory(atPath: root.path), ["source"])
    }

    private func runCLI(executable: String, arguments: [String]) throws -> (
        status: Int32, reason: Process.TerminationReason, output: String
    ) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (process.terminationStatus, process.terminationReason, String(decoding: data, as: UTF8.self))
    }
}
