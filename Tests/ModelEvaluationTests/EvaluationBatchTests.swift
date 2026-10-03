import CryptoKit
import Foundation
import XCTest

@testable import ModelEvaluation

final class EvaluationBatchTests: XCTestCase, @unchecked Sendable {
    private func fixture(wait: Bool) throws -> (URL, URL, [String: Any]) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "afterglow-batch-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        let suite = root.appendingPathComponent("suite.json")
        try Data(
            #"{"version":1,"name":"test","split":"development","cases":[{"id":"one","category":"tools","sourceFamily":"test","prompt":"Return JSON true","tools":[],"turns":[{"calls":[],"answer":true}]}]}"#
                .utf8
        ).write(to: suite)
        let weights = root.appendingPathComponent("model")
        try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: false)
        try Data([1, 2, 3]).write(to: weights.appendingPathComponent("fixture.safetensors"))
        func reference(_ url: URL) throws -> [String: Any] {
            let data = try Data(contentsOf: url)
            return [
                "path": url.path, "bytes": data.count,
                "sha256": SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(),
            ]
        }
        let harmless = try reference(URL(fileURLWithPath: "/usr/bin/true"))
        var pins = Dictionary(
            uniqueKeysWithValues: ["evaluator", "evaluator_metal", "server", "server_metal", "config", "probe"].map {
                ($0, harmless)
            })
        pins["suite"] = try reference(suite)
        var predecessors: [[String: Any]] = []
        if wait {
            let progress = root.appendingPathComponent("prior.json")
            try JSONSerialization.data(withJSONObject: [
                "status": "running_fixture", "pid": ProcessInfo.processInfo.processIdentifier,
            ]).write(to: progress)
            predecessors = [
                [
                    "progress": progress.path, "pid": ProcessInfo.processInfo.processIdentifier, "plan": harmless,
                    "controller": harmless, "terminal": ["complete_fixture"],
                ]
            ]
        }
        let plan: [String: Any] = [
            "version": 1,
            "models": [
                [
                    "family": "fixture", "recipe": "fixture-incumbent-ls2-q4", "path": weights.path,
                    "source_revision": "test-only",
                ]
            ], "wait_for": predecessors, "minimum_free_bytes": 100 * 1024 * 1024 * 1024,
            "python_image": "python@sha256:" + String(repeating: "a", count: 64), "pins": pins,
        ]
        return (root, root.appendingPathComponent("plan.json"), plan)
    }

    func testRejectsTraversalDuplicateRecipesAndLowReserve() throws {
        let (root, _, original) = try fixture(wait: false)
        defer { try? FileManager.default.removeItem(at: root) }
        for kind in ["traversal", "duplicate", "reserve"] {
            var data = original
            var models = data["models"] as! [[String: String]]
            if kind == "traversal" { models[0]["recipe"] = "../outside" }
            if kind == "duplicate" { models.append(models[0]) }
            if kind == "reserve" { data["minimum_free_bytes"] = 1 }
            data["models"] = models
            let plan = try JSONDecoder().decode(
                EvaluationBatchPlan.self, from: JSONSerialization.data(withJSONObject: data))
            XCTAssertThrowsError(try plan.validate())
        }
    }

    func testCancellationPreservesEvidenceBeforeAnyModelStarts() async throws {
        let (root, planURL, data) = try fixture(wait: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try JSONSerialization.data(withJSONObject: data).write(to: planURL)
        let output = root.appendingPathComponent("run")
        let job = Task { try await EvaluationBatch.run(planURL: planURL, output: output) }
        let progress = output.appendingPathComponent("progress.json")
        for _ in 0..<100 {
            if FileManager.default.fileExists(atPath: progress.path) { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        job.cancel()
        do {
            try await job.value
            XCTFail("Cancelled batch completed")
        } catch {}
        let state = try JSONSerialization.jsonObject(with: Data(contentsOf: progress)) as! [String: Any]
        XCTAssertEqual(state["status"] as? String, "failed_model_suite_evidence_preserved")
        XCTAssertEqual((state["completed"] as? [Any])?.count, 0)
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: output.appendingPathComponent("fixture-incumbent-ls2-q4").path))
    }

    func testOwnedServerFailureCannotProduceSuccessfulResults() async throws {
        let (root, planURL, data) = try fixture(wait: false)
        defer { try? FileManager.default.removeItem(at: root) }
        try JSONSerialization.data(withJSONObject: data).write(to: planURL)
        let output = root.appendingPathComponent("run")
        do {
            try await EvaluationBatch.run(planURL: planURL, output: output)
            XCTFail("Exited server passed")
        } catch {}
        let state =
            try JSONSerialization.jsonObject(with: Data(contentsOf: output.appendingPathComponent("progress.json")))
            as! [String: Any]
        XCTAssertEqual(state["status"] as? String, "failed_model_suite_evidence_preserved")
        XCTAssertEqual((state["completed"] as? [Any])?.count, 0)
        XCTAssertNil(state["server_pid"])
    }
}
