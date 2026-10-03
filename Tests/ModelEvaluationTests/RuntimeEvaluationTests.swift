import CryptoKit
import Foundation
import XCTest

@testable import ModelEvaluation

final class RuntimeEvaluationTests: XCTestCase, @unchecked Sendable {
    func testStatisticsRejectDriftOrderAndFalseControlEquivalence() throws {
        let fixed = Array(repeating: 100.0, count: 8)
        let stable = try RuntimeStatistics.summarize(
            baseline: fixed, candidate: fixed, latency: false, control: true)
        XCTAssertTrue(stable.qualified)
        XCTAssertEqual(stable.interval95, [1, 1])
        let faster = Array(repeating: 120.0, count: 8)
        XCTAssertFalse(
            try RuntimeStatistics.summarize(
                baseline: fixed, candidate: faster, latency: false, control: true
            ).qualified
        )
        XCTAssertTrue(
            try RuntimeStatistics.summarize(
                baseline: fixed, candidate: faster, latency: false, control: false
            )
            .qualified)
        let drifting = [100.0, 100, 100, 100, 150, 150, 150, 150]
        XCTAssertFalse(
            try RuntimeStatistics.summarize(
                baseline: fixed, candidate: drifting, latency: false, control: false
            )
            .qualified)
        let order = [120.0, 100, 120, 100, 120, 100, 120, 100]
        XCTAssertFalse(
            try RuntimeStatistics.summarize(
                baseline: fixed, candidate: order, latency: false, control: false
            ).qualified
        )
        XCTAssertThrowsError(
            try RuntimeStatistics.summarize(
                baseline: [100], candidate: [100], latency: false, control: true))
        XCTAssertThrowsError(
            try RuntimeStatistics.summarize(
                baseline: fixed, candidate: Array(repeating: .nan, count: 8), latency: false, control: false
            ))
    }

    func testNativeValidationRejectsCachingEarlyStopsAndSettingChanges() throws {
        let workload = RuntimeEvaluationPlan.Workload(
            name: "coding", prompt: "fixture", expectedPromptTokens: 128, contextLength: 8192)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        for fault in ["cached", "short", "engine", "reuse", "warmups"] {
            var object = native(model: "/fixture", rate: 100)
            if fault == "engine" { object["engine"] = "cpu" }
            if fault == "reuse" { object["prompt_reuse_policy"] = "enabled" }
            if fault == "warmups" { object["warmup_count"] = 7 }
            if fault == "cached" || fault == "short" {
                var trials = object["trials"] as! [[String: Any]]
                var metrics = trials[0]["metrics"] as! [String: Any]
                metrics[fault == "cached" ? "cached_prompt_token_count" : "generation_token_count"] = 1
                trials[0]["metrics"] = metrics
                object["trials"] = trials
            }
            let report = try decoder.decode(
                RuntimeNativeReport.self, from: JSONSerialization.data(withJSONObject: object))
            XCTAssertThrowsError(try report.validate(model: "/fixture", workload: workload))
        }
    }

    func testResidencyValidationRejectsMissingRestorationAndSessionCapacityChanges() throws {
        let workload = RuntimeEvaluationPlan.Workload(
            name: "coding", prompt: "fixture", expectedPromptTokens: 128, contextLength: 8192)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        for scope in ["request", "session"] {
            for fault in [
                "none", "cached", "restoration", "setter", "scope", "tickets", "missing", "capacity",
            ] {
                var object = residencyNative(scope: scope)
                if fault == "scope" { object["wired_memory_scope"] = "other" }
                if fault == "missing" { object.removeValue(forKey: "wired_memory_requests") }
                if fault == "restoration" || fault == "setter" {
                    var after = object["wired_memory_after_measured"] as! [String: Any]
                    after[
                        fault == "restoration"
                            ? "last_confirmed_restored_baseline_bytes" : "last_attempt_succeeded"] =
                        fault == "restoration" ? 1 : false
                    object["wired_memory_after_measured"] = after
                }
                if fault == "tickets" || fault == "capacity" {
                    var requests = object["wired_memory_requests"] as! [[String: Any]]
                    var state = requests[8]["active_state"] as! [String: Any]
                    state[fault == "tickets" ? "active_ticket_count" : "current_limit_bytes"] = 99
                    requests[8]["active_state"] = state
                    object["wired_memory_requests"] = requests
                }
                if fault == "cached" {
                    var trials = object["trials"] as! [[String: Any]]
                    var metrics = trials[7]["metrics"] as! [String: Any]
                    metrics["cached_prompt_token_count"] = 128
                    trials[7]["metrics"] = metrics
                    object["trials"] = trials
                }
                let report = try decoder.decode(
                    RuntimeNativeReport.self, from: JSONSerialization.data(withJSONObject: object))
                if fault == "none" {
                    XCTAssertNoThrow(
                        try report.validateResidency(model: "/fixture", workload: workload, scope: scope))
                    // Eight-trial diagnostics cannot silently replace the normal one-trial comparison.
                    XCTAssertThrowsError(try report.validate(model: "/fixture", workload: workload))
                } else {
                    XCTAssertThrowsError(
                        try report.validateResidency(model: "/fixture", workload: workload, scope: scope))
                }
            }
        }
    }

    private func residencyNative(scope: String) -> [String: Any] {
        func state(limit: Int, tickets: Int, successes: Int) -> [String: Any] {
            [
                "active_baseline_bytes": tickets == 0 ? NSNull() : 0,
                "active_ticket_count": tickets, "ticket_count": tickets,
                "backend_failure_count": 0, "backend_success_count": successes, "backend_supported": true,
                "baseline_bytes": 0, "current_limit_bytes": limit, "last_attempt_succeeded": true,
                "last_attempted_limit_bytes": limit, "last_confirmed_restored_baseline_bytes": 0,
                "last_successful_backend_limit_bytes": limit,
            ]
        }
        var object = native(model: "/fixture", rate: 100)
        object["measured_trials"] = 8
        object["trials"] = Array(repeating: (object["trials"] as! [[String: Any]])[0], count: 8)
        object["wired_memory_scope"] = scope
        object["wired_memory_session_failures"] = [String]()
        object["wired_memory_requests"] = (0..<16).map { index -> [String: Any] in
            let measuredSession = scope == "session" && index >= 8
            return [
                "requested_limit_bytes": 1000, "start_returned_limit_bytes": 1000,
                "active_state": state(
                    limit: 1000, tickets: measuredSession ? 2 : 1,
                    successes: measuredSession ? 17 : index * 2 + 1),
            ]
        }
        object["wired_memory_after_measured"] = state(
            limit: 0, tickets: 0, successes: scope == "session" ? 18 : 32)
        if scope == "session" {
            object["wired_memory_session_eligible"] = true
            object["wired_memory_session"] = [
                "before": state(limit: 0, tickets: 0, successes: 16),
                "started": state(limit: 1000, tickets: 1, successes: 17),
                "ended": state(limit: 0, tickets: 0, successes: 18),
                "requested_limit_bytes": 1000, "start_returned_limit_bytes": 1000,
                "end_returned_limit_bytes": 0,
            ]
        }
        return object
    }

    func testResidencyDiagnosticRunsFrozenOrderAndRetainsEveryTrial() async throws {
        for fault in ["none", "protocol", "restoration", "waiting", "checkpoint"] {
            let (root, originalPlan) = try fixture(mode: "pass")
            defer { try? FileManager.default.removeItem(at: root) }
            let original = try JSONDecoder().decode(
                RuntimeEvaluationPlan.self, from: Data(contentsOf: originalPlan))
            let model = original.baseline
            let runtime = URL(fileURLWithPath: original.runtime.path)
            for scope in ["request", "session"] {
                var report = residencyNative(scope: scope)
                report["model_path"] = model.checkpoint.path
                if fault == "restoration" { report.removeValue(forKey: "wired_memory_after_measured") }
                try JSONSerialization.data(withJSONObject: report).write(
                    to: root.appendingPathComponent(scope + ".json"))
            }
            let script = """
                #!/bin/sh
                set -eu
                output="$2"
                scope=request
                for argument in "$@"; do
                    if [ "$argument" = '--wired-memory-session' ]; then scope=session; fi
                done
                /bin/cp '\(root.path)'/"$scope.json" "$output"
                """
            try Data(script.utf8).write(to: runtime)
            let registrationURL = root.appendingPathComponent("registration.json")
            var registration: [String: Any] = [
                "format": "afterglow-runtime-residency-diagnostic-preregistration-v1",
                "minimumFreeBytes": 100 * 1024 * 1024 * 1024,
                "source_pins": [],
                "fixed_protocol": [
                    "scope_order": ["request", "session", "session", "request"], "warmups_per_process": 8,
                    "measured_trials_per_process": 8, "generated_tokens_per_trial": 256, "engine": "metal",
                    "temperature": 0, "top_p": 1, "prefill_step_size": 512, "kv_compression": "none",
                    "prompt_reuse": "disabled", "assistant": "none", "warmup_wiring": "request",
                    "measured_session_flag": "--wired-memory-session", "adaptive_warmups": false,
                    "selective_retry": false, "outlier_removal": false,
                ],
            ]
            func object<T: Encodable>(_ value: T) throws -> Any {
                try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
            }
            registration["model"] = try object(model)
            registration["runtime"] = try object(pin(runtime))
            registration["metal"] = try object(original.metal)
            registration["workloads"] = try object(original.workloads)
            registration["timeoutSeconds"] = 60
            if fault == "protocol" {
                var fixed = registration["fixed_protocol"] as! [String: Any]
                fixed["outlier_removal"] = true
                registration["fixed_protocol"] = fixed
            }
            try JSONSerialization.data(withJSONObject: registration).write(to: registrationURL)
            let plan = RuntimeResidencyDiagnosticPlan(
                version: 1, preregistration: try pin(registrationURL),
                waitFor: [], exclusiveProcessIDs: fault == "waiting" ? [ProcessInfo.processInfo.processIdentifier] : [])
            let planURL = root.appendingPathComponent("diagnostic-plan.json")
            try JSONEncoder().encode(plan).write(to: planURL)
            let output = root.appendingPathComponent("diagnostic")
            if fault == "checkpoint" {
                try Data("changed".utf8).write(
                    to: URL(fileURLWithPath: model.checkpoint.path).appendingPathComponent("model.safetensors"))
            }
            if fault == "waiting" {
                let worker = Task {
                    try await RuntimeResidencyDiagnostic.run(
                        planURL: planURL, output: output, pollInterval: .milliseconds(10))
                }
                let progressURL = output.appendingPathComponent("progress.json")
                for _ in 0..<100 {
                    if FileManager.default.fileExists(atPath: progressURL.path) { break }
                    try await Task.sleep(for: .milliseconds(10))
                }
                let waiting = try JSONDecoder().decode(
                    RuntimeResidencyDiagnostic.Progress.self, from: Data(contentsOf: progressURL))
                XCTAssertEqual(waiting.status, "waiting_for_existing_campaigns")
                XCTAssertNil(waiting.childPID)
                worker.cancel()
                do {
                    try await worker.value
                    XCTFail("Cancelled wait completed")
                } catch {}
                XCTAssertFalse(
                    FileManager.default.fileExists(atPath: output.appendingPathComponent("w0-0-request.json").path))
            } else if fault == "none" {
                try await RuntimeResidencyDiagnostic.run(
                    planURL: planURL, output: output, pollInterval: .milliseconds(10))
                let progress = try JSONDecoder().decode(
                    RuntimeResidencyDiagnostic.Progress.self,
                    from: Data(contentsOf: output.appendingPathComponent("progress.json")))
                XCTAssertEqual(progress.status, "complete_diagnostic_only")
                XCTAssertEqual(
                    progress.observations.map(\.scope), ["request", "session", "session", "request"])
                XCTAssertTrue(
                    progress.observations.allSatisfy { $0.trials.count == 8 && $0.warmups.count == 8 })
                XCTAssertFalse(progress.performanceQualified)
                XCTAssertFalse(progress.defaultsChanged)
                do {
                    try await RuntimeResidencyDiagnostic.run(planURL: planURL, output: output)
                    XCTFail("Existing diagnostic was overwritten")
                } catch {}
            } else {
                do {
                    try await RuntimeResidencyDiagnostic.run(
                        planURL: planURL, output: output, pollInterval: .milliseconds(10))
                    XCTFail("Invalid diagnostic accepted")
                } catch {}
                if fault == "protocol" {
                    XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
                } else if fault == "checkpoint" {
                    XCTAssertFalse(
                        FileManager.default.fileExists(atPath: output.appendingPathComponent("w0-0-request.json").path))
                } else {
                    XCTAssertTrue(
                        FileManager.default.fileExists(
                            atPath: output.appendingPathComponent("w0-0-request.json").path))
                    let progress = try JSONDecoder().decode(
                        RuntimeResidencyDiagnostic.Progress.self,
                        from: Data(contentsOf: output.appendingPathComponent("progress.json")))
                    XCTAssertEqual(progress.status, "failed_diagnostic_evidence_preserved")
                    XCTAssertNil(progress.childPID)
                }
            }
        }
    }

    func testSerialFixturesRequirePreAndPostControlsAndProtectRuns() async throws {
        for mode in ["pass", "pre-drift", "post-drift"] {
            let (root, plan) = try fixture(mode: mode)
            defer { try? FileManager.default.removeItem(at: root) }
            let output = root.appendingPathComponent("run")
            let qualified = try await RuntimeEvaluation.run(
                planURL: plan, output: output, pollInterval: .milliseconds(10))
            XCTAssertEqual(qualified, mode == "pass")
            let state = try JSONDecoder().decode(
                RuntimeEvaluation.Progress.self,
                from: Data(contentsOf: output.appendingPathComponent("progress.json")))
            XCTAssertEqual(state.performanceQualified, mode == "pass")
            XCTAssertFalse(state.defaultsChanged)
            XCTAssertEqual(state.arms.count, mode == "pre-drift" ? 16 : 48)
            XCTAssertEqual(state.normalSpeedBudgetPassed, mode == "pass")
            XCTAssertNil(state.childPID)
            do {
                _ = try await RuntimeEvaluation.run(
                    planURL: plan, output: output, pollInterval: .milliseconds(10))
                XCTFail("Existing run was overwritten")
            } catch {}
        }
    }

    func testScreeningUsesFourWorkersWithoutQualificationAndRejectsWrongWarmups() async throws {
        for warmups in [1, 8] {
            let (root, planURL) = try fixture(mode: "pass")
            defer { try? FileManager.default.removeItem(at: root) }
            var plan = try JSONDecoder().decode(RuntimeEvaluationPlan.self, from: Data(contentsOf: planURL))
            plan.mode = "screening"
            try JSONEncoder().encode(plan).write(to: planURL)
            if warmups == 1 {
                for role in ["baseline", "candidate"] {
                    let url = root.appendingPathComponent(role + "-100.json")
                    var object = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
                    object["warmup_count"] = 1
                    object["warmups"] = Array((object["warmups"] as! [[String: Any]]).prefix(1))
                    try JSONSerialization.data(withJSONObject: object).write(to: url)
                }
            }
            let output = root.appendingPathComponent("screen")
            if warmups == 1 {
                let completed = try await RuntimeEvaluation.run(
                    planURL: planURL, output: output, pollInterval: .milliseconds(10))
                XCTAssertTrue(completed)
                let progress = try JSONDecoder().decode(
                    RuntimeEvaluation.Progress.self,
                    from: Data(contentsOf: output.appendingPathComponent("progress.json")))
                XCTAssertEqual(progress.status, "complete_screening_only")
                XCTAssertEqual(progress.arms.count, 4)
                XCTAssertEqual(progress.arms.map(\.role), ["baseline", "candidate", "candidate", "baseline"])
                XCTAssertEqual(progress.screeningComparisons[0].pairedDecodeRatios, [0.9, 0.9])
                XCTAssertTrue(progress.comparisons.isEmpty)
                XCTAssertFalse(progress.performanceQualified)
                XCTAssertFalse(progress.normalSpeedBudgetPassed)
                XCTAssertFalse(progress.defaultsChanged)
            } else {
                do {
                    _ = try await RuntimeEvaluation.run(
                        planURL: planURL, output: output, pollInterval: .milliseconds(10))
                    XCTFail("Screen accepted incorrect warmup count")
                } catch {}
            }
            plan.mode = "unknown"
            XCTAssertThrowsError(try plan.validate())
        }
    }

    func testCheckpointMismatchFailsBeforeLaunchingWorker() async throws {
        let (root, plan) = try fixture(mode: "pass")
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("changed".utf8).write(to: root.appendingPathComponent("candidate/model.safetensors"))
        do {
            _ = try await RuntimeEvaluation.run(planURL: plan, output: root.appendingPathComponent("run"))
            XCTFail("Changed quality-bound weights accepted")
        } catch {}
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: root.appendingPathComponent("counter").path))
    }

    func testQualityInventoryMismatchAndErrorsFailBeforeLaunchingWorker() async throws {
        for fault in ["inventory", "error"] {
            let (root, planURL) = try fixture(mode: "pass")
            defer { try? FileManager.default.removeItem(at: root) }
            var plan = try JSONDecoder().decode(
                RuntimeEvaluationPlan.self, from: Data(contentsOf: planURL))
            let reportURL = URL(fileURLWithPath: plan.candidate.qualityReport.path)
            var report = try JSONDecoder().decode(
                EvaluationReport.self, from: Data(contentsOf: reportURL))
            if fault == "inventory" { report.results[0].id = "different-case" }
            if fault == "error" {
                report.results[0].status = "error"
                report.categoryScores["coding"] = .init(passed: 0, failed: 0, errors: 1)
            }
            try JSONEncoder().encode(report).write(to: reportURL)
            plan.candidate.qualityReport = try pin(reportURL)
            try JSONEncoder().encode(plan).write(to: planURL)
            do {
                _ = try await RuntimeEvaluation.run(
                    planURL: planURL, output: root.appendingPathComponent("run"))
                XCTFail("Invalid accuracy evidence accepted")
            } catch {}
            XCTAssertFalse(
                FileManager.default.fileExists(atPath: root.appendingPathComponent("counter").path))
        }
    }

    func testCancellationStopsOwnedWorkerAndPreservesPartialEvidence() async throws {
        let (root, plan) = try fixture(mode: "sleep")
        defer { try? FileManager.default.removeItem(at: root) }
        let output = root.appendingPathComponent("run")
        let job = Task {
            try await RuntimeEvaluation.run(
                planURL: plan, output: output, pollInterval: .milliseconds(10))
        }
        let progressURL = output.appendingPathComponent("progress.json")
        var workerPID: Int32?
        for _ in 0..<200 {
            if let data = try? Data(contentsOf: progressURL),
                let progress = try? JSONDecoder().decode(RuntimeEvaluation.Progress.self, from: data),
                let pid = progress.childPID
            {
                workerPID = pid
                break
            }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertNotNil(workerPID)
        job.cancel()
        do {
            _ = try await job.value
            XCTFail("Cancelled runtime completed")
        } catch {}
        let state = try JSONDecoder().decode(
            RuntimeEvaluation.Progress.self, from: Data(contentsOf: progressURL))
        XCTAssertEqual(state.status, "failed_runtime_evidence_preserved")
        XCTAssertNil(state.childPID)
        XCTAssertFalse(state.performanceQualified)
        if let workerPID { XCTAssertNotEqual(kill(workerPID, 0), 0) }
    }

    private func fixture(mode: String) throws -> (URL, URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "afterglow-runtime-fixture-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        let encoder = JSONEncoder()
        var models: [RuntimeEvaluationPlan.Model] = []
        for role in ["baseline", "candidate"] {
            let directory = root.appendingPathComponent(role)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
            try Data("fixture-only-weights".utf8).write(
                to: directory.appendingPathComponent("model.safetensors"))
            let checkpoint = EvaluationBatchPlan.Model(
                family: "fixture", recipe: role, path: directory.path, sourceRevision: "fixture-only")
            let identity = root.appendingPathComponent(role + "-identity.json")
            try encoder.encode(EvaluationBatch.checkpoint(checkpoint)).write(to: identity)
            let qualityReport = root.appendingPathComponent(role + "-quality-report.json")
            let qualityManifest = root.appendingPathComponent(role + "-quality-manifest.json")
            let result = EvaluationCaseResult(
                id: "fixture", category: "coding", sourceFamily: "fixture", status: "passed",
                reason: "fixture",
                responses: [])
            let report = EvaluationReport(
                suiteName: "fixture", split: "development", results: [result],
                categoryScores: ["coding": .init(passed: 1, failed: 0, errors: 0)])
            try encoder.encode(report).write(to: qualityReport)
            try JSONSerialization.data(withJSONObject: [
                "revision": "fixture-only", "suiteSHA256": String(repeating: "a", count: 64),
                "maximumTokens": 2048,
            ]).write(to: qualityManifest)
            models.append(
                .init(
                    checkpoint: checkpoint, qualityIdentity: try pin(identity),
                    qualityReport: try pin(qualityReport),
                    qualityManifest: try pin(qualityManifest)))
            for rate in [100.0, 200.0] {
                try JSONSerialization.data(
                    withJSONObject: native(model: directory.path, rate: role == "candidate" ? 90 : rate)
                )
                .write(to: root.appendingPathComponent(role + "-\(Int(rate)).json"))
            }
        }
        let metal = root.appendingPathComponent("mlx.metallib")
        try Data("fixture-only-metal".utf8).write(to: metal)
        let runtime = root.appendingPathComponent("native-fixture")
        let script = """
            #!/bin/sh
            set -eu
            root='\(root.path)'
            if [ '\(mode)' = 'sleep' ]; then exec /bin/sleep 30; fi
            n=0
            if [ -f "$root/counter" ]; then n=$(/bin/cat "$root/counter"); fi
            n=$((n + 1))
            echo "$n" > "$root/counter"
            role=baseline
            if [ "$1" = "$root/candidate" ]; then role=candidate; fi
            rate=100
            if [ '\(mode)' = 'pre-drift' ] && [ "$n" -ge 9 ]; then rate=200; fi
            if [ '\(mode)' = 'post-drift' ] && [ "$n" -ge 41 ]; then rate=200; fi
            /bin/cp "$root/$role-$rate.json" "$2"
            """
        try Data(script.utf8).write(to: runtime)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: runtime.path)
        let plan = RuntimeEvaluationPlan(
            version: 1, runtime: try pin(runtime), metal: try pin(metal), baseline: models[0],
            candidate: models[1],
            workloads: [
                .init(name: "coding", prompt: "fixture", expectedPromptTokens: 128, contextLength: 8192)
            ],
            minimumFreeBytes: 100 * 1024 * 1024 * 1024, exclusiveProcessIDs: [], timeoutSeconds: 60)
        let planURL = root.appendingPathComponent("plan.json")
        try encoder.encode(plan).write(to: planURL)
        return (root, planURL)
    }

    private func pin(_ url: URL) throws -> EvaluationBatchPlan.Artifact {
        let data = try Data(contentsOf: url)
        return .init(
            path: url.path, bytes: UInt64(data.count),
            sha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
    }

    private func native(model: String, rate: Double) -> [String: Any] {
        let metrics: [String: Any] = [
            "cached_prompt_token_count": 0, "generation_token_count": 256,
            "prefilled_prompt_token_count": 128,
            "prompt_token_count": 128, "prompt_tokens_per_second": rate * 3, "tokens_per_second": rate,
            "stop_reason": "length",
        ]
        let trial: [String: Any] = [
            "content": "fixture output", "reasoning": "", "mode": "target_only",
            "prompt_token_id_fingerprint": "fixture-tokens",
            "time_to_first_token_milliseconds": 1000 / rate, "metrics": metrics,
        ]
        return [
            "status": "measured", "prompt": "fixture", "engine": "metal", "model_path": model,
            "model_implementation": "FixtureModel",
            "context_length": 8192, "prefill_step_size": 512, "kv_compression": "none",
            "prompt_reuse_policy": "disabled",
            "temperature": 0, "top_p": 1, "requested_tokens": 256, "warmup_count": 8,
            "measured_trials": 1,
            "warmups": Array(repeating: trial, count: 8), "trials": [trial],
        ]
    }
}
