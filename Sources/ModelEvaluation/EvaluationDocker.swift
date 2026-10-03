import Foundation

/// Executes generated Python only inside a disposable, bounded, network-disabled container.
public struct EvaluationDockerSandbox: EvaluationCodeSandbox {
    public let image: String
    public let timeout: Double
    public init(image: String, timeout: Double = 15) throws {
        let parts = image.components(separatedBy: "@sha256:")
        guard parts.count == 2, !parts[0].isEmpty, parts[1].count == 64,
            parts[1].allSatisfy({ $0.isHexDigit }), timeout > 0, timeout <= 300
        else { throw EvaluationFailure.sandbox("Use a digest-pinned Docker image and a timeout in (0, 300].") }
        self.image = image
        self.timeout = timeout
    }

    public func check(code: String, tests: String) async throws -> Bool {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "afterglow-eval-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(code.utf8).write(to: directory.appendingPathComponent("solution.py"))
        try Data(tests.utf8).write(to: directory.appendingPathComponent("tests.py"))
        let name = "afterglow-eval-\(UUID().uuidString.lowercased())"
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [
            "docker", "run", "--pull=never", "--rm", "--name", name,
            "--label", "afterglow.evaluation.owner=\(ProcessInfo.processInfo.processIdentifier)", "--network=none",
            "--read-only",
            "--memory=256m", "--memory-swap=256m", "--cpus=1", "--pids-limit=64", "--cap-drop=ALL",
            "--security-opt=no-new-privileges", "--user=65534:65534", "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
            "--mount", "type=bind,src=\(directory.path),dst=/work,readonly", "--workdir=/work",
            image, "python", "-B", "tests.py",
        ]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        let deadline = Date().addingTimeInterval(timeout)
        var timedOut = false
        while process.isRunning {
            if Date() >= deadline || Task.isCancelled {
                timedOut = true
                break
            }
            try? await Task.sleep(for: .milliseconds(50))
        }
        if timedOut {
            let cleanup = Process()
            cleanup.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            cleanup.arguments = ["docker", "rm", "--force", name]
            cleanup.standardOutput = FileHandle.nullDevice
            cleanup.standardError = FileHandle.nullDevice
            try cleanup.run()
            cleanup.waitUntilExit()
            if process.isRunning { process.terminate() }
            process.waitUntilExit()
            if Task.isCancelled { throw CancellationError() }
            return false
        }
        if [125, 126, 127].contains(process.terminationStatus) {
            throw EvaluationFailure.sandbox("Docker failed to launch the pinned Python sandbox.")
        }
        return process.terminationStatus == 0
    }
}
