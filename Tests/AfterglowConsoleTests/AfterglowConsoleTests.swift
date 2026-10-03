import Foundation
import Testing
import loom

@testable import AfterglowConsole

struct AfterglowConsoleTests {
    @Test func quantizationArgumentsPreservePathsAndSource() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let source = root.appendingPathComponent("source with spaces")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        var form = AfterglowForm(workflow: .quantization)
        form.fields[0].value = source.path
        form.fields[1].value = root.appendingPathComponent("result").path
        let arguments = try form.arguments()
        #expect(arguments.first == "quantize")
        #expect(arguments.suffix(2).first == source.path)
        #expect(arguments.contains("4"))
        form.fields[1].value = source.path
        #expect(throws: FormError.self) { try form.arguments() }
        form.fields[1].value = source.appendingPathComponent("nested").path
        #expect(throws: FormError.self) { try form.arguments() }
        form.fields[1].value = root.appendingPathComponent("result").path
        form.fields[2].value = "8"
        form.fields[3].value = "scale-search"
        #expect(throws: FormError.self) { try form.arguments() }
    }

    @Test func trainingLaunchAndResumeChecks() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let base = root.appendingPathComponent("base")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        var form = AfterglowForm(workflow: .training)
        form.fields[0].value = base.path
        for (index, name) in [(1, "contract.json"), (2, "train.jsonl"), (3, "dev.jsonl")] {
            let file = root.appendingPathComponent(name)
            try Data("{}".utf8).write(to: file)
            form.fields[index].value = file.path
        }
        let run = root.appendingPathComponent("run")
        form.fields[4].value = run.path
        #expect(try form.arguments().prefix(2) == ["train", "decision"])
        form.fields[8].value = "yes"
        #expect(throws: FormError.self) { try form.arguments() }
        try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
        #expect(try form.arguments().contains("--resume"))
        form.fields[8].value = "no"
        #expect(throws: FormError.self) { try form.arguments() }
    }

    @Test @MainActor func rendererHandlesResizeAndUntrustedLogText() {
        for (width, height) in [(0, 0), (1, 1), (20, 5), (100, 30)] {
            var frame = Frame(width: width, height: height)
            AfterglowTerminal.render(
                frame: &frame, form: AfterglowForm(workflow: .training), selected: 8,
                edit: nil, message: "Ready", status: "Idle", logs: ["\u{001B}[2J unsafe"])
            #expect(frame.buffer.width == width)
            #expect(frame.buffer.height == height)
            #expect(!frame.buffer.plainText.contains("\u{001B}"))
        }
    }

    @Test func ownedJobCapturesOutputAndCancels() async throws {
        let job = try AfterglowJob(executable: URL(fileURLWithPath: "/bin/echo"), arguments: ["native child output"])
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while !job.snapshot().finished, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(20)) }
        try await Task.sleep(for: .milliseconds(50))
        #expect(job.snapshot().finished)
        #expect(job.snapshot().status == "Completed")
        #expect(job.snapshot().lines.joined().contains("native child output"))
        await job.shutdown()
        let sleeper = try AfterglowJob(executable: URL(fileURLWithPath: "/bin/sleep"), arguments: ["30"])
        await sleeper.shutdown()
        #expect(sleeper.snapshot().finished)
        #expect(sleeper.snapshot().status.contains("Cancelled"))
    }
}
