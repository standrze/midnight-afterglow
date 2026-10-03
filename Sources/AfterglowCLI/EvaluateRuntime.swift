import ArgumentParser
import Foundation
import ModelEvaluation

#if canImport(Darwin)
    import Darwin
#endif

struct EvaluateRuntime: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "runtime",
        abstract: "Run pinned native speed screening or comparisons with bracketing same-model controls.")
    @Option(help: "Versioned native runtime plan; all artifacts must be locally available and pinned.") var plan: String
    @Option(help: "New output directory; incomplete and previous evidence are never overwritten.") var output: String

    mutating func run() async throws {
        let planURL = URL(fileURLWithPath: plan)
        let directory = URL(fileURLWithPath: output)
        let worker = Task { try await RuntimeEvaluation.run(planURL: planURL, output: directory) }
        let priorInterrupt = signal(SIGINT, SIG_IGN)
        let priorTerminate = signal(SIGTERM, SIG_IGN)
        let interrupt = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
        let terminate = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .global())
        interrupt.setEventHandler { worker.cancel() }
        terminate.setEventHandler { worker.cancel() }
        interrupt.resume()
        terminate.resume()
        defer {
            interrupt.cancel()
            terminate.cancel()
            signal(SIGINT, priorInterrupt)
            signal(SIGTERM, priorTerminate)
        }
        let completed = try await worker.value
        print("Review \(directory.appendingPathComponent("progress.json").path)")
        if !completed { throw ExitCode.failure }
    }
}
