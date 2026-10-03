import Foundation

#if canImport(Darwin)
    import Darwin
#else
    import Glibc
#endif

/// Owns one native CLI child, retaining a bounded log and never adopting another process.
final class AfterglowJob: @unchecked Sendable {
    private let lock = NSLock()
    private let process = Process()
    private let pipe = Pipe()
    private var bytes = Data()
    private var code: Int32?
    private var cancelling = false
    private var cancellationTime: TimeInterval?
    let started = ProcessInfo.processInfo.systemUptime

    init(executable: URL, arguments: [String]) throws {
        process.executableURL = executable
        process.arguments = arguments
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = pipe
        process.standardError = pipe
        var environment = ProcessInfo.processInfo.environment
        environment["NSUnbufferedIO"] = "YES"
        environment["TERM"] = "dumb"
        process.environment = environment
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard let self, !data.isEmpty else { return }
            self.append(data)
        }
        process.terminationHandler = { [weak self] process in
            guard let self else { return }
            self.lock.withLock { self.code = process.terminationStatus }
        }
        do { try process.run() } catch {
            pipe.fileHandleForReading.readabilityHandler = nil
            throw error
        }
    }

    private func append(_ data: Data) {
        lock.withLock {
            bytes.append(data)
            if bytes.count > 32 * 1024 { bytes.removeFirst(bytes.count - 32 * 1024) }
        }
    }

    func snapshot() -> (lines: [String], status: String, finished: Bool) {
        let expired = lock.withLock { cancellationTime.map { ProcessInfo.processInfo.systemUptime - $0 >= 5 } ?? false }
        if expired, process.isRunning { _ = kill(process.processIdentifier, SIGKILL) }
        return lock.withLock {
            let elapsed = Int(ProcessInfo.processInfo.systemUptime - started)
            let status =
                code.map { cancelling ? "Cancelled (exit \($0))" : $0 == 0 ? "Completed" : "Failed (exit \($0))" }
                ?? "\(cancelling ? "Cancelling" : "Running") · \(elapsed)s"
            return (String(decoding: bytes, as: UTF8.self).components(separatedBy: .newlines), status, code != nil)
        }
    }

    func cancel() {
        let shouldSignal = lock.withLock {
            guard code == nil, !cancelling else { return false }
            cancelling = true
            cancellationTime = ProcessInfo.processInfo.systemUptime
            return true
        }
        if shouldSignal, process.isRunning { process.terminate() }
    }

    func shutdown() async {
        cancel()
        while process.isRunning {
            let expired = lock.withLock {
                cancellationTime.map { ProcessInfo.processInfo.systemUptime - $0 >= 5 } ?? false
            }
            if expired { _ = kill(process.processIdentifier, SIGKILL) }
            try? await Task.sleep(for: .milliseconds(50))
        }
        lock.withLock { code = process.terminationStatus }
        pipe.fileHandleForReading.readabilityHandler = nil
    }
}
