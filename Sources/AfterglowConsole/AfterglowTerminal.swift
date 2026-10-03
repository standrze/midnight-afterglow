import Foundation
import loom
import weft

#if canImport(Darwin)
    import Darwin
#else
    import Glibc
#endif

/// A small form and log viewer. weft owns terminal modes; this main actor owns UI state.
@MainActor
public final class AfterglowTerminal {
    private var terminal: Terminal<ANSIBackend>?
    private var forms = [AfterglowForm(workflow: .training), AfterglowForm(workflow: .quantization)]
    private var workflow = 0
    private var field = 0
    private var edit: String?
    private var message = "Set paths and options, then press r to run."
    private var job: AfterglowJob?

    public nonisolated static var isInteractive: Bool {
        isatty(STDIN_FILENO) == 1 && isatty(STDOUT_FILENO) == 1
            && ProcessInfo.processInfo.environment["TERM"] != "dumb"
    }

    public init() {}

    public func run() async throws {
        guard Self.isInteractive else { throw FormError("The interface requires an interactive terminal.") }
        let session = try TerminalSession(options: TerminalOptions(alternateScreen: true, hideCursor: true))
        defer {
            terminal = nil
            session.close()
        }
        let events = TerminalEvents(session: session)
        defer { events.stop() }
        let size = TerminalSession.size()
        terminal = Terminal(backend: ANSIBackend(), width: size.width, height: size.height)
        try draw()
        do {
            try await withThrowingTaskGroup(of: Void.self) { group in
                group.addTask { try await self.consume(events) }
                group.addTask {
                    while !Task.isCancelled {
                        try await Task.sleep(for: .milliseconds(200))
                        try await self.draw()
                    }
                }
                defer { group.cancelAll() }
                _ = try await group.next()
            }
        } catch {
            await job?.shutdown()
            throw error
        }
        await job?.shutdown()
    }

    private func consume(_ events: TerminalEvents) async throws {
        for try await batch in events.events {
            try Task.checkCancellation()
            for event in batch {
                if case .signal = event { return }
                if case .paste(let text) = event, edit != nil {
                    append(text)
                    continue
                }
                if case .text(let text) = event, text.count > 1, edit != nil {
                    append(text)
                    continue
                }
                guard let keyboard = event.keyboardEvent, keyboard.kind != .release else { continue }
                if keyboard.key == .interrupt || keyboard.key == .endOfInput { return }
                if edit != nil {
                    switch keyboard.key {
                    case .escape:
                        edit = nil
                        message = "Edit cancelled."
                    case .enter:
                        forms[workflow].fields[field].value = edit!
                        edit = nil
                        message = "Updated \(forms[workflow].fields[field].label)."
                    case .backspace: if !edit!.isEmpty { edit!.removeLast() }
                    case .character(let character): append(String(character))
                    default: break
                    }
                    continue
                }
                switch keyboard.key {
                case .character("q"): return
                case .character("c"):
                    job?.cancel()
                    message = "Cancelling the owned job; completed checkpoints are preserved."
                case .character("1"), .character("2"):
                    if job?.snapshot().finished != false {
                        workflow = keyboard.key == .character("1") ? 0 : 1
                        field = 0
                    }
                case .up: field = max(0, field - 1)
                case .down: field = min(forms[workflow].fields.count - 1, field + 1)
                case .tab: field = (field + 1) % forms[workflow].fields.count
                case .enter:
                    if job?.snapshot().finished != false { edit = forms[workflow].fields[field].value }
                case .character("r"): start()
                default: break
                }
            }
            try draw()
        }
    }

    private func append(_ text: String) {
        guard !text.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }),
            (edit?.utf8.count ?? 0) + text.utf8.count <= 4096
        else {
            message = "Enter one line of text, at most 4096 bytes."
            return
        }
        edit?.append(text)
    }

    private func start() {
        guard job?.snapshot().finished != false else {
            message = "One job at a time. Press c to cancel."
            return
        }
        do {
            let arguments = try forms[workflow].arguments()
            let executable = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
                .resolvingSymlinksInPath()
            job = try AfterglowJob(executable: executable, arguments: arguments)
            message = "Native \(forms[workflow].workflow.rawValue.lowercased()) started."
        } catch { message = error.localizedDescription }
    }

    private func draw() throws {
        let size = TerminalSession.size()
        terminal?.resize(width: size.width, height: size.height)
        let snapshot = job?.snapshot()
        try terminal?.draw { frame in
            Self.render(
                frame: &frame, form: forms[workflow], selected: field, edit: edit,
                message: message, status: snapshot?.status ?? "Idle", logs: snapshot?.lines ?? [])
        }
    }

    /// Renders bounded rows in terminal cells, replacing control characters before drawing.
    public static func render(
        frame: inout Frame, form: AfterglowForm, selected: Int, edit: String?,
        message: String, status: String, logs: [String]
    ) {
        let width = max(1, frame.area.width - 2)
        func row(_ text: String, at y: Int, bold: Bool = false) {
            guard y >= 0, y < frame.area.height else { return }
            let safe = text.unicodeScalars.map { CharacterSet.controlCharacters.contains($0) ? " " : String($0) }
                .joined()
            frame.render(
                Paragraph(TextLayout(safe, width: width), style: Style(bold: bold)),
                in: Rect(x: 1, y: y, width: width, height: 1))
        }
        row("midnight afterglow", at: 0, bold: true)
        row("1 Training   2 Quantization   ·   \(status)", at: 1)
        let capacity = max(1, frame.area.height - 8)
        let first = max(0, selected - capacity + 1)
        let fields = form.fields.dropFirst(first).prefix(capacity)
        for (index, item) in fields.enumerated() {
            let selectedRow = first + index == selected
            let value = selectedRow ? edit ?? item.value : item.value
            row(
                "\(selectedRow ? "›" : " ") \(item.label): \(value.isEmpty ? "(not set)" : value)\(selectedRow && edit != nil ? " ▏" : "")",
                at: index + 3, bold: selectedRow)
        }
        let logStart = 4 + fields.count
        let logCount = max(0, frame.area.height - logStart - 3)
        for (index, line) in logs.filter({ !$0.isEmpty }).suffix(logCount).enumerated() {
            row(line, at: logStart + index)
        }
        row(message, at: frame.area.height - 2)
        row(
            edit == nil
                ? "↑↓/Tab select · Enter edit · r run · c cancel · q quit (cancels job)"
                : "Type/paste a value · Enter save · Esc cancel · Ctrl-C quit", at: frame.area.height - 1)
    }
}
