import Foundation

/// Editable launch settings. Paths become individual process arguments, never shell commands.
public struct AfterglowForm: Sendable {
    public enum Workflow: String, Sendable {
        case training = "Decision training"
        case quantization = "Quantization"
    }
    public struct Field: Sendable {
        public let label: String
        public var value: String
    }
    public let workflow: Workflow
    public var fields: [Field]

    public init(workflow: Workflow) {
        self.workflow = workflow
        switch workflow {
        case .training:
            fields = [
                Field(label: "Base model folder", value: ""),
                Field(label: "Decision contract JSON", value: ""),
                Field(label: "Training JSONL", value: ""),
                Field(label: "Development JSONL", value: ""),
                Field(label: "New run folder", value: ""),
                Field(label: "Initial adapter (optional)", value: ""),
                Field(label: "Epochs", value: "1"),
                Field(label: "Maximum prompt tokens", value: "512"),
                Field(label: "Resume existing run (yes/no)", value: "no"),
            ]
        case .quantization:
            fields = [
                Field(label: "Source model folder", value: ""),
                Field(label: "New output folder", value: ""),
                Field(label: "Weight bits (4/8)", value: "4"),
                Field(label: "Calibration (standard/scale-search)", value: "standard"),
            ]
        }
    }

    public func arguments() throws -> [String] {
        func path(_ index: Int, directory: Bool, mustExist: Bool = true) throws -> String {
            let text = fields[index].value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty, !text.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) })
            else {
                throw FormError("\(fields[index].label) is required and cannot contain control characters.")
            }
            let url = URL(fileURLWithPath: (text as NSString).expandingTildeInPath).standardizedFileURL
                .resolvingSymlinksInPath()
            if mustExist {
                var isDirectory: ObjCBool = false
                guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
                    isDirectory.boolValue == directory
                else {
                    throw FormError("\(fields[index].label) must be an existing \(directory ? "folder" : "file").")
                }
            }
            return url.path
        }
        let source = try path(0, directory: true)
        switch workflow {
        case .training:
            let contract = try path(1, directory: false)
            let training = try path(2, directory: false)
            let development = try path(3, directory: false)
            let output = try path(4, directory: true, mustExist: false)
            let resumeText = fields[8].value.lowercased()
            guard ["yes", "no"].contains(resumeText) else { throw FormError("Resume must be yes or no.") }
            let resume = resumeText == "yes"
            try validateOutput(output, source: source, resume: resume)
            guard let epochs = Int(fields[6].value), (1...1000).contains(epochs),
                let length = Int(fields[7].value), (1...8192).contains(length)
            else {
                throw FormError("Epochs must be 1–1000 and maximum prompt tokens 1–8192.")
            }
            var arguments = [
                "train", "decision", "--model", source, "--contract", contract,
                "--data", training, "--development", development, "--output", output,
                "--epochs", String(epochs), "--max-length", String(length), "--batch-size", "1",
            ]
            if !fields[5].value.isEmpty { arguments += ["--adapter", try path(5, directory: true)] }
            if resume { arguments.append("--resume") }
            return arguments
        case .quantization:
            let output = try path(1, directory: true, mustExist: false)
            try validateOutput(output, source: source, resume: false)
            guard ["4", "8"].contains(fields[2].value) else { throw FormError("Weight bits must be 4 or 8.") }
            let calibration = fields[3].value
            guard ["standard", "scale-search"].contains(calibration),
                !(fields[2].value == "8" && calibration == "scale-search")
            else {
                throw FormError("Use standard calibration, or scale-search with 4-bit weights.")
            }
            return [
                "quantize", "--mode", "affine", "--bits", fields[2].value,
                "--group-size", "64", "--calibration", calibration, source, output,
            ]
        }
    }

    private func validateOutput(_ output: String, source: String, resume: Bool) throws {
        guard output != source, !output.hasPrefix(source + "/"), !source.hasPrefix(output + "/") else {
            throw FormError("Keep the output and source model in separate folders.")
        }
        let exists = FileManager.default.fileExists(atPath: output)
        guard resume ? exists : !exists else {
            throw FormError(
                resume
                    ? "Resume requires an existing run folder."
                    : "Choose a new output folder; existing outputs are preserved.")
        }
    }
}

public struct FormError: LocalizedError, Sendable {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}
