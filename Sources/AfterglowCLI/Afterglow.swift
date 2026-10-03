import AfterglowConsole
import ArgumentParser
import DecisionMLX
import DecisionModels
import Foundation
import QuantizationCommands

@main
struct Afterglow: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "midnight-afterglow", abstract: "Train, calibrate and prepare models for Midnight.",
        version: "0.1.0",
        subcommands: [
            Inspect.self, Validate.self, Train.self, Evaluate.self, Calibrate.self,
            Convert.self, Export.self, ModelQuantizer.self, QuantizationFormats.self, Interface.self,
        ], defaultSubcommand: Interface.self)
}

struct ModelOptions: ParsableArguments {
    @Option(help: "Local base checkpoint directory.") var model: String
    @Option(help: "Native decision contract JSON.") var contract: String
    @Option(help: "Optional local adapter directory.") var adapter: String?

    var modelURL: URL { URL(fileURLWithPath: model) }
    var adapterURL: URL? { adapter.map { URL(fileURLWithPath: $0) } }
    func readContract() throws -> DecisionModelContract {
        let result = try JSONDecoder().decode(
            DecisionModelContract.self, from: Data(contentsOf: URL(fileURLWithPath: contract)))
        try result.validate()
        return result
    }
}

struct Inspect: ParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Inspect a decision contract and local checkpoint.")
    @OptionGroup var options: ModelOptions
    mutating func run() throws {
        let contract = try options.readContract()
        print("Base: \(contract.model) @ \(contract.revision)")
        print(
            "Task: \(contract.task); choices: \(contract.candidateCodes.count); context: \(contract.maxLength); T=\(contract.temperature)"
        )
        print("Local fingerprint: \(try DecisionFiles.modelHash(options.modelURL))")
    }
}

struct Validate: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Validate labeled datasets and family separation without loading weights.")
    @Option var contract: String
    @Argument var datasets: [String]
    mutating func run() throws {
        guard !datasets.isEmpty else { throw ValidationError("Supply at least one JSONL dataset.") }
        let contract = try JSONDecoder().decode(
            DecisionModelContract.self, from: Data(contentsOf: URL(fileURLWithPath: contract)))
        try contract.validate()
        let splits = try datasets.map { try DecisionDataset.read(URL(fileURLWithPath: $0), contract: contract) }
        try DecisionDataset.validateSplits(splits)
        print("Validated \(splits.map(\.count)) examples; no ID or source-family overlap.")
    }
}

struct Train: AsyncParsableCommand {
    static let configuration = CommandConfiguration(subcommands: [TrainDecision.self])
}

struct TrainDecision: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "decision", abstract: "Native LoRA training over candidate logits.")
    @OptionGroup var options: ModelOptions
    @Option var data: String
    @Option var development: String
    @Option var output: String
    @Option var rank = 16
    @Option var learningRate: Float = 5e-5
    @Option var batchSize = 8
    @Option var epochs = 1
    @Option var seed: UInt64 = 17
    @Option var maxLength = 2048
    @Option var checkpointEvery = 10
    @Option(help: "Stop at this total update count to exercise resume.") var stopAfterUpdates: Int?
    @Flag var resume = false
    mutating func run() async throws {
        var training = DecisionTrainingOptions()
        training.rank = rank
        training.learningRate = learningRate
        training.batchSize = batchSize
        training.epochs = epochs
        training.seed = seed
        training.maxLength = maxLength
        training.checkpointEvery = checkpointEvery
        training.initializationAdapter = options.adapter
        let container = try await DecisionRuntime.load(model: options.modelURL)
        let report = try await DecisionTrainer.train(
            container: container, model: options.modelURL,
            data: URL(fileURLWithPath: data), development: URL(fileURLWithPath: development),
            contract: options.readContract(), options: training, output: URL(fileURLWithPath: output),
            resume: resume, stopAfterUpdates: stopAfterUpdates)
        print("Completed \(report.updates) updates; development loss \(report.initialLoss) → \(report.finalLoss).")
    }
}

struct Evaluate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(subcommands: [
        EvaluateDecision.self, EvaluateModel.self, EvaluateBatch.self, EvaluateRuntime.self,
        EvaluateRuntimeDiagnostic.self,
    ])
}

struct EvaluationReport: Codable {
    var count: Int
    var accuracy: Double
    var nll: Double
    var brier: Double
    var temperature: Double
    var predictions: [DecisionResponse]
}

struct EvaluateDecision: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "decision", abstract: "Score labeled decision data and report accuracy, NLL and Brier score.")
    @OptionGroup var options: ModelOptions
    @Option var data: String
    @Option var output: String
    mutating func run() async throws {
        let contract = try options.readContract()
        let examples = try DecisionDataset.read(URL(fileURLWithPath: data), contract: contract)
        try DecisionDataset.validateSplits([examples])
        let container = try await DecisionRuntime.load(model: options.modelURL, adapter: options.adapterURL)
        var metrics = DecisionMetrics()
        var predictions: [DecisionResponse] = []
        for example in examples {
            let result = try await container.perform { context in
                try DecisionRuntime.score(context: context, request: example.request, contract: contract)
            }
            for (name, field) in result.fields { try metrics.record(field, gold: example.labels[name]!) }
            predictions.append(result)
        }
        let count = Double(metrics.count)
        let report = EvaluationReport(
            count: metrics.count, accuracy: metrics.accuracy, nll: metrics.nll / count,
            brier: metrics.brier / count, temperature: contract.temperature, predictions: predictions)
        try DecisionFiles.write(report, to: URL(fileURLWithPath: output))
        print("Accuracy \(report.accuracy), NLL \(report.nll), Brier \(report.brier)")
    }
}

struct Calibrate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(subcommands: [CalibrateDecision.self])
}

struct CalibrateDecision: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "decision", abstract: "Fit temperature on a dedicated labeled calibration split.")
    @OptionGroup var options: ModelOptions
    @Option var data: String
    @Option(help: "Train/development/test splits to check for calibration overlap.") var exclude: [String] = []
    @Option var output: String
    mutating func run() async throws {
        guard !exclude.isEmpty else {
            throw ValidationError("Supply --exclude splits to verify calibration separation.")
        }
        var contract = try options.readContract()
        let examples = try DecisionDataset.read(URL(fileURLWithPath: data), contract: contract)
        let other = try exclude.map { try DecisionDataset.read(URL(fileURLWithPath: $0), contract: contract) }
        try DecisionDataset.validateSplits([examples] + other)
        let container = try await DecisionRuntime.load(model: options.modelURL, adapter: options.adapterURL)
        var records: [(logits: [Double], gold: Int)] = []
        for example in examples {
            let fixed = contract
            let result = try await container.perform { context in
                try DecisionRuntime.score(context: context, request: example.request, contract: fixed)
            }
            for name in example.request.names {
                let values = example.request.schema[name]!.values
                records.append(
                    (
                        values.map { result.fields[name]!.logits[$0.key]! },
                        values.firstIndex { $0.key == example.labels[name]!.key }!
                    ))
            }
        }
        func objective(_ logTemperature: Double) -> Double {
            let temperature = exp(logTemperature)
            return records.reduce(0) { sum, record in
                let maximum = record.logits.max()!
                let denominator = record.logits.reduce(0) { $0 + exp(($1 - maximum) / temperature) }
                return sum + log(denominator) - (record.logits[record.gold] - maximum) / temperature
            } / Double(records.count)
        }
        var low = log(0.05)
        var high = log(20.0)
        for _ in 0..<80 {
            let left = low + (high - low) / 3
            let right = high - (high - low) / 3
            if objective(left) < objective(right) { high = right } else { low = left }
        }
        contract.temperature = exp((low + high) / 2)
        try DecisionFiles.write(contract, to: URL(fileURLWithPath: output))
        print("Fitted T=\(contract.temperature); calibration NLL \(objective(log(contract.temperature))).")
    }
}

struct Convert: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Convert an official PEFT decision adapter to native MLX layout.")
    @Option var source: String
    @Option var output: String
    @Option(help: "Immutable publisher adapter revision.") var revision: String
    mutating func run() throws {
        let source = URL(fileURLWithPath: source)
        guard revision.count == 40, revision.allSatisfy(\.isHexDigit) else {
            throw ValidationError("Revision must be an immutable 40-character commit.")
        }
        struct Temperature: Decodable {
            let temperature: Double
            let adapterSHA256: String

            enum CodingKeys: String, CodingKey {
                case temperature
                case adapterSHA256 = "adapter_sha256"
            }
        }
        let temperature = try JSONDecoder().decode(
            Temperature.self, from: Data(contentsOf: source.appendingPathComponent("temperature_config.json")))
        let hash = try DecisionFiles.hash(source.appendingPathComponent("adapter_model.safetensors"))
        guard hash == temperature.adapterSHA256 else {
            throw ValidationError("Publisher adapter hash differs from its temperature contract.")
        }
        var contract = try DecisionModelContract.importNimble(
            schema: Data(contentsOf: source.appendingPathComponent("schema_config.json")),
            temperature: temperature.temperature)
        contract.adapterSHA256 = hash
        contract.sourceRevision = revision
        try DecisionRuntime.convertAdapter(
            source: source, destination: URL(fileURLWithPath: output), contract: contract)
        print("Converted adapter and decision contract to \(output).")
    }
}

struct Export: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Export a self-contained base-plus-adapter bundle after reload verification.")
    @OptionGroup var options: ModelOptions
    @Option(help: "Decision request JSON used to verify the export.") var probe: String
    @Option var output: String
    mutating func run() async throws {
        guard let adapter = options.adapterURL else { throw ValidationError("Export requires --adapter.") }
        var contract = try options.readContract()
        contract.baseFingerprint = try DecisionFiles.modelHash(options.modelURL)
        contract.adapterFingerprint = try DecisionFiles.hash(adapter.appendingPathComponent("adapters.safetensors"))
        let request = try DecisionRequest.decode(Data(contentsOf: URL(fileURLWithPath: probe)))
        let expected = try await exportProbe(
            model: options.modelURL, adapter: adapter, contract: contract, request: request)
        let destination = URL(fileURLWithPath: output)
        guard !FileManager.default.fileExists(atPath: destination.path) else {
            throw ValidationError("Export destination exists.")
        }
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        let staging = destination.deletingLastPathComponent().appendingPathComponent(
            ".afterglow-export-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: staging) }
        try FileManager.default.copyItem(at: options.modelURL, to: staging.appendingPathComponent("base"))
        try FileManager.default.copyItem(at: adapter, to: staging.appendingPathComponent("adapter"))
        try DecisionFiles.write(contract, to: staging.appendingPathComponent(DecisionModelContract.filename))
        try DecisionFiles.write(
            contract, to: staging.appendingPathComponent("adapter/" + DecisionModelContract.filename))
        try DecisionFiles.write(expected, to: staging.appendingPathComponent("reload-probe.json"))
        let actual = try await exportProbe(
            model: staging.appendingPathComponent("base"), adapter: staging.appendingPathComponent("adapter"),
            contract: contract, request: request)
        guard expected.output == actual.output,
            expected.fields.allSatisfy({ name, field in
                field.probabilities.allSatisfy { key, probability in
                    abs(probability - (actual.fields[name]?.probabilities[key] ?? -1)) <= 1e-5
                }
            })
        else {
            throw ValidationError("Export reload differs; candidate was not published.")
        }
        try DecisionFiles.write(
            [
                "verified": "true", "base_sha256": contract.baseFingerprint!,
                "adapter_sha256": contract.adapterFingerprint!,
            ], to: staging.appendingPathComponent("export-verification.json"))
        try FileManager.default.moveItem(at: staging, to: destination)
        print("Export reload verified at \(output).")
    }
}

private func exportProbe(
    model: URL, adapter: URL, contract: DecisionModelContract,
    request: DecisionRequest
) async throws -> DecisionResponse {
    let container = try await DecisionRuntime.load(model: model, adapter: adapter)
    return try await container.perform { context in
        try DecisionRuntime.score(context: context, request: request, contract: contract)
    }
}

struct Interface: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ui", abstract: "Simple loom/weft training and quantization interface.")
    mutating func run() async throws {
        guard AfterglowTerminal.isInteractive else {
            print(Afterglow.helpMessage())
            return
        }
        try await AfterglowTerminal().run()
    }
}
