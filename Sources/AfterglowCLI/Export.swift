import AfterglowConsole
import ArgumentParser
import DecisionMLX
import DecisionModels
import Foundation
import QuantizationCommands

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
