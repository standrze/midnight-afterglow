import AfterglowConsole
import ArgumentParser
import DecisionMLX
import DecisionModels
import Foundation
import QuantizationCommands

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
