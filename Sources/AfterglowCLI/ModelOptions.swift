import AfterglowConsole
import ArgumentParser
import DecisionMLX
import DecisionModels
import Foundation
import QuantizationCommands

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
