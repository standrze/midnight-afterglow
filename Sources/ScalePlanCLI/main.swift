import ArgumentParser
import Foundation
import ScalePlanMLX

@main
struct ScalePlanCommand: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: CommandLine.arguments.first?.split(separator: "/").last.map(String.init)
      ?? "wick-scale-plan",
    abstract: "Build experimental decode-first and prefill-first Q4R8 plans from a measured ledger."
  )

  @Argument(help: "Measured ScalePlan ledger JSON.")
  var ledger: String

  @Argument(help: "New ScalePlan bundle JSON to write.")
  var output: String

  @Option(help: "Maximum stored bytes for either plan.")
  var maxStoredBytes: Int64

  @Option(help: "Maximum summed warm decode latency in nanoseconds.")
  var maxDecodeNS: Int64

  @Option(help: "Maximum summed warm prefill latency in nanoseconds.")
  var maxPrefillNS: Int64

  @Option(help: "Conservative stored-byte DP quantum.")
  var byteQuantum: Int64 = 4_096

  @Option(help: "Conservative latency DP quantum in nanoseconds.")
  var latencyQuantumNS: Int64 = 100

  @Option(help: "Maximum dynamic-programming states before failing closed.")
  var maxStates: Int = 250_000

  @Flag(
    help: "Permit an unvalidated quality proxy for plumbing experiments; output remains explicitly experimental."
  )
  var allowUnvalidatedProxy = false

  @Flag(help: "Replace an existing output file.")
  var overwrite = false

  mutating func run() throws {
    let ledgerURL = URL(fileURLWithPath: ledger).standardizedFileURL
    let outputURL = URL(fileURLWithPath: output).standardizedFileURL
    guard ledgerURL != outputURL else {
      throw ValidationError("The ledger and output paths must differ.")
    }
    if FileManager.default.fileExists(atPath: outputURL.path) && !overwrite {
      throw ValidationError("Output already exists; choose a new path or pass --overwrite.")
    }

    let decoder = JSONDecoder()
    let source = try decoder.decode(
      ScalePlanLedger.self,
      from: Data(contentsOf: ledgerURL)
    )
    let common = (
      maximumStoredBytes: maxStoredBytes,
      byteQuantum: byteQuantum,
      latencyQuantumNS: latencyQuantumNS,
      maximumStates: maxStates,
      allowUnvalidatedProxy: allowUnvalidatedProxy
    )
    let bundle = try ScalePlanner().makeBundle(
      ledger: source,
      decodeConstraints: ScalePlanConstraints(
        maximumStoredBytes: common.maximumStoredBytes,
        maximumLatencyNS: maxDecodeNS,
        byteQuantum: common.byteQuantum,
        latencyQuantumNS: common.latencyQuantumNS,
        maximumStates: common.maximumStates,
        allowUnvalidatedProxy: common.allowUnvalidatedProxy
      ),
      prefillConstraints: ScalePlanConstraints(
        maximumStoredBytes: common.maximumStoredBytes,
        maximumLatencyNS: maxPrefillNS,
        byteQuantum: common.byteQuantum,
        latencyQuantumNS: common.latencyQuantumNS,
        maximumStates: common.maximumStates,
        allowUnvalidatedProxy: common.allowUnvalidatedProxy
      )
    )

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    let data = try encoder.encode(bundle)
    try data.write(to: outputURL, options: .atomic)

    print(ScalePlanner.experimentalNotice)
    print("Wrote \(outputURL.path)")
    printSummary(bundle.decodeFirst)
    printSummary(bundle.prefillFirst)
  }

  private func printSummary(_ profile: ScalePlanProfile) {
    let q8Count = profile.selections.count { $0.candidate.bits == 8 }
    print(
      "\(profile.objective.rawValue): \(profile.selections.count) units, "
        + "\(q8Count) Q8, \(profile.totals.storedBytes) bytes, "
        + "decode=\(profile.totals.decodeNS) ns, prefill=\(profile.totals.prefillNS) ns, "
        + "predicted_loss=\(profile.totals.predictedQualityLoss)"
    )
  }
}
