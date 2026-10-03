import Foundation
import MLX
import MistralActivationScaleSearchCore
import WickModelSupport

/// Publishes a bounded, source-bound input dataset separately from checkpoints.
final class GemmaProjectionInputExport {
    struct Target: Codable {
        let path: String
        let expert: Int?
    }
    struct Layer: Codable {
        let index: Int
        let targets: [Target]
    }
    struct Plan: Codable {
        let maximumPositions: Int
        let maximumRetainedBytes: Int
        let maximumExportBytes: Int
        let minimumFreeBytes: Int
        let layers: [Layer]

        enum CodingKeys: String, CodingKey {
            case maximumPositions = "maximum_positions"
            case maximumRetainedBytes = "maximum_retained_bytes"
            case maximumExportBytes = "maximum_export_bytes"
            case minimumFreeBytes = "minimum_free_bytes"
            case layers
        }
    }
    struct Entry: Codable {
        let layer: Int
        let path: String
        let expert: Int?
        let file: String
        let fingerprint: String
        let shape: [Int]
        let dtype: String
        let observedPositions: Int
        let capturedPositions: Int
    }
    struct Manifest: Codable {
        let format: String
        let selection: String
        let plan: Plan
        let planFingerprint: String
        let provenance: GemmaActivationProvenance
        let entries: [Entry]
    }

    private let plan: Plan
    private let planFingerprint: String
    private let destination: URL
    private let staging: URL
    private var entries = [Entry]()
    private var bytesWritten = 0
    private var committed = false

    init(planURL: URL, destination: URL, source: URL, configuration: Gemma4CalibrationConfiguration) throws {
        let data = try Data(contentsOf: planURL)
        plan = try JSONDecoder().decode(Plan.self, from: data)
        planFingerprint = GemmaActivationProvenance.fingerprint(data)
        let source = source.resolvingSymlinksInPath()
        let parent = destination.deletingLastPathComponent().resolvingSymlinksInPath()
        self.destination = parent.appendingPathComponent(destination.lastPathComponent, isDirectory: true)
        guard !FileManager.default.fileExists(atPath: self.destination.path),
            FileManager.default.fileExists(atPath: parent.path), parent != source,
            !parent.path.hasPrefix(source.path + "/"), !plan.layers.isEmpty,
            (1...4096).contains(plan.maximumPositions),
            (1...1_073_741_824).contains(plan.maximumRetainedBytes),
            (1...4_294_967_296).contains(plan.maximumExportBytes), plan.minimumFreeBytes >= 0,
            Set(plan.layers.map(\.index)).count == plan.layers.count
        else { throw Gemma4CalibrationError.invalidInput("invalid capture plan, limits or output directory") }
        for layer in plan.layers {
            let prefix = configuration.moduleRoot + ".layers.\(layer.index)."
            guard (0..<configuration.layerCount).contains(layer.index), !layer.targets.isEmpty,
                layer.targets.allSatisfy({ $0.path.hasPrefix(prefix) && ($0.expert ?? 0) >= 0 })
            else { throw Gemma4CalibrationError.invalidInput("capture target does not belong to selected layer") }
            _ = try Gemma4ProjectionInputRecorder(
                targets: layer.targets.map { .init(path: $0.path, expert: $0.expert) },
                maximumPositions: plan.maximumPositions, maximumRetainedBytes: plan.maximumRetainedBytes)
        }
        staging = parent.appendingPathComponent(".gemma-projection-inputs-\(UUID().uuidString)", isDirectory: true)
        try Self.requireFreeBytes(parent: parent, minimum: plan.minimumFreeBytes)
        try FileManager.default.createDirectory(
            at: staging, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
    }

    deinit {
        if !committed { try? FileManager.default.removeItem(at: staging) }
    }

    func recorder(layer: Int) throws -> Gemma4ProjectionInputRecorder? {
        guard let selected = plan.layers.first(where: { $0.index == layer }) else { return nil }
        return try Gemma4ProjectionInputRecorder(
            targets: selected.targets.map { .init(path: $0.path, expert: $0.expert) },
            maximumPositions: plan.maximumPositions, maximumRetainedBytes: plan.maximumRetainedBytes)
    }

    func consume(layer: Int, inputs: [Gemma4CapturedProjectionInputs]) throws {
        for input in inputs {
            let (estimate, estimateOverflow) = input.inputs.nbytes.addingReportingOverflow(131_072)
            let (next, overflow) = bytesWritten.addingReportingOverflow(estimate)
            guard !estimateOverflow, !overflow, next <= plan.maximumExportBytes else {
                throw Gemma4CalibrationError.invalidInput("projection export exceeds explicit byte budget")
            }
            let (requiredFree, freeOverflow) = plan.minimumFreeBytes.addingReportingOverflow(estimate)
            guard !freeOverflow else { throw Gemma4CalibrationError.invalidInput("disk reserve overflow") }
            try Self.requireFreeBytes(parent: staging, minimum: requiredFree)
            let name = String(format: "layer-%04d-input-%04d.safetensors", layer, entries.count)
            let file = staging.appendingPathComponent(name)
            try MLX.save(arrays: ["inputs": input.inputs], url: file)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: file.path)
            let data = try Data(contentsOf: file)
            guard data.count <= estimate else {
                throw Gemma4CalibrationError.invalidInput("projection file exceeds reserved byte estimate")
            }
            bytesWritten += data.count
            entries.append(
                Entry(
                    layer: layer, path: input.target.path, expert: input.target.expert, file: name,
                    fingerprint: GemmaActivationProvenance.fingerprint(data), shape: input.inputs.shape,
                    dtype: String(describing: input.inputs.dtype), observedPositions: input.observedPositions,
                    capturedPositions: input.inputs.dim(0)))
        }
    }

    /// Call only after the normal statistics writer verifies the source unchanged.
    func finish(provenance: GemmaActivationProvenance) throws {
        guard entries.count == plan.layers.reduce(0, { $0 + $1.targets.count }) else {
            throw Gemma4CalibrationError.invalidInput("incomplete selected projection export")
        }
        let manifest = Manifest(
            format: "gemma4_projection_inputs_v1", selection: "deterministic_prefix_per_projection_or_selected_expert",
            plan: plan, planFingerprint: planFingerprint, provenance: provenance, entries: entries)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let metadata = try encoder.encode(manifest)
        let (total, overflow) = bytesWritten.addingReportingOverflow(metadata.count)
        guard !overflow, total <= plan.maximumExportBytes else {
            throw Gemma4CalibrationError.invalidInput("projection manifest exceeds export budget")
        }
        let (requiredFree, freeOverflow) = plan.minimumFreeBytes.addingReportingOverflow(metadata.count)
        guard !freeOverflow else { throw Gemma4CalibrationError.invalidInput("disk reserve overflow") }
        try Self.requireFreeBytes(parent: staging, minimum: requiredFree)
        try metadata.write(to: staging.appendingPathComponent("manifest.json"), options: .atomic)
        guard !FileManager.default.fileExists(atPath: destination.path) else {
            throw Gemma4CalibrationError.invalidInput("projection output appeared during collection")
        }
        try FileManager.default.moveItem(at: staging, to: destination)
        committed = true
    }

    private static func requireFreeBytes(parent: URL, minimum: Int) throws {
        guard let available = try parent.resourceValues(forKeys: [.volumeAvailableCapacityKey]).volumeAvailableCapacity,
            available >= minimum
        else { throw Gemma4CalibrationError.invalidInput("insufficient or unavailable projection export disk reserve") }
    }
}
