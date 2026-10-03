import Foundation
import MLX
import QuantizerSupport

/// Accumulates at most one configured shard plus one oversized tensor. It never
/// retains a complete checkpoint. Index publication happens only after flushing.
final class GemmaCheckpointWriter {
    private let directory: URL
    private let maximumBytes: Int
    private var arrays = [String: MLXArray]()
    private var bufferedBytes = 0
    private var totalBytes = 0
    private var weightMap = [String: String]()
    private var shardCount = 0

    init(directory: URL, maximumBytes: Int) {
        self.directory = directory
        self.maximumBytes = maximumBytes
    }

    func append(_ name: String, _ tensor: MLXArray) throws {
        guard weightMap[name] == nil, arrays[name] == nil else {
            throw QuantizerInputError("Duplicate output tensor: \(name)")
        }
        if !arrays.isEmpty, tensor.nbytes > maximumBytes - bufferedBytes { try flush() }
        try MLX.checkedEval(tensor)
        arrays[name] = tensor
        bufferedBytes += tensor.nbytes
        totalBytes += tensor.nbytes
        if bufferedBytes >= maximumBytes { try flush() }
    }

    func finish() throws {
        try flush()
        try JSONSerialization.data(
            withJSONObject: [
                "metadata": ["total_size": totalBytes], "weight_map": weightMap,
            ], options: [.prettyPrinted, .sortedKeys]
        ).write(
            to: directory.appendingPathComponent("model.safetensors.index.json"))
    }

    private func flush() throws {
        guard !arrays.isEmpty else { return }
        shardCount += 1
        let name = String(format: "model-%05d.safetensors", shardCount)
        try MLX.save(arrays: arrays, url: directory.appendingPathComponent(name))
        for key in arrays.keys { weightMap[key] = name }
        arrays.removeAll()
        bufferedBytes = 0
        Memory.clearCache()
    }
}
