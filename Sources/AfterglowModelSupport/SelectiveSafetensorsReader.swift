// Extracted for Wick from Midnight: Sources/ModelRunnerCore/SelectiveSafetensorsReader.swift
// Source snapshot SHA256: c35e55d6f0322c8616a836f790fdcf74d3cf43fc2f05f361426e86927e972f8b
// Retains the Apache-2.0 license and original third-party attribution.
// This local copy is maintained independently; no Midnight checkout is required.

import Foundation
import MLX

/// Reads only requested tensor byte ranges.
///
/// Headers are cached; weight payloads and shard mappings are never retained by the reader.
public final class SelectiveSafetensorsReader {
    public struct TensorDescription {
        public let shape: [Int]
        public let dtype: DType
        public let bytes: Int
        fileprivate let offset: UInt64
        fileprivate let shard: URL
    }
    private let directory: URL
    private let weightMap: [String: String]
    private var headers: [String: [String: TensorDescription]] = [:]
    public var keys: [String] { weightMap.keys.sorted() }

    public init(directory: URL) throws {
        self.directory = directory.standardizedFileURL.resolvingSymlinksInPath()
        let data = try Data(contentsOf: directory.appendingPathComponent("model.safetensors.index.json"))
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let map = json["weight_map"] as? [String: String], !map.isEmpty
        else { throw LagunaActivationStatisticsError.invalidInput("invalid source safetensors index") }
        weightMap = map
    }

    public func description(for key: String) throws -> TensorDescription {
        guard let shardName = weightMap[key] else {
            throw LagunaActivationStatisticsError.invalidInput("missing source tensor \(key)")
        }
        if headers[shardName] == nil { headers[shardName] = try readHeader(shardName) }
        guard let description = headers[shardName]?[key] else {
            throw LagunaActivationStatisticsError.invalidInput("tensor \(key) absent from its indexed shard")
        }
        return description
    }

    public func read(_ key: String) throws -> MLXArray {
        #if canImport(ObjectiveC)
            // FileHandle/Data can leave autoreleased NSData payloads alive until the
            // outer worker pool drains. Bound that lifetime to one tensor, not a layer
            // or checkpoint. MLXArray(Data, ...) copies synchronously through
            // mlx_array_new_data -> array::init (allocator::malloc + std::copy).
            return try autoreleasepool { try readPayload(key) }
        #else
            return try readPayload(key)
        #endif
    }

    private func readPayload(_ key: String) throws -> MLXArray {
        let description = try description(for: key)
        let file = try FileHandle(forReadingFrom: description.shard)
        defer { try? file.close() }
        try file.seek(toOffset: description.offset)
        guard let data = try file.read(upToCount: description.bytes), data.count == description.bytes else {
            throw LagunaActivationStatisticsError.invalidInput("truncated payload for \(key)")
        }
        return MLXArray(data, description.shape, dtype: description.dtype)
    }

    public func read(keys: [String]) throws -> [String: MLXArray] {
        var arrays = [String: MLXArray]()
        for key in keys { arrays[key] = try read(key) }
        return arrays
    }

    private func readHeader(_ name: String) throws -> [String: TensorDescription] {
        // Reject lexical index traversal while allowing normal HF snapshot files
        // that are symlinks into a shared external blob directory.
        let shard = directory.appendingPathComponent(name).standardizedFileURL
        guard !name.hasPrefix("/"), shard.path.hasPrefix(directory.path + "/") else {
            throw LagunaActivationStatisticsError.invalidInput("shard escapes source directory")
        }
        let file = try FileHandle(forReadingFrom: shard)
        defer { try? file.close() }
        let fileSize = try file.seekToEnd()
        try file.seek(toOffset: 0)
        guard let prefix = try file.read(upToCount: 8), prefix.count == 8 else {
            throw LagunaActivationStatisticsError.invalidInput("truncated safetensors header")
        }
        let length = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << ($1.offset * 8) }
        guard length > 0, length <= 64 * 1_024 * 1_024, length <= fileSize - 8,
            let data = try file.read(upToCount: Int(length)), data.count == Int(length),
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { throw LagunaActivationStatisticsError.invalidInput("invalid safetensors header length/object") }
        let dtypes: [String: DType] = [
            "BF16": .bfloat16, "F16": .float16, "F32": .float32,
            "F64": .float64, "I32": .int32, "I64": .int64, "U32": .uint32, "U8": .uint8,
        ]
        var result = [String: TensorDescription]()
        for (key, value) in json where key != "__metadata__" {
            guard let object = value as? [String: Any], let tag = object["dtype"] as? String,
                let dtype = dtypes[tag], let shape = object["shape"] as? [Int],
                shape.allSatisfy({ $0 > 0 }), let offsets = object["data_offsets"] as? [UInt64],
                offsets.count == 2, offsets[0] <= offsets[1], offsets[1] <= fileSize - 8 - length
            else { throw LagunaActivationStatisticsError.invalidInput("invalid tensor descriptor \(key)") }
            var bytes = dtype.size
            for dimension in shape {
                let (next, overflow) = bytes.multipliedReportingOverflow(by: dimension)
                guard !overflow else { throw LagunaActivationStatisticsError.invalidInput("tensor size overflow") }
                bytes = next
            }
            guard UInt64(bytes) == offsets[1] - offsets[0] else {
                throw LagunaActivationStatisticsError.invalidInput("tensor byte length mismatch for \(key)")
            }
            result[key] = TensorDescription(
                shape: shape, dtype: dtype, bytes: bytes,
                offset: 8 + length + offsets[0], shard: shard)
        }
        return result
    }
}
