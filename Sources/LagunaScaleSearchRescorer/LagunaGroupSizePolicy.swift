import Foundation

struct LagunaQuantizationGeometry: Codable, Equatable {
  let groupSize: Int
  let bits: Int
  let mode: String

  enum CodingKeys: String, CodingKey {
    case groupSize = "group_size"
    case bits, mode
  }
}

/// G128 changes scale/bias geometry. Explicit per-module entries prevent the
/// preserved G64 embedding and Q8 routers from inheriting the new default.
struct LagunaGroupSizePolicy {
  let groupSize: Int
  let overrides: [String: LagunaQuantizationGeometry]
  let configuration: Data

  init(templateConfiguration: Data, groupSize: Int,
    q4Modules: [String], q8Modules: [String], embeddings: [String]) throws
  {
    guard [64, 128].contains(groupSize),
      var config = try JSONSerialization.jsonObject(with: templateConfiguration) as? [String: Any]
    else { throw LagunaActivationInputError(message: "unsupported output group size or template configuration") }
    let original = (config["quantization"] ?? config["quantization_config"]) as? [String: Any]
      ?? ["bits": 4, "group_size": 64, "mode": "affine"]
    for (modules, bits) in [(q4Modules + embeddings, 4), (q8Modules, 8)] {
      for module in modules {
        let moduleOverride = original[module] as? [String: Any] ?? [:]
        let actualBits = moduleOverride["bits"] as? Int ?? original["bits"] as? Int ?? 4
        let actualGroupSize = moduleOverride["group_size"] as? Int ?? original["group_size"] as? Int ?? 64
        let mode = moduleOverride["mode"] as? String ?? original["mode"] as? String ?? "affine"
        guard actualBits == bits, actualGroupSize == 64, mode == "affine" else {
          throw LagunaActivationInputError(message: "group-size challenger requires an affine Q4/G64 template with Q8/G64 routers: \(module)")
        }
      }
    }
    var overrides: [String: LagunaQuantizationGeometry] = [:]
    for module in q4Modules { overrides[module] = .init(groupSize: groupSize, bits: 4, mode: "affine") }
    for module in q8Modules { overrides[module] = .init(groupSize: 64, bits: 8, mode: "affine") }
    for module in embeddings { overrides[module] = .init(groupSize: 64, bits: 4, mode: "affine") }
    self.groupSize = groupSize
    self.overrides = overrides
    var quantization: [String: Any] = ["bits": 4, "group_size": groupSize, "mode": "affine"]
    for (module, geometry) in overrides {
      quantization[module] = ["bits": geometry.bits, "group_size": geometry.groupSize, "mode": geometry.mode]
    }
    config["quantization"] = quantization
    config["quantization_config"] = quantization
    configuration = try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
  }

  static func updatedIndex(_ original: Data, tensorBytes: Int) throws -> Data {
    guard tensorBytes >= 0,
      var index = try JSONSerialization.jsonObject(with: original) as? [String: Any],
      index["weight_map"] is [String: String]
    else { throw LagunaActivationInputError(message: "invalid template index for group-size conversion") }
    var metadata = index["metadata"] as? [String: Any] ?? [:]
    metadata["total_size"] = tensorBytes
    index["metadata"] = metadata
    return try JSONSerialization.data(withJSONObject: index, options: [.prettyPrinted, .sortedKeys])
  }
}
