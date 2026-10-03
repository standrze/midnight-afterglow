// Extracted for Wick from Midnight: Sources/ModelRunnerCore/LagunaRoutedActivationStatistics.swift
// Source snapshot SHA256: d5c76678df7bf4763e5b8a839fead33bf153593ed3feb37ef86162d742a4704b
// Retains the Apache-2.0 license and original third-party attribution.
// This local copy is maintained independently; no Midnight checkout is required.

import Foundation
import MLX

/// Receives the actual inputs selected for each routed projection during an
/// explicit calibration run. Calls stay on the model's serialized MLX worker.
public protocol LagunaRoutedActivationObserver: AnyObject {
  func observeRoutedProjection(path: String, input: MLXArray, indices: MLXArray, expertCount: Int)
}

public struct LagunaExpertActivationStatistics {
  public let path: String
  /// Shape [experts, input channels], including zero rows for unobserved experts.
  public let secondMoments: MLXArray
  public let expertPositionCounts: [Int]
  public let minimumExpertPositions: Int

  /// Unobserved/under-covered experts must retain their original quantization.
  public var eligibleExperts: [Bool] {
    expertPositionCounts.map { $0 >= minimumExpertPositions }
  }
}

public enum LagunaActivationStatisticsError: Error, LocalizedError {
  case invalidInput(String)
  public var errorDescription: String? {
    switch self {
    case .invalidInput(let message): "Invalid Laguna expert activation statistics: \(message)"
    }
  }
}

/// Accumulates E[x² | expert selected] separately for every expert. Gate/up
/// receives the selected pre-MLP hidden state; down receives that expert's
/// actual post-SwiGLU activation. No dense average is substituted for coverage.
public final class LagunaRoutedActivationRecorder: LagunaRoutedActivationObserver {
  private struct Entry {
    var sums: MLXArray
    var counts: MLXArray
    var routedPositions: Int
  }
  private var entries = [String: Entry]()
  private var failure: LagunaActivationStatisticsError?
  public let minimumExpertPositions: Int

  public init(minimumExpertPositions: Int = 32) throws {
    guard minimumExpertPositions > 0 else {
      throw LagunaActivationStatisticsError.invalidInput("minimum expert positions must be positive")
    }
    self.minimumExpertPositions = minimumExpertPositions
  }

  public func observeRoutedProjection(
    path: String, input: MLXArray, indices: MLXArray, expertCount: Int
  ) {
    guard failure == nil else { return }
    do {
      try observe(path: path, input: input, indices: indices, expertCount: expertCount)
    } catch let error as LagunaActivationStatisticsError {
      failure = error
    } catch {
      failure = .invalidInput(error.localizedDescription)
    }
  }

  private func observe(
    path: String, input: MLXArray, indices: MLXArray, expertCount: Int
  ) throws {
    guard expertCount > 0, expertCount <= Int(Int32.max), !path.isEmpty, input.ndim >= 2,
      input.dim(-2) == 1, input.dim(-1) > 0, indices.size > 0,
      indices.dtype == .int32 || indices.dtype == .uint32
    else {
      throw LagunaActivationStatisticsError.invalidInput("invalid projection geometry for \(path)")
    }
    let width = input.dim(-1)
    let targetShape = indices.shape + [1, width]
    guard input.ndim <= targetShape.count else {
      throw LagunaActivationStatisticsError.invalidInput("input rank exceeds routed shape for \(path)")
    }
    let paddedShape = Array(repeating: 1, count: targetShape.count - input.ndim) + input.shape
    guard zip(paddedShape, targetShape).allSatisfy({ $0.0 == 1 || $0.0 == $0.1 }) else {
      throw LagunaActivationStatisticsError.invalidInput("input does not broadcast to selected experts for \(path)")
    }
    let ids = indices.flattened().asType(.int32)
    try MLX.checkedEval(ids)
    let idValues = ids.asArray(Int32.self)
    guard idValues.allSatisfy({ $0 >= 0 && $0 < expertCount }) else {
      throw LagunaActivationStatisticsError.invalidInput("out-of-range expert ID for \(path)")
    }
    var entry = entries[path] ?? Entry(
      sums: MLXArray.zeros([expertCount, width], type: Float.self),
      counts: MLXArray.zeros([expertCount], type: Int32.self),
      routedPositions: 0)
    guard entry.sums.shape == [expertCount, width],
      indices.size <= Int(Int32.max) - entry.routedPositions
    else {
      throw LagunaActivationStatisticsError.invalidInput("changed geometry or count overflow for \(path)")
    }
    let routed = MLX.broadcast(input, to: targetShape).reshaped(indices.size, width).asType(.float32)
    entry.sums = entry.sums.at[ids].add(MLX.square(routed))
    entry.counts = entry.counts.at[ids].add(MLXArray.ones([indices.size], type: Int32.self))
    entry.routedPositions += indices.size
    entries[path] = entry
  }

  public func evaluatePending() throws {
    if let failure { throw failure }
    guard !entries.isEmpty else {
      throw LagunaActivationStatisticsError.invalidInput("no routed activations were observed")
    }
    try MLX.checkedEval(entries.values.flatMap { [$0.sums, $0.counts] })
  }

  public func finalize() throws -> [LagunaExpertActivationStatistics] {
    try evaluatePending()
    return try entries.keys.sorted().map { path in
      let entry = entries[path]!
      let counts = entry.counts.asArray(Int32.self).map(Int.init)
      let divisor = MLX.maximum(entry.counts, 1).asType(.float32).expandedDimensions(axis: -1)
      let moments = entry.sums / divisor
      try MLX.checkedEval(moments)
      guard moments.asArray(Float.self).allSatisfy({ $0.isFinite && $0 >= 0 }) else {
        throw LagunaActivationStatisticsError.invalidInput("non-finite moments for \(path)")
      }
      return LagunaExpertActivationStatistics(
        path: path, secondMoments: moments, expertPositionCounts: counts,
        minimumExpertPositions: minimumExpertPositions)
    }
  }
}
