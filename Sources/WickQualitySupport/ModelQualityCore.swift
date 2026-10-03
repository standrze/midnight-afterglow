// Extracted for Wick from Midnight: Sources/ModelQualityCore/ModelQualityCore.swift
// Source snapshot SHA256: 1ed0810b330f73ac38ba1bb5d785a233415f38d88df67088e7c5cad2d04167d4
// Retains the Apache-2.0 license and original third-party attribution.
// This local copy is maintained independently; no Midnight checkout is required.

import Foundation

public struct ModelQualityCorpusSample: Codable, Equatable, Sendable {
  public var id: String
  public var category: String?
  public var text: String

  public init(id: String, category: String? = nil, text: String) {
    self.id = id
    self.category = category
    self.text = text
  }
}

public struct ModelQualityTokenSequence: Equatable, Sendable {
  public var sampleID: String
  public var tokenIDs: [Int]

  public init(sampleID: String, tokenIDs: [Int]) {
    self.sampleID = sampleID
    self.tokenIDs = tokenIDs
  }
}

public struct ModelQualitySampleResult: Codable, Equatable, Sendable {
  public var id: String
  public var category: String?
  public var originalTokenCount: Int
  public var evaluatedTokenCount: Int
  public var scoredTokenCount: Int
  public var truncated: Bool
  public var tokenIDFingerprint: String
  public var nllSum: Double
  public var nll: Double
  public var perplexity: Double

  public init(
    id: String,
    category: String? = nil,
    originalTokenCount: Int,
    evaluatedTokenCount: Int,
    tokenIDFingerprint: String,
    nllSum: Double
  ) throws {
    guard evaluatedTokenCount >= 2, originalTokenCount >= evaluatedTokenCount else {
      throw ModelQualityCoreError.invalidMeasurement(
        id: id,
        detail: "token counts must satisfy original >= evaluated >= 2"
      )
    }
    let scoredTokenCount = evaluatedTokenCount - 1
    guard nllSum.isFinite, nllSum >= 0 else {
      throw ModelQualityCoreError.invalidMeasurement(
        id: id,
        detail: "NLL sum must be finite and nonnegative"
      )
    }
    let nll = nllSum / Double(scoredTokenCount)
    let perplexity = exp(nll)
    guard nll.isFinite, perplexity.isFinite else {
      throw ModelQualityCoreError.invalidMeasurement(
        id: id,
        detail: "mean NLL and perplexity must be finite"
      )
    }

    self.id = id
    self.category = category
    self.originalTokenCount = originalTokenCount
    self.evaluatedTokenCount = evaluatedTokenCount
    self.scoredTokenCount = scoredTokenCount
    self.truncated = evaluatedTokenCount < originalTokenCount
    self.tokenIDFingerprint = tokenIDFingerprint
    self.nllSum = nllSum
    self.nll = nll
    self.perplexity = perplexity
  }

  enum CodingKeys: String, CodingKey {
    case id, category, truncated, nll, perplexity
    case originalTokenCount = "original_token_count"
    case evaluatedTokenCount = "evaluated_token_count"
    case scoredTokenCount = "scored_token_count"
    case tokenIDFingerprint = "token_id_fingerprint"
    case nllSum = "nll_sum"
  }
}

public struct ModelQualitySummary: Codable, Equatable, Sendable {
  public var sampleCount: Int
  public var scoredTokenCount: Int
  public var nllSum: Double
  public var tokenWeightedNLL: Double
  public var perplexity: Double

  public init(
    sampleCount: Int,
    scoredTokenCount: Int,
    nllSum: Double,
    tokenWeightedNLL: Double,
    perplexity: Double
  ) {
    self.sampleCount = sampleCount
    self.scoredTokenCount = scoredTokenCount
    self.nllSum = nllSum
    self.tokenWeightedNLL = tokenWeightedNLL
    self.perplexity = perplexity
  }

  enum CodingKeys: String, CodingKey {
    case perplexity
    case sampleCount = "sample_count"
    case scoredTokenCount = "scored_token_count"
    case nllSum = "nll_sum"
    case tokenWeightedNLL = "token_weighted_nll"
  }
}

public enum ModelQualityCoreError: Error, LocalizedError, Equatable, Sendable {
  case unreadableCorpus(String)
  case corpusIsNotUTF8
  case malformedJSON(line: Int, detail: String)
  case invalidSample(line: Int, detail: String)
  case duplicateID(id: String, line: Int)
  case emptyCorpus
  case invalidMaximumTokenCount(Int)
  case invalidMeasurement(id: String, detail: String)
  case emptyMeasurements
  case tokenCountOverflow

  public var errorDescription: String? {
    switch self {
    case .unreadableCorpus(let detail):
      "Cannot read quality corpus: \(detail)"
    case .corpusIsNotUTF8:
      "Quality corpus is not valid UTF-8."
    case .malformedJSON(let line, let detail):
      "Quality corpus line \(line) is not valid sample JSON: \(detail)"
    case .invalidSample(let line, let detail):
      "Quality corpus line \(line) is invalid: \(detail)"
    case .duplicateID(let id, let line):
      "Quality corpus line \(line) repeats sample id '\(id)'."
    case .emptyCorpus:
      "Quality corpus contains no samples."
    case .invalidMaximumTokenCount(let count):
      "Maximum token count must be at least 2, got \(count)."
    case .invalidMeasurement(let id, let detail):
      "Quality measurement '\(id)' is invalid: \(detail)"
    case .emptyMeasurements:
      "Quality summary requires at least one sample measurement."
    case .tokenCountOverflow:
      "Quality sample token counts overflowed Int."
    }
  }
}

public enum ModelQualityCore {
  public static func loadCorpus(from url: URL) throws -> [ModelQualityCorpusSample] {
    let data: Data
    do {
      data = try Data(contentsOf: url)
    } catch {
      throw ModelQualityCoreError.unreadableCorpus(error.localizedDescription)
    }
    guard let contents = String(data: data, encoding: .utf8) else {
      throw ModelQualityCoreError.corpusIsNotUTF8
    }

    let decoder = JSONDecoder()
    var samples = [ModelQualityCorpusSample]()
    var seenIDs = Set<String>()
    for (offset, line) in contents.split(separator: "\n", omittingEmptySubsequences: false)
      .enumerated()
    {
      let lineNumber = offset + 1
      let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
      if trimmed.isEmpty { continue }

      let sample: ModelQualityCorpusSample
      do {
        sample = try decoder.decode(
          ModelQualityCorpusSample.self,
          from: Data(trimmed.utf8)
        )
      } catch {
        throw ModelQualityCoreError.malformedJSON(
          line: lineNumber,
          detail: error.localizedDescription
        )
      }

      let trimmedID = sample.id.trimmingCharacters(in: .whitespacesAndNewlines)
      guard !trimmedID.isEmpty else {
        throw ModelQualityCoreError.invalidSample(
          line: lineNumber,
          detail: "id must be nonblank"
        )
      }
      guard trimmedID == sample.id else {
        throw ModelQualityCoreError.invalidSample(
          line: lineNumber,
          detail: "id must not have leading or trailing whitespace"
        )
      }
      guard !sample.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        throw ModelQualityCoreError.invalidSample(
          line: lineNumber,
          detail: "text must be nonblank"
        )
      }
      if let category = sample.category {
        let trimmedCategory = category.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedCategory.isEmpty, trimmedCategory == category else {
          throw ModelQualityCoreError.invalidSample(
            line: lineNumber,
            detail: "category must be nonblank and trimmed when present"
          )
        }
      }
      guard seenIDs.insert(sample.id).inserted else {
        throw ModelQualityCoreError.duplicateID(id: sample.id, line: lineNumber)
      }
      samples.append(sample)
    }

    guard !samples.isEmpty else {
      throw ModelQualityCoreError.emptyCorpus
    }
    return samples
  }

  public static func boundedTokens(_ tokens: [Int], maximumCount: Int) throws -> [Int] {
    guard maximumCount >= 2 else {
      throw ModelQualityCoreError.invalidMaximumTokenCount(maximumCount)
    }
    return Array(tokens.prefix(maximumCount))
  }

  public static func corpusFingerprint(_ samples: [ModelQualityCorpusSample]) -> String {
    var hasher = FNV1a64()
    hasher.update(UInt64(samples.count))
    for sample in samples {
      hasher.update(sample.id)
      if let category = sample.category {
        hasher.update(byte: 1)
        hasher.update(category)
      } else {
        hasher.update(byte: 0)
      }
      hasher.update(sample.text)
    }
    return hasher.fingerprint
  }

  public static func tokenIDFingerprint(_ tokens: [Int]) -> String {
    var hasher = FNV1a64()
    hasher.update(UInt64(tokens.count))
    for token in tokens {
      hasher.update(UInt64(bitPattern: Int64(token)))
    }
    return hasher.fingerprint
  }

  public static func combinedTokenIDFingerprint(
    _ sequences: [ModelQualityTokenSequence]
  ) -> String {
    var hasher = FNV1a64()
    hasher.update(UInt64(sequences.count))
    for sequence in sequences {
      hasher.update(sequence.sampleID)
      hasher.update(UInt64(sequence.tokenIDs.count))
      for token in sequence.tokenIDs {
        hasher.update(UInt64(bitPattern: Int64(token)))
      }
    }
    return hasher.fingerprint
  }

  public static func summarize(
    _ samples: [ModelQualitySampleResult]
  ) throws -> ModelQualitySummary {
    guard !samples.isEmpty else {
      throw ModelQualityCoreError.emptyMeasurements
    }

    var scoredTokenCount = 0
    var nllSum = 0.0
    for sample in samples {
      let (updatedCount, overflow) = scoredTokenCount.addingReportingOverflow(
        sample.scoredTokenCount)
      guard !overflow else {
        throw ModelQualityCoreError.tokenCountOverflow
      }
      scoredTokenCount = updatedCount
      nllSum += sample.nllSum
    }
    guard scoredTokenCount > 0, nllSum.isFinite, nllSum >= 0 else {
      throw ModelQualityCoreError.invalidMeasurement(
        id: "aggregate",
        detail: "token count and NLL sum must be finite and positive/nonnegative"
      )
    }
    let tokenWeightedNLL = nllSum / Double(scoredTokenCount)
    let perplexity = exp(tokenWeightedNLL)
    guard tokenWeightedNLL.isFinite, perplexity.isFinite else {
      throw ModelQualityCoreError.invalidMeasurement(
        id: "aggregate",
        detail: "token-weighted NLL and perplexity must be finite"
      )
    }

    return ModelQualitySummary(
      sampleCount: samples.count,
      scoredTokenCount: scoredTokenCount,
      nllSum: nllSum,
      tokenWeightedNLL: tokenWeightedNLL,
      perplexity: perplexity
    )
  }
}

private struct FNV1a64 {
  private var value: UInt64 = 0xcbf29ce484222325

  mutating func update(byte: UInt8) {
    value ^= UInt64(byte)
    value &*= 0x100000001b3
  }

  mutating func update(_ integer: UInt64) {
    for shift in stride(from: 0, through: 56, by: 8) {
      update(byte: UInt8(truncatingIfNeeded: integer >> UInt64(shift)))
    }
  }

  mutating func update(_ string: String) {
    let bytes = Array(string.utf8)
    update(UInt64(bytes.count))
    for byte in bytes {
      update(byte: byte)
    }
  }

  var fingerprint: String {
    let hex = String(value, radix: 16)
    return "fnv1a64:" + String(repeating: "0", count: 16 - hex.count) + hex
  }
}
