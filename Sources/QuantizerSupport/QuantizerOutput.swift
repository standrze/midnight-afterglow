import Foundation

public struct QuantizerInputError: Error, LocalizedError {
  public let message: String

  public init(_ message: String) { self.message = message }

  public var errorDescription: String? { message }
}

public enum QuantizerShardSize {
  /// Convert only after rounding and checking representability. Double(Int64.max)
  /// rounds up to 2^63, so comparing against it before an Int64 cast is unsafe.
  public static func bytes(fromGiB gib: Double) throws -> Int64 {
    let bytes = gib * 1_024 * 1_024 * 1_024
    guard gib.isFinite, gib > 0, bytes.isFinite, bytes >= 1,
      let size = Int64(exactly: bytes.rounded(.towardZero))
    else {
      throw QuantizerInputError(
        "--max-shard-gib must describe at least one byte and fit in a signed 64-bit byte count.")
    }
    return size
  }
}

/// Writes a complete checkpoint beside its final location before publishing it.
/// Failed conversions leave an existing output intact, including with overwrite.
public struct QuantizerOutputTransaction {
  public let stagingDirectory: URL
  public let destinationDirectory: URL
  private let sourceDirectory: URL
  private let overwrite: Bool

  public static func validate(
    sourceDirectory: URL, destinationDirectory: URL, overwrite: Bool
  ) throws {
    let source = canonicalDirectory(sourceDirectory).pathComponents
    let destination = canonicalDirectory(destinationDirectory).pathComponents
    guard !source.starts(with: destination), !destination.starts(with: source) else {
      throw QuantizerInputError("source and destination directories must not overlap, including symbolic links.")
    }
    guard overwrite || !FileManager.default.fileExists(atPath: destinationDirectory.path) else {
      throw QuantizerInputError("destination already exists; choose a new directory or use --overwrite.")
    }
  }

  private static func canonicalDirectory(_ directory: URL) -> URL {
    // Foundation does not consistently resolve a symlink ancestor when the
    // final path does not exist yet, which is normal for conversion outputs.
    var ancestor = directory.standardizedFileURL
    var missingComponents = [String]()
    while !FileManager.default.fileExists(atPath: ancestor.path), ancestor.path != "/" {
      missingComponents.append(ancestor.lastPathComponent)
      ancestor.deleteLastPathComponent()
    }
    return missingComponents.reversed().reduce(ancestor.resolvingSymlinksInPath()) {
      $0.appendingPathComponent($1)
    }.standardizedFileURL
  }

  public init(sourceDirectory: URL, destinationDirectory: URL, overwrite: Bool) throws {
    try Self.validate(sourceDirectory: sourceDirectory, destinationDirectory: destinationDirectory,
      overwrite: overwrite)
    self.sourceDirectory = sourceDirectory
    self.destinationDirectory = destinationDirectory
    self.overwrite = overwrite
    stagingDirectory = destinationDirectory.deletingLastPathComponent().appendingPathComponent(
      ".\(destinationDirectory.lastPathComponent).partial-\(UUID().uuidString)", isDirectory: true)
  }

  public func cleanup() {
    try? FileManager.default.removeItem(at: stagingDirectory)
  }

  /// Returns a retained backup path only when publication succeeded but removal
  /// of the old checkpoint failed. The caller can report it without marking the
  /// successfully published checkpoint as a failed conversion.
  @discardableResult
  public func commit() throws -> URL? {
    let manager = FileManager.default
    try Self.validate(sourceDirectory: sourceDirectory, destinationDirectory: destinationDirectory,
      overwrite: overwrite)
    let backup = destinationDirectory.deletingLastPathComponent().appendingPathComponent(
      ".\(destinationDirectory.lastPathComponent).backup-\(UUID().uuidString)", isDirectory: true)
    let hasExistingOutput = manager.fileExists(atPath: destinationDirectory.path)
    if hasExistingOutput {
      try manager.moveItem(at: destinationDirectory, to: backup)
    }
    do {
      try manager.moveItem(at: stagingDirectory, to: destinationDirectory)
    } catch {
      if hasExistingOutput {
        do {
          try manager.moveItem(at: backup, to: destinationDirectory)
        } catch {
          throw QuantizerInputError(
            "Could not publish output or restore it. The previous checkpoint is preserved at \(backup.path).")
        }
      }
      throw error
    }
    if hasExistingOutput {
      do { try manager.removeItem(at: backup) }
      catch { return backup }
    }
    return nil
  }
}
