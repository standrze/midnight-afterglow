// Extracted for Wick from Midnight: Sources/ModelRunnerCore/MLXResourceGuard.swift
// Source snapshot SHA256: 30591ab4c28f91866ed092b7085088c2403e7c5cdaecabca56748531a8d8c6f6
// Retains the Apache-2.0 license and original third-party attribution.
// This local copy is maintained independently; no Midnight checkout is required.

import Foundation
import MLX

public enum MLXResourceGuard {
    public static func apply(_ limits: MLXResourceLimits) throws {
        Memory.memoryLimit = limits.memoryLimitBytes
        Memory.cacheLimit = limits.cacheLimitBytes
        Memory.clearCache()

        let appliedMemoryLimit = Memory.memoryLimit
        let appliedCacheLimit = Memory.cacheLimit
        guard appliedMemoryLimit == limits.memoryLimitBytes,
            appliedCacheLimit == limits.cacheLimitBytes
        else {
            throw MLXResourceGuardError.applicationFailed(
                requestedMemoryBytes: limits.memoryLimitBytes,
                appliedMemoryBytes: appliedMemoryLimit,
                requestedCacheBytes: limits.cacheLimitBytes,
                appliedCacheBytes: appliedCacheLimit
            )
        }
    }
}

enum MLXResourceGuardError: LocalizedError {
    case applicationFailed(
        requestedMemoryBytes: Int,
        appliedMemoryBytes: Int,
        requestedCacheBytes: Int,
        appliedCacheBytes: Int
    )

    var errorDescription: String? {
        switch self {
        case .applicationFailed(
            let requestedMemory,
            let appliedMemory,
            let requestedCache,
            let appliedCache
        ):
            "MLX rejected the resource guard "
                + "(memory requested/applied: \(requestedMemory)/\(appliedMemory), "
                + "cache requested/applied: \(requestedCache)/\(appliedCache)); "
                + "refusing to load the model."
        }
    }
}
