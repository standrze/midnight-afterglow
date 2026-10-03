// Extracted for Wick from Midnight: Sources/ModelRunnerProtocol/ModelEngine.swift
// Source snapshot SHA256: b00be16b1ee7c8f4f6c580f07fead1fc170386311d0ee10151446d4af6b6ac8a
// Retains the Apache-2.0 license and original third-party attribution.
// This local copy is maintained independently; no Midnight checkout is required.

import Foundation

public enum ModelEngine: String, Codable, CaseIterable, Sendable {
    case auto
    case metal
    case cuda
    case cpu

    public init(argument: String) throws {
        let normalized = argument.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard let engine = Self(rawValue: normalized) else {
            throw ModelEngineError.unknownEngine(argument)
        }
        self = engine
    }

    public func resolve(for backend: CompiledMLXBackend = .current) throws -> ModelEngine {
        switch self {
        case .auto:
            return backend.engine
        case .cpu:
            return .cpu
        case .metal where backend == .metal:
            return .metal
        case .cuda where backend == .cuda:
            return .cuda
        case .metal, .cuda:
            throw ModelEngineError.unavailable(requested: self, compiled: backend)
        }
    }
}

public enum CompiledMLXBackend: String, Codable, Sendable {
    case metal
    case cuda
    case cpu

    public static var current: Self {
        #if MLX_METAL_BACKEND
            .metal
        #elseif MLX_CUDA_BACKEND
            .cuda
        #else
            .cpu
        #endif
    }

    public var engine: ModelEngine {
        switch self {
        case .metal: .metal
        case .cuda: .cuda
        case .cpu: .cpu
        }
    }
}

public enum ModelEngineError: LocalizedError, Equatable {
    case unknownEngine(String)
    case unavailable(requested: ModelEngine, compiled: CompiledMLXBackend)

    public var errorDescription: String? {
        switch self {
        case .unknownEngine(let value):
            let choices = ModelEngine.allCases.map(\.rawValue).joined(separator: ", ")
            return "Unknown engine '\(value)'. Choose one of: \(choices)."
        case .unavailable(let requested, let compiled):
            return "Engine '\(requested.rawValue)' is unavailable in this \(compiled.rawValue) build."
        }
    }
}
