// swift-tools-version: 6.4

import PackageDescription

// Reviewed backend revisions are pinned for reproducible Wick builds.
#if os(macOS)
    let backendSwiftSettings: [SwiftSetting] = [
        .define("MLX_METAL_BACKEND")
    ]
#elseif os(Linux)
    let backendSwiftSettings: [SwiftSetting] =
        Context.environment["SPM_CUDA"] == "0"
        ? [.define("MLX_CPU_BACKEND")]
        : [.define("MLX_CUDA_BACKEND")]
#else
    let backendSwiftSettings: [SwiftSetting] = [
        .define("MLX_CPU_BACKEND")
    ]
#endif

#if os(Linux)
    // CUDA support currently lives on this exact post-0.31.6 MLX-Swift
    // revision. Keeping the selection in one conditional manifest lets this
    // package remain the canonical source tree for both CUDA and Metal.
    let mlxSwiftDependency: Package.Dependency = .package(
        url: "https://github.com/ml-explore/mlx-swift",
        revision: "2d2724006b62855c6c2a71df633baf4ee4ad8a0f"
    )
#else
    // The official 0.32 update branch synchronizes MLX Swift with MLX 0.32.2
    // and MLX-C. Pin the reviewed commit until it is published as a release.
    let mlxSwiftDependency: Package.Dependency = .package(
        url: "https://github.com/ml-explore/mlx-swift",
        revision: "72f3c3ad8aeee39bfc94f8fbeb446cac89e3a798"
    )
#endif

let package = Package(
    name: "midnight-afterglow",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "afterglow-int4-proof", targets: ["CompactInt4Proof"]),
        .executable(name: "afterglow-int4-runtime-screen", targets: ["CompactInt4RuntimeScreen"]),
        .executable(name: "midnight-afterglow", targets: ["AfterglowCLI"]),
        .executable(name: "wick", targets: ["ModelQuantizer"]),
        .executable(name: "facet", targets: ["ModelQuantizer"]),
        .executable(
            name: "wick-metal-quant-bench",
            targets: ["MetalQuantizationBenchmark"]
        ),
        .executable(
            name: "model-runner-metal-quant-bench",
            targets: ["MetalQuantizationBenchmark"]
        ),
        .executable(
            name: "wick-laguna-quantize",
            targets: ["LagunaQuantizer"]
        ),
        .executable(
            name: "model-runner-laguna-quantize",
            targets: ["LagunaQuantizer"]
        ),
        .executable(
            name: "wick-scale-plan",
            targets: ["ScalePlanCLI"]
        ),
        .executable(
            name: "model-runner-scale-plan",
            targets: ["ScalePlanCLI"]
        ),
        .executable(
            name: "wick-q4-scale-search-audit",
            targets: ["Q4ScaleSearchAudit"]
        ),
        .executable(
            name: "model-runner-q4-scale-search-audit",
            targets: ["Q4ScaleSearchAudit"]
        ),
        .executable(
            name: "wick-laguna-q4r8-rescore",
            targets: ["LagunaScaleSearchRescorerCLI"]
        ),
        .executable(
            name: "model-runner-laguna-q4r8-rescore",
            targets: ["LagunaScaleSearchRescorerCLI"]
        ),
        .executable(
            name: "model-runner-quantize",
            targets: ["ModelQuantizer"]
        ),
        .executable(
            name: "wick-laguna-q4r8-verify",
            targets: ["LagunaQ4R8Verifier"]
        ),
        .executable(
            name: "model-runner-laguna-q4r8-verify",
            targets: ["LagunaQ4R8Verifier"]
        ),
        .executable(name: "wick-gemma-activation-stats", targets: ["GemmaActivationStats"]),
        .executable(name: "wick-gemma-awss-quantize", targets: ["GemmaActivationQuantizer"]),
        .executable(
            name: "wick-mistral-activation-stats",
            targets: ["MistralActivationStats"]
        ),
        .executable(
            name: "model-runner-mistral-activation-stats",
            targets: ["MistralActivationStats"]
        ),
        .executable(
            name: "wick-mistral-awss-quantize",
            targets: ["MistralActivationScaleSearchRescorer"]
        ),
        .executable(
            name: "model-runner-mistral-awss-quantize",
            targets: ["MistralActivationScaleSearchRescorer"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/standrze/loom.git", exact: "0.1.1"),
        .package(url: "https://github.com/standrze/weft.git", exact: "0.1.1"),
        mlxSwiftDependency,
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm",
            revision: "14414441fa44f45eee35a61e9fa0bab577cf9734",
            traits: []
        ),
        .package(
            url: "https://github.com/huggingface/swift-huggingface",
            exact: "0.9.0"
        ),
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            exact: "1.3.3"
        ),
        .package(
            url: "https://github.com/apple/swift-argument-parser",
            exact: "1.8.2"
        ),
    ],
    targets: [
        .target(name: "CompactInt4", resources: [.copy("Kernels")]),
        .executableTarget(name: "CompactInt4Proof", dependencies: ["CompactInt4"]),
        .testTarget(name: "CompactInt4Tests", dependencies: ["CompactInt4"]),
        .executableTarget(
            name: "CompactInt4RuntimeScreen",
            dependencies: ["CompactInt4", .product(name: "MLX", package: "mlx-swift")],
            path: "Benchmarks/CompactInt4RuntimeScreen"
        ),
        .target(
            name: "AfterglowConsole",
            dependencies: [.product(name: "loom", package: "loom"), .product(name: "weft", package: "weft")]),
        .testTarget(
            name: "AfterglowConsoleTests", dependencies: ["AfterglowConsole", .product(name: "loom", package: "loom")]),
        .target(name: "ModelEvaluation"),
        .testTarget(name: "ModelEvaluationTests", dependencies: ["ModelEvaluation"]),
        .target(name: "DecisionModels"),
        .target(
            name: "DecisionMLX",
            dependencies: [
                "DecisionModels", .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"), .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]),
        .executableTarget(
            name: "AfterglowCLI",
            dependencies: [
                "DecisionModels", "DecisionMLX", "QuantizationCommands", "AfterglowConsole", "ModelEvaluation",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]),
        .testTarget(name: "DecisionModelsTests", dependencies: ["DecisionModels"]),
        .testTarget(
            name: "DecisionMLXTests",
            dependencies: [
                "DecisionModels", "DecisionMLX", "QuantizationCommands", .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"), .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]),
        .target(
            name: "WickModelSupport",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            swiftSettings: backendSwiftSettings
        ),
        .target(name: "WickQualitySupport"),
        .target(name: "QuantizerSupport"),
        .testTarget(name: "QuantizerSupportTests", dependencies: ["QuantizerSupport"]),
        .executableTarget(
            name: "MetalQuantizationBenchmark",
            dependencies: [
                "WickModelSupport",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            path: "Benchmarks/MetalQuantization"
        ),
        .executableTarget(
            name: "LagunaQuantizer",
            dependencies: [
                "QuantizerSupport",
                "WickModelSupport",
                "ScalePlanMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .target(
            name: "ScalePlanMLX"
        ),
        .executableTarget(
            name: "ScalePlanCLI",
            dependencies: [
                "ScalePlanMLX",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .executableTarget(
            name: "Q4ScaleSearchAudit",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .target(
            name: "LagunaScaleSearchCore",
            dependencies: [
                "MistralActivationScaleSearchCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/LagunaScaleSearchRescorer"
        ),
        .executableTarget(
            name: "LagunaScaleSearchRescorerCLI",
            dependencies: [
                "LagunaScaleSearchCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .executableTarget(
            name: "ModelQuantizer",
            dependencies: ["QuantizationCommands", .product(name: "ArgumentParser", package: "swift-argument-parser")]),
        .target(
            name: "QuantizationCommands",
            dependencies: [
                "QuantizerSupport",
                "LagunaScaleSearchCore",
                "WickModelSupport",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .executableTarget(
            name: "LagunaQ4R8Verifier",
            dependencies: [
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ]
        ),
        .target(
            name: "GemmaActivationQuantizerCore",
            dependencies: [
                "MistralActivationScaleSearchCore", "WickModelSupport", "QuantizerSupport",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]
        ),
        .executableTarget(
            name: "GemmaActivationQuantizer",
            dependencies: [
                "GemmaActivationQuantizerCore", "WickModelSupport", "QuantizerSupport",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: backendSwiftSettings
        ),
        .executableTarget(
            name: "GemmaActivationStats",
            dependencies: [
                "MistralActivationScaleSearchCore",
                "WickModelSupport",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: backendSwiftSettings
        ),
        .executableTarget(
            name: "MistralActivationStats",
            dependencies: [
                "MistralActivationScaleSearchCore",
                "WickQualitySupport",
                "WickModelSupport",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: backendSwiftSettings
        ),
        .target(
            name: "MistralActivationScaleSearchCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift")
            ]
        ),
        .executableTarget(
            name: "MistralActivationScaleSearchRescorer",
            dependencies: [
                "MistralActivationScaleSearchCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .testTarget(
            name: "ScalePlanMLXTests",
            dependencies: ["ScalePlanMLX"]
        ),
        .testTarget(
            name: "ModelQuantizerTests",
            dependencies: [
                "GemmaActivationQuantizerCore",
                "QuantizerSupport",
                "MistralActivationScaleSearchCore",
                "LagunaScaleSearchCore",
                "WickModelSupport",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]
        ),
        .testTarget(
            name: "MistralActivationScaleSearchCoreTests",
            dependencies: [
                "MistralActivationScaleSearchCore",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
    ],
    swiftLanguageModes: [.v6]
)
