// swift-tools-version: 6.4

import PackageDescription

// Reviewed backend revisions are pinned for reproducible Afterglow builds.
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
    products: [.executable(name: "midnight-afterglow", targets: ["AfterglowCLI"])],
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
            name: "AfterglowModelSupport",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            swiftSettings: backendSwiftSettings
        ),
        .target(name: "QuantizerSupport"),
        .testTarget(name: "QuantizerSupportTests", dependencies: ["QuantizerSupport"]),
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
        .target(
            name: "QuantizationCommands",
            dependencies: [
                "QuantizerSupport",
                "LagunaScaleSearchCore",
                "AfterglowModelSupport",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .target(
            name: "MistralActivationScaleSearchCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift")
            ]
        ),
        .testTarget(
            name: "ModelQuantizerTests",
            dependencies: [
                "QuantizerSupport",
                "MistralActivationScaleSearchCore",
                "LagunaScaleSearchCore",
                "AfterglowModelSupport",
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
