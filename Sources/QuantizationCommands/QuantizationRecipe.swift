import ArgumentParser
import MLX
import MLXLMCommon

enum QuantizerMode: String, CaseIterable, ExpressibleByArgument {
    case affine
    case mxfp4
    case mxfp8
    case nvfp4

    var mlxMode: QuantizationMode {
        switch self {
        case .affine: .affine
        case .mxfp4: .mxfp4
        case .mxfp8: .mxfp8
        case .nvfp4: .nvfp4
        }
    }

    var defaultBits: Int { self == .mxfp8 ? 8 : 4 }

    var defaultGroupSize: Int {
        switch self {
        case .affine: 64
        case .mxfp4, .mxfp8: 32
        case .nvfp4: 16
        }
    }
}

enum QuantizerCalibration: String, ExpressibleByArgument {
    case standard
    case scaleSearch = "scale-search"
}

/// Resolves CLI geometry before any native quantization operation can assert.
struct QuantizationRecipe {
    var mode: QuantizerMode = .affine
    var bits = 4
    var groupSize = 64
    var calibration: QuantizerCalibration = .scaleSearch

    var mlxCalibration: ModelConversionQuantizationCalibration {
        !usesScaleSearch ? .standard : (bits == 5 ? .q5AffineScaleSearch : .q4AffineScaleSearch)
    }

    var usesScaleSearch: Bool { calibration == .scaleSearch }
    var isAffineQ4: Bool { mode == .affine && bits == 4 }
    var isStandardQ4: Bool { isAffineQ4 && !usesScaleSearch }
    var isStandardQ8: Bool { mode == .affine && bits == 8 && groupSize == 64 && !usesScaleSearch }
    var usesLegacyProvenance: Bool { groupSize == 64 && (isAffineQ4 || isStandardQ8) }

    static func resolve(
        mode: QuantizerMode?, bits: Int?, groupSize: Int?, calibration: QuantizerCalibration?,
        standardQ4: Bool, standardQ8: Bool
    ) throws -> Self {
        guard !standardQ4 || !standardQ8 else {
            throw ValidationError("--standard-q4 and --standard-q8 are mutually exclusive.")
        }
        if standardQ4 || standardQ8 {
            let aliasBits = standardQ8 ? 8 : 4
            guard mode == nil || mode == .affine, bits == nil || bits == aliasBits,
                calibration == nil || calibration == .standard
            else {
                throw ValidationError(
                    "--standard-q4/--standard-q8 conflict with the selected mode, bits, or calibration.")
            }
        }

        guard !standardQ8 || groupSize == nil || groupSize == 64 else {
            throw ValidationError("--standard-q8 requires group size 64; use --mode affine --bits 8 for other groups.")
        }

        let resolvedMode = mode ?? .affine
        let resolvedBits = bits ?? (standardQ8 ? 8 : resolvedMode.defaultBits)
        let resolvedGroupSize = groupSize ?? resolvedMode.defaultGroupSize
        let resolvedCalibration =
            calibration ?? ((mode != nil || bits != nil || standardQ4 || standardQ8) ? .standard : .scaleSearch)

        if resolvedMode == .affine {
            guard [2, 3, 4, 5, 6, 8].contains(resolvedBits), [32, 64, 128].contains(resolvedGroupSize) else {
                throw ValidationError("affine supports --bits 2, 3, 4, 5, 6, or 8 and --group-size 32, 64, or 128.")
            }
        } else {
            guard resolvedBits == resolvedMode.defaultBits, resolvedGroupSize == resolvedMode.defaultGroupSize else {
                throw ValidationError(
                    "\(resolvedMode.rawValue) requires --bits \(resolvedMode.defaultBits) and --group-size \(resolvedMode.defaultGroupSize)."
                )
            }
        }
        if resolvedCalibration == .scaleSearch {
            let supportedGroups = resolvedBits == 4 ? [32, 64, 128] : [64, 128]
            guard resolvedMode == .affine, [4, 5].contains(resolvedBits), supportedGroups.contains(resolvedGroupSize)
            else {
                throw ValidationError(
                    "--calibration scale-search requires affine Q4 with group size 32, 64, or 128; Q5 requires 64 or 128."
                )
            }
        }
        return Self(
            mode: resolvedMode, bits: resolvedBits, groupSize: resolvedGroupSize, calibration: resolvedCalibration)
    }
}

public struct QuantizationFormats: ParsableCommand {
    public init() {}
    public static let configuration = CommandConfiguration(
        commandName: "formats", abstract: "List MLX quantization formats, geometry, and calibration limits.")

    public func run() {
        print(
            """
            Mode     Bits          Group sizes  Default
            affine   2,3,4,5,6,8   32,64,128    4 bits / 64
            mxfp4    4             32           4 bits / 32
            mxfp8    8             32           8 bits / 32
            nvfp4    4             16           4 bits / 16

            --mode or --bits selects standard MLX quantization by default.
            Without either, the default is affine Q4/G64 ScaleSearch.
            --calibration scale-search supports affine Q4 with G32/G64/G128 and Q5 with G64/G128.
            Protected and requested Q8 modules always use standard affine Q8/G64.
            Module widths must be divisible by their selected group size; use --dry-run to check.
            Runtime and backend support vary. Conversion does not establish model quality.
            """)
    }
}
