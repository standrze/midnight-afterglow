import Foundation
import MLX
import XCTest

@testable import MistralActivationScaleSearchCore

final class GemmaActivationStatisticsTests: XCTestCase {
    func testRoundTripRequiresCurrentSourceAndIndependentInventory() throws {
        try Device.withDefaultDevice(.cpu) {
            let fixture = try Fixture()
            defer { fixture.remove() }
            let provenance = try fixture.provenance()
            let statistics = try fixture.statistics(provenance: provenance)
            let destination = fixture.root.appendingPathComponent("stats.safetensors")
            try statistics.write(to: destination)
            let bytes = try Data(contentsOf: destination)
            let loaded = try GemmaActivationStatistics.load(
                from: destination, source: fixture.source, expectedProjectionShapes: fixture.shapes,
                expectedExpertsPerToken: 1, expectedMinimumExpertPositions: 3)
            XCTAssertEqual(loaded.provenance, provenance)
            XCTAssertEqual(loaded.expertCounts, [fixture.expertPath: [3, 2, 0]])
            XCTAssertEqual(loaded.minimumExpertPositions, 3)
            for path in fixture.shapes.keys {
                XCTAssertEqual(loaded.moments[path]?.asArray(Float.self), statistics.moments[path]?.asArray(Float.self))
            }
            let permissions =
                try FileManager.default.attributesOfItem(atPath: destination.path)[.posixPermissions] as? NSNumber
            XCTAssertEqual(permissions?.intValue, 0o600)
            XCTAssertThrowsError(try statistics.write(to: destination))
            XCTAssertEqual(try Data(contentsOf: destination), bytes)
            XCTAssertThrowsError(try statistics.write(to: fixture.source.appendingPathComponent("stats.safetensors")))
            XCTAssertThrowsError(
                try GemmaActivationStatistics.load(
                    from: destination, source: fixture.source, expectedProjectionShapes: [fixture.densePath: [2]],
                    expectedExpertsPerToken: 1, expectedMinimumExpertPositions: 3))
            XCTAssertThrowsError(
                try GemmaActivationStatistics.load(
                    from: destination, source: fixture.source, expectedProjectionShapes: fixture.shapes,
                    expectedExpertsPerToken: 2, expectedMinimumExpertPositions: 3))
            XCTAssertThrowsError(
                try GemmaActivationStatistics.load(
                    from: destination, source: fixture.source, expectedProjectionShapes: fixture.shapes,
                    expectedExpertsPerToken: 1, expectedMinimumExpertPositions: 2))
            for filename in ["model.safetensors", "config.json", "tokenizer.json"] {
                let url = fixture.source.appendingPathComponent(filename)
                let original = try Data(contentsOf: url)
                var changed = original
                if filename.hasSuffix("safetensors") { changed[changed.count - 1] ^= 1 } else { changed.append(32) }
                try changed.write(to: url)
                XCTAssertThrowsError(
                    try GemmaActivationStatistics.load(
                        from: destination, source: fixture.source, expectedProjectionShapes: fixture.shapes,
                        expectedExpertsPerToken: 1, expectedMinimumExpertPositions: 3))
                XCTAssertThrowsError(
                    try statistics.write(to: fixture.root.appendingPathComponent("changed.safetensors")))
                XCTAssertFalse(
                    FileManager.default.fileExists(
                        atPath: fixture.root.appendingPathComponent("changed.safetensors").path))
                try original.write(to: url)
            }
            let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: destination)
            var incomplete = arrays
            incomplete.removeValue(forKey: fixture.densePath + GemmaActivationStatistics.momentSuffix)
            let malformed = fixture.root.appendingPathComponent("malformed.safetensors")
            try MLX.save(arrays: incomplete, metadata: metadata, url: malformed)
            XCTAssertThrowsError(
                try GemmaActivationStatistics.load(
                    from: malformed, source: fixture.source, expectedProjectionShapes: fixture.shapes,
                    expectedExpertsPerToken: 1, expectedMinimumExpertPositions: 3))
            var manifest = try XCTUnwrap(
                try JSONSerialization.jsonObject(
                    with: Data(try XCTUnwrap(metadata["gemma_activation_manifest"]).utf8)) as? [String: Any])
            manifest["algorithm"] = "uniform_dense_for_experts"
            let unsupportedMetadata = [
                "gemma_activation_manifest": String(
                    decoding: try JSONSerialization.data(withJSONObject: manifest), as: UTF8.self)
            ]
            try MLX.save(arrays: arrays, metadata: unsupportedMetadata, url: malformed)
            XCTAssertThrowsError(
                try GemmaActivationStatistics.load(
                    from: malformed, source: fixture.source, expectedProjectionShapes: fixture.shapes,
                    expectedExpertsPerToken: 1, expectedMinimumExpertPositions: 3))
            XCTAssertFalse(
                try FileManager.default.contentsOfDirectory(atPath: fixture.root.path).contains {
                    $0.hasPrefix(".gemma-stats-")
                })
        }
    }

    func testRejectsInvalidMomentsCoverageAndTokenPartitions() throws {
        try Device.withDefaultDevice(.cpu) {
            let fixture = try Fixture()
            defer { fixture.remove() }
            let source = try GemmaActivationSourceIdentity.capture(source: fixture.source)
            for segments in [[[1, 2, 3, 4, 5]], [[2, 1], [3], [4, 5]], [[1, 2], [3]], [[]], [[-1]]] {
                XCTAssertThrowsError(
                    try GemmaActivationProvenance(
                        source: source, corpus: Data("fit".utf8), tokenSamples: [[1, 2, 3], [4, 5]],
                        tokenSegments: segments, sourceFamilies: ["fit-family"]))
            }
            let provenance = try fixture.provenance()
            for values in [[Float.nan, Float(1)], [Float(-1), Float(1)]] {
                var moments = fixture.moments
                moments[fixture.densePath] = MLXArray(values)
                XCTAssertThrowsError(
                    try GemmaActivationStatistics(
                        moments: moments, expertCounts: [fixture.expertPath: [3, 2, 0]],
                        provenance: provenance, minimumExpertPositions: 3, expertsPerToken: 1))
            }
            for counts in [[2, 2, 0], [6, 0, 0], [3, 2], [-1, 6, 0]] {
                XCTAssertThrowsError(
                    try GemmaActivationStatistics(
                        moments: fixture.moments, expertCounts: [fixture.expertPath: counts],
                        provenance: provenance, minimumExpertPositions: 3, expertsPerToken: 1))
            }
            var nonzeroUnobserved = fixture.moments
            nonzeroUnobserved[fixture.expertPath] = MLXArray([Float(1), 2, 2, 4, 1, 0]).reshaped(3, 2)
            XCTAssertThrowsError(
                try GemmaActivationStatistics(
                    moments: nonzeroUnobserved, expertCounts: [fixture.expertPath: [3, 2, 0]],
                    provenance: provenance, minimumExpertPositions: 3, expertsPerToken: 1))
            XCTAssertThrowsError(
                try GemmaActivationStatistics(
                    moments: fixture.moments, expertCounts: [:], provenance: provenance,
                    minimumExpertPositions: 3, expertsPerToken: 1))
        }
    }

    func testFitDevelopmentRejectsCorpusSampleSegmentAndFamilyOverlap() throws {
        try Device.withDefaultDevice(.cpu) {
            let fixture = try Fixture()
            defer { fixture.remove() }
            let source = try GemmaActivationSourceIdentity.capture(source: fixture.source)
            func record(_ corpus: String, _ samples: [[Int]], _ segments: [[Int]], _ families: [String]) throws
                -> GemmaActivationProvenance
            {
                try GemmaActivationProvenance(
                    source: source, corpus: Data(corpus.utf8), tokenSamples: samples,
                    tokenSegments: segments, sourceFamilies: families)
            }
            let fit = try record("fit", [[1, 2, 3], [4, 5]], [[1, 2], [3], [4, 5]], ["fit-family"])
            let dev = try record("dev", [[6, 7]], [[6, 7]], ["dev-family"])
            XCTAssertNoThrow(try fit.requireDisjoint(from: dev))
            XCTAssertThrowsError(try fit.requireDisjoint(from: record("fit", [[6, 7]], [[6, 7]], ["dev-family"])))
            XCTAssertThrowsError(
                try fit.requireDisjoint(from: record("dev", [[1, 2, 3]], [[1], [2, 3]], ["dev-family"])))
            XCTAssertThrowsError(
                try fit.requireDisjoint(from: record("dev", [[1, 2, 9]], [[1, 2], [9]], ["dev-family"])))
            XCTAssertThrowsError(try fit.requireDisjoint(from: record("dev", [[6, 7]], [[6, 7]], [" fit-family "])))
        }
    }

    private struct Fixture {
        let root: URL
        let source: URL
        let densePath = "model.layers.0.self_attn.q_proj"
        let expertPath = "model.layers.0.experts.switch_glu.gate_proj"
        var shapes: [String: [Int]] { [densePath: [2], expertPath: [3, 2]] }
        var moments: [String: MLXArray] {
            [densePath: MLXArray([Float(2), 3]), expertPath: MLXArray([Float(1), 2, 2, 4, 0, 0]).reshaped(3, 2)]
        }

        init() throws {
            root = FileManager.default.temporaryDirectory.appendingPathComponent("gemma-stats-\(UUID().uuidString)")
            source = root.appendingPathComponent("source")
            try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
            try Data(#"{"model_type":"gemma4_text"}"#.utf8).write(to: source.appendingPathComponent("config.json"))
            try Data("{}".utf8).write(to: source.appendingPathComponent("tokenizer.json"))
            try MLX.save(
                arrays: ["fixture.weight": MLXArray([Float(1), 2]).asType(.bfloat16)],
                url: source.appendingPathComponent("model.safetensors"))
            try Data(#"{"weight_map":{"fixture.weight":"model.safetensors"}}"#.utf8).write(
                to: source.appendingPathComponent("model.safetensors.index.json"))
        }

        func provenance() throws -> GemmaActivationProvenance {
            try GemmaActivationProvenance(
                source: GemmaActivationSourceIdentity.capture(source: source), corpus: Data("fit".utf8),
                tokenSamples: [[1, 2, 3], [4, 5]], tokenSegments: [[1, 2], [3], [4, 5]], sourceFamilies: ["fit-family"])
        }

        func statistics(provenance: GemmaActivationProvenance) throws -> GemmaActivationStatistics {
            try GemmaActivationStatistics(
                moments: moments, expertCounts: [expertPath: [3, 2, 0]], provenance: provenance,
                minimumExpertPositions: 3, expertsPerToken: 1)
        }

        func remove() { try? FileManager.default.removeItem(at: root) }
    }
}
