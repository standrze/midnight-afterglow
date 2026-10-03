// Checkpoint and FP32 Adam patterns adapted from Training's NativeTrainer.swift.
// Original attribution and source hashes are recorded in Docs/import-manifest.json.
import DecisionModels
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXOptimizers

public struct DecisionTrainingOptions: Codable, Sendable {
    public var rank: Int = 16
    public var learningRate: Float = 5e-5
    public var batchSize: Int = 8
    public var epochs: Int = 1
    public var seed: UInt64 = 17
    public var maxLength: Int = 2048
    public var checkpointEvery: Int = 10
    public var initializationAdapter: String? = nil
    public init() {}
}

public struct DecisionTrainingReport: Codable, Sendable {
    public var updates: Int
    public var initialLoss: Float
    public var finalLoss: Float
    public var nonzeroGradientSteps: Int
    public var peakMemoryBytes: Int
}

public enum DecisionTrainer {
    private struct Row {
        let tokens: [Int]
        let candidates: [Int]
        let gold: Int
    }

    private struct State: Codable {
        var format = 1
        var identity: String
        var options: DecisionTrainingOptions
        var completed: Int
        var cursor: Int
        var order: [Int]
        var randomState: UInt64
        var totalUpdates: Int
        var nonzeroSteps: Int
        var initialLoss: Float
    }

    public static func train(
        container: ModelContainer, model: URL, data: URL, development: URL,
        contract: DecisionModelContract, options: DecisionTrainingOptions, output: URL,
        resume: Bool = false, stopAfterUpdates: Int? = nil
    ) async throws -> DecisionTrainingReport {
        guard options.rank > 0, options.learningRate.isFinite, options.learningRate > 0,
            options.batchSize > 0, options.epochs > 0, options.maxLength > 0,
            options.maxLength <= contract.maxLength, options.checkpointEvery > 0,
            stopAfterUpdates.map({ $0 > 0 }) ?? true
        else {
            throw DecisionError.invalidRequest(
                "Training sizes, rank, epochs and learning rate must be positive and within the contract.")
        }
        let training = try DecisionDataset.read(data, contract: contract)
        let validation = try DecisionDataset.read(development, contract: contract)
        try DecisionDataset.validateSplits([training, validation])
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let baseFingerprint = try DecisionFiles.modelHash(model)
        let identity =
            try baseFingerprint + ":" + DecisionFiles.hash(data) + ":"
            + DecisionFiles.hash(development) + ":" + String(decoding: encoder.encode(contract), as: UTF8.self)
            + ":"
            + (options.initializationAdapter.map { try DecisionFiles.modelHash(URL(fileURLWithPath: $0)) } ?? "fresh")
        if !resume, FileManager.default.fileExists(atPath: output.path) {
            throw DecisionError.invalidRequest("Use a new run directory or explicitly resume.")
        }
        return try await container.perform { context in
            try GatedDeltaExecution.$useMetalKernel.withValue(false) {
                try GatedDeltaExecution.$checkpointTraining.withValue(true) {
                    let model = context.model
                    guard let adaptable = model as? LoRAModel else {
                        throw DecisionError.unsupported("Model cannot train LoRA adapters.")
                    }
                    var trainingContract = contract
                    trainingContract.maxLength = options.maxLength
                    let rows = try training.flatMap { example in
                        try DecisionRuntime.prepare(
                            request: example.request, contract: trainingContract, tokenizer: context.tokenizer
                        ).map { row in
                            let values = example.request.schema[row.name]!.values
                            let gold = values.firstIndex { $0.key == example.labels[row.name]!.key }!
                            return Row(tokens: row.tokens, candidates: row.candidates, gold: gold)
                        }
                    }
                    let devRows = try validation.flatMap { example in
                        try DecisionRuntime.prepare(
                            request: example.request, contract: trainingContract, tokenizer: context.tokenizer
                        ).map { row in
                            Row(
                                tokens: row.tokens, candidates: row.candidates,
                                gold: example.request.schema[row.name]!.values.firstIndex {
                                    $0.key == example.labels[row.name]!.key
                                }!)
                        }
                    }
                    let configuration: LoRAConfiguration
                    MLXRandom.seed(options.seed)
                    if let initial = options.initializationAdapter {
                        let adapter = try DecisionRuntime.loadAdapter(URL(fileURLWithPath: initial))
                        configuration = adapter.configuration
                        try adapter.load(into: model)
                        try DecisionRuntime.auditAdapter(adapter, model: model)
                        model.freeze()
                        model.unfreeze(keys: ["lora_a", "lora_b"])
                    } else {
                        configuration = LoRAConfiguration(
                            numLayers: adaptable.loraLayers.count,
                            loraParameters: .init(
                                rank: options.rank, scale: 2, dropout: 0, keys: adaptable.loraDefaultKeys)
                        )
                        _ = try LoRAContainer.from(model: model, configuration: configuration)
                    }
                    let parameters = Dictionary(uniqueKeysWithValues: model.trainableParameters().flattened())
                    guard !parameters.isEmpty,
                        parameters.keys.allSatisfy({ $0.hasSuffix(".lora_a") || $0.hasSuffix(".lora_b") })
                    else {
                        throw DecisionError.invalidContract("Only LoRA tensors may be trainable.")
                    }
                    var optimizer = DecisionAdam()
                    var state: State
                    if resume {
                        let pointer = try JSONDecoder().decode(
                            String.self, from: Data(contentsOf: output.appendingPathComponent("latest-checkpoint.json"))
                        )
                        guard !pointer.contains("/"), pointer.hasPrefix("checkpoint-") else {
                            throw DecisionError.invalidContract("Unsafe checkpoint pointer.")
                        }
                        let checkpoint = output.appendingPathComponent(pointer)
                        state = try JSONDecoder().decode(
                            State.self, from: Data(contentsOf: checkpoint.appendingPathComponent("state.json")))
                        guard state.format == 1, state.identity == identity,
                            state.totalUpdates == options.epochs
                                * ((rows.count + options.batchSize - 1) / options.batchSize),
                            try encoder.encode(state.options) == encoder.encode(options),
                            state.order.sorted() == Array(rows.indices), (0...rows.count).contains(state.cursor),
                            (0...state.totalUpdates).contains(state.completed)
                        else {
                            throw DecisionError.invalidContract("Resume model, data, options or cursor differs.")
                        }
                        let arrays = try loadArrays(url: checkpoint.appendingPathComponent("training.safetensors"))
                        var adapter: [String: MLXArray] = [:]
                        for key in parameters.keys {
                            guard let value = arrays["adapter." + key], value.shape == parameters[key]!.shape,
                                let first = arrays["first." + key], first.shape == value.shape,
                                let second = arrays["second." + key], second.shape == value.shape
                            else {
                                throw DecisionError.invalidContract("Incomplete checkpoint tensor state.")
                            }
                            adapter[key] = value
                            optimizer.first[key] = first
                            optimizer.second[key] = second
                        }
                        try model.update(parameters: .unflattened(adapter), verify: [.noUnusedKeys, .shapeMismatch])
                        optimizer.step = state.completed
                    } else {
                        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
                        var order = Array(rows.indices)
                        var random = options.seed
                        shuffle(&order, state: &random)
                        model.train(false)
                        state = State(
                            identity: identity, options: options, completed: 0, cursor: 0,
                            order: order, randomState: random,
                            totalUpdates: options.epochs * ((rows.count + options.batchSize - 1) / options.batchSize),
                            nonzeroSteps: 0, initialLoss: try evaluate(model: model, rows: devRows))
                        try checkpoint(model: model, optimizer: optimizer, state: state, root: output)
                    }
                    defer { model.train(false) }
                    model.train(true)
                    let module: Module = model
                    let gradient = valueAndGrad(model: module) { module, arrays in
                        let model = module as! any LanguageModel
                        let logits = model(arrays[0], cache: nil)[0, -1, 0...].asType(.float32)
                        let selected = take(logits, arrays[1], axis: 0).reshaped(1, -1)
                        return [crossEntropy(logits: selected, targets: arrays[2]).mean()]
                    }
                    let limit = min(state.totalUpdates, stopAfterUpdates ?? state.totalUpdates)
                    while state.completed < limit {
                        try Task.checkCancellation()
                        if state.cursor == rows.count {
                            shuffle(&state.order, state: &state.randomState)
                            state.cursor = 0
                        }
                        let end = min(state.cursor + options.batchSize, rows.count)
                        var sums: [String: MLXArray] = [:]
                        for position in state.cursor..<end {
                            let row = rows[state.order[position]]
                            MLXRandom.seed(
                                options.seed
                                    &+ UInt64(state.completed * options.batchSize + position - state.cursor + 1))
                            let (values, gradients) = gradient(
                                module,
                                [MLXArray(row.tokens).reshaped(1, -1), MLXArray(row.candidates), MLXArray([row.gold])])
                            eval(values, gradients)
                            guard values[0].item(Float.self).isFinite else {
                                throw DecisionError.numericalFailure("Non-finite candidate loss.")
                            }
                            for (key, value) in gradients.flattened() {
                                sums[key] =
                                    (sums[key] ?? MLXArray.zeros(like: value)) + value.asType(.float32)
                                    / Float(end - state.cursor)
                            }
                        }
                        let (clipped, norm) = clipGradNorm(gradients: .unflattened(sums), maxNorm: 1)
                        eval(norm)
                        let magnitude = norm.item(Float.self)
                        guard magnitude.isFinite, magnitude > 0 else {
                            throw DecisionError.numericalFailure("Non-finite or zero gradient.")
                        }
                        // Preserve a three-epoch linear schedule while stopping after the selected epochs.
                        let schedule = max(1, state.totalUpdates * 3 / options.epochs)
                        let warmup = max(1, schedule / 10)
                        let step = state.completed + 1
                        let factor =
                            step <= warmup
                            ? Float(step) / Float(warmup)
                            : max(0, Float(schedule - step) / Float(max(1, schedule - warmup)))
                        try optimizer.update(
                            model: module, gradients: clipped, learningRate: options.learningRate * factor)
                        eval(model, Array(optimizer.first.values), Array(optimizer.second.values))
                        state.completed += 1
                        state.cursor = end
                        state.nonzeroSteps += 1
                        if state.completed % options.checkpointEvery == 0 || state.completed == limit {
                            try checkpoint(model: model, optimizer: optimizer, state: state, root: output)
                            print(
                                "decision training update \(state.completed)/\(state.totalUpdates), gradient \(magnitude)"
                            )
                        }
                    }
                    model.train(false)
                    let report = DecisionTrainingReport(
                        updates: state.completed, initialLoss: state.initialLoss,
                        finalLoss: try evaluate(model: model, rows: devRows), nonzeroGradientSteps: state.nonzeroSteps,
                        peakMemoryBytes: Memory.peakMemory)
                    try DecisionFiles.transaction(
                        destination: output.appendingPathComponent("adapter-step-\(state.completed)")
                    ) { directory in
                        try save(
                            arrays: Dictionary(uniqueKeysWithValues: model.trainableParameters().flattened()),
                            url: directory.appendingPathComponent("adapters.safetensors"))
                        try DecisionFiles.write(
                            configuration, to: directory.appendingPathComponent("adapter_config.json"))
                        var trained = contract
                        trained.temperature = 1
                        trained.adapterSHA256 = nil
                        trained.sourceRevision = nil
                        trained.baseFingerprint = baseFingerprint
                        trained.adapterFingerprint = try DecisionFiles.hash(
                            directory.appendingPathComponent("adapters.safetensors"))
                        try DecisionFiles.write(
                            trained, to: directory.appendingPathComponent(DecisionModelContract.filename))
                        try DecisionFiles.write(report, to: directory.appendingPathComponent("training-report.json"))
                    }
                    try DecisionFiles.write(report, to: output.appendingPathComponent("training-report.json"))
                    return report
                }
            }
        }
    }

    private static func evaluate(model: any LanguageModel, rows: [Row]) throws -> Float {
        var sum: Float = 0
        for row in rows {
            try Task.checkCancellation()
            let logits = model(MLXArray(row.tokens).reshaped(1, -1), cache: nil)[0, -1, 0...].asType(.float32)
            let loss = crossEntropy(
                logits: take(logits, MLXArray(row.candidates), axis: 0).reshaped(1, -1), targets: MLXArray([row.gold])
            ).mean()
            eval(loss)
            sum += loss.item(Float.self)
        }
        guard sum.isFinite else { throw DecisionError.numericalFailure("Non-finite evaluation loss.") }
        return sum / Float(rows.count)
    }

    private static func checkpoint(model: any LanguageModel, optimizer: DecisionAdam, state: State, root: URL) throws {
        let name = "checkpoint-\(state.completed)-\(UUID().uuidString)"
        try DecisionFiles.transaction(destination: root.appendingPathComponent(name)) { directory in
            var arrays: [String: MLXArray] = [:]
            for (key, value) in model.trainableParameters().flattened() {
                arrays["adapter." + key] = value
                arrays["first." + key] = optimizer.first[key] ?? MLXArray.zeros(like: value).asType(.float32)
                arrays["second." + key] = optimizer.second[key] ?? MLXArray.zeros(like: value).asType(.float32)
            }
            try save(arrays: arrays, url: directory.appendingPathComponent("training.safetensors"))
            try DecisionFiles.write(state, to: directory.appendingPathComponent("state.json"))
        }
        try DecisionFiles.write(name, to: root.appendingPathComponent("latest-checkpoint.json"))
    }

    private static func shuffle(_ order: inout [Int], state: inout UInt64) {
        guard order.count > 1 else { return }
        for index in stride(from: order.count - 1, through: 1, by: -1) {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            order.swapAt(index, Int(state % UInt64(index + 1)))
        }
    }
}

private struct DecisionAdam {
    var step = 0
    var first: [String: MLXArray] = [:]
    var second: [String: MLXArray] = [:]

    mutating func update(model: Module, gradients: ModuleParameters, learningRate: Float) throws {
        let parameters = Dictionary(uniqueKeysWithValues: model.trainableParameters().flattened())
        let gradients = Dictionary(uniqueKeysWithValues: gradients.flattened())
        guard Set(parameters.keys) == Set(gradients.keys) else {
            throw DecisionError.numericalFailure("Gradient keys differ from adapter keys.")
        }
        step += 1
        var updated: [String: MLXArray] = [:]
        for key in parameters.keys.sorted() {
            let gradient = gradients[key]!.asType(.float32)
            let m = 0.9 * (first[key] ?? MLXArray.zeros(like: gradient)) + 0.1 * gradient
            let v = 0.999 * (second[key] ?? MLXArray.zeros(like: gradient)) + 0.001 * square(gradient)
            first[key] = m
            second[key] = v
            let corrected =
                (m / (1 - Foundation.pow(0.9, Float(step))))
                / (sqrt(v / (1 - Foundation.pow(0.999, Float(step)))) + 1e-8)
            updated[key] = (parameters[key]!.asType(.float32) - learningRate * corrected).asType(parameters[key]!.dtype)
        }
        try model.update(parameters: .unflattened(updated), verify: [.noUnusedKeys, .shapeMismatch])
    }
}
