import Foundation

/// JSON values preserve argument types when scoring tool calls and structured answers.
public enum EvaluationJSON: Codable, Equatable, Sendable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([EvaluationJSON])
    case object([String: EvaluationJSON])

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() {
            self = .null
        } else if let v = try? c.decode(Bool.self) {
            self = .bool(v)
        } else if let v = try? c.decode(Double.self) {
            self = .number(v)
        } else if let v = try? c.decode(String.self) {
            self = .string(v)
        } else if let v = try? c.decode([EvaluationJSON].self) {
            self = .array(v)
        } else {
            self = .object(try c.decode([String: EvaluationJSON].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let v): try c.encode(v)
        case .number(let v): try c.encode(v)
        case .string(let v): try c.encode(v)
        case .array(let v): try c.encode(v)
        case .object(let v): try c.encode(v)
        }
    }

    public static func parse(_ text: String) throws -> Self {
        try JSONDecoder().decode(Self.self, from: Data(text.utf8))
    }
}

public struct EvaluationTool: Codable, Sendable {
    public var name: String
    public var description: String
    public var parameters: EvaluationJSON
}

public struct ExpectedToolCall: Codable, Sendable {
    public var name: String
    public var arguments: EvaluationJSON
    public var result: String
}

/// Expectations and tool results stay out of model requests until their turn is reached.
public struct EvaluationTurn: Codable, Sendable {
    public var calls: [ExpectedToolCall]
    public var answer: EvaluationJSON?
    public var pythonTests: String?
}

public struct EvaluationCase: Codable, Sendable {
    public var id: String
    public var category: String
    public var sourceFamily: String
    public var prompt: String
    public var tools: [EvaluationTool]
    public var turns: [EvaluationTurn]
}

public struct EvaluationSuite: Codable, Sendable {
    public var version: Int
    public var name: String
    public var split: String
    public var cases: [EvaluationCase]

    public func validate() throws {
        guard version == 1, !name.isEmpty, !cases.isEmpty,
            ["development", "holdout"].contains(split),
            Set(cases.map(\.id)).count == cases.count
        else {
            throw EvaluationFailure.invalidSuite("Version, name, split, nonempty cases and unique IDs are required.")
        }
        for c in cases {
            guard !c.id.isEmpty, !c.category.isEmpty, !c.sourceFamily.isEmpty, !c.prompt.isEmpty,
                !c.turns.isEmpty, c.turns.count <= 16,
                Set(c.tools.map(\.name)).count == c.tools.count
            else { throw EvaluationFailure.invalidSuite("Invalid case \(c.id).") }
            for (index, t) in c.turns.enumerated() {
                let toolTurn = !t.calls.isEmpty && t.answer == nil && t.pythonTests == nil
                let answerTurn = t.calls.isEmpty && ((t.answer != nil) != (t.pythonTests != nil))
                guard toolTurn || answerTurn,
                    t.calls.allSatisfy({ call in c.tools.contains { $0.name == call.name } }),
                    index == c.turns.count - 1 || toolTurn
                else { throw EvaluationFailure.invalidSuite("Invalid expectation in \(c.id), turn \(index).") }
            }
            guard c.turns.last?.calls.isEmpty == true else {
                throw EvaluationFailure.invalidSuite("Case \(c.id) must end with a scored answer.")
            }
        }
    }
}

public enum EvaluationFailure: Error, LocalizedError {
    case invalidSuite(String)
    case transport(String)
    case sandbox(String)
    public var errorDescription: String? {
        switch self {
        case .invalidSuite(let text), .transport(let text), .sandbox(let text): return text
        }
    }
}

public struct EvaluationMessage: Codable, Sendable {
    public var role: String
    public var content: String?
    public var toolCalls: [EvaluationCall]?
    public var toolCallID: String?

    enum CodingKeys: String, CodingKey {
        case role, content
        case toolCalls = "tool_calls"
        case toolCallID = "tool_call_id"
    }

    public init(role: String, content: String? = nil, calls: [EvaluationCall]? = nil, callID: String? = nil) {
        self.role = role
        self.content = content
        toolCalls = calls
        toolCallID = callID
    }
}

public struct EvaluationCall: Codable, Sendable {
    public var id: String
    public var type: String
    public var function: Function
    public struct Function: Codable, Sendable {
        public var name: String
        public var arguments: String
        public init(name: String, arguments: String) {
            self.name = name
            self.arguments = arguments
        }
    }
    public init(id: String, name: String, arguments: String) {
        self.id = id
        type = "function"
        function = Function(name: name, arguments: arguments)
    }
}

public struct EvaluationResponse: Codable, Sendable {
    public var message: EvaluationMessage
    public var finishReason: String
    public var completionTokens: Int?
    public var elapsedSeconds: Double
    public init(
        message: EvaluationMessage, finishReason: String, completionTokens: Int? = nil, elapsedSeconds: Double = 0
    ) {
        self.message = message
        self.finishReason = finishReason
        self.completionTokens = completionTokens
        self.elapsedSeconds = elapsedSeconds
    }
}

public protocol EvaluationTransport: Sendable {
    func complete(messages: [EvaluationMessage], tools: [EvaluationTool]) async throws -> EvaluationResponse
}

public protocol EvaluationCodeSandbox: Sendable {
    func check(code: String, tests: String) async throws -> Bool
}

public struct EvaluationCaseResult: Codable, Sendable {
    public var id: String
    public var category: String
    public var sourceFamily: String
    public var status: String
    public var reason: String
    public var responses: [EvaluationResponse]
}

public struct EvaluationReport: Codable, Sendable {
    public var suiteName: String
    public var split: String
    public var results: [EvaluationCaseResult]
    public var categoryScores: [String: Score]
    public struct Score: Codable, Sendable {
        public var passed: Int
        public var failed: Int
        public var errors: Int
        public var total: Int { passed + failed + errors }
    }
}
