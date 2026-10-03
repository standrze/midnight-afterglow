import Foundation

/// Exact Nimble text prompt, using Python JSON separators and escaped delimiters.
public enum DecisionPrompt {
    public static func text(request: DecisionRequest, field: String, contract: DecisionModelContract) throws -> String {
        try contract.validate()
        try request.validate(maximumChoices: contract.candidateCodes.count)
        guard request.schema[field] != nil else { throw DecisionError.invalidRequest("Unknown field.") }
        let definitions = request.names.map { name -> String in
            let definition = request.schema[name]!
            let choices = definition.values.enumerated().map { index, value -> String in
                var pairs = ["\"code\": " + quote(contract.candidateCodes[index]), "\"value\": " + json(value)]
                if let description = definition.choiceDescriptions?[value.key] {
                    pairs.append("\"description\": " + quote(description))
                }
                return "{" + pairs.joined(separator: ", ") + "}"
            }
            return "{\"name\": " + quote(name) + ", \"description\": " + quote(definition.description)
                + ", \"choices\": [" + choices.joined(separator: ", ") + "]}"
        }
        let context =
            "{\"context\": " + quote(request.context) + ", \"schema\": ["
            + definitions.joined(separator: ", ") + "]}\n\nRequested field: " + quote(field)
        let extended = request.schema.values.contains { $0.values.count > 26 }
        let system =
            extended
            ? DecisionModelContract.systemPrompt.replacingOccurrences(of: "one-letter", with: "short")
            : DecisionModelContract.systemPrompt
        return "<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n"
            + context + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }

    private static func json(_ value: DecisionValue) -> String {
        switch value {
        case .string(let text): quote(text)
        case .boolean(let flag): flag ? "true" : "false"
        case .integer(let number): String(number)
        }
    }

    private static func quote(_ text: String) -> String {
        // JSONEncoder may escape '/' and use different Unicode escaping; build Python's
        // ensure_ascii=False string representation directly to preserve prompt bytes.
        var result = "\""
        for scalar in text.unicodeScalars {
            switch scalar.value {
            case 34: result += "\\\""
            case 92: result += "\\\\"
            case 8: result += "\\b"
            case 9: result += "\\t"
            case 10: result += "\\n"
            case 12: result += "\\f"
            case 13: result += "\\r"
            case 60: result += "\\u003c"
            case 62: result += "\\u003e"
            case 0...31: result += String(format: "\\u%04x", scalar.value)
            default: result.unicodeScalars.append(scalar)
            }
        }
        return result + "\""
    }
}

/// Structural scan supplements JSONDecoder with duplicate detection and schema order.
struct DecisionJSONScanner {
    let data: Data
    private var bytes: [UInt8] = []
    private var cursor = 0
    private var schemaOrder: [String] = []

    init(data: Data) { self.data = data }

    mutating func scan() throws -> [String] {
        bytes = Array(data)
        try value(path: [], depth: 0)
        whitespace()
        guard cursor == bytes.count else { throw DecisionError.invalidRequest("Trailing JSON content.") }
        return schemaOrder
    }

    private mutating func whitespace() {
        while cursor < bytes.count && [9, 10, 13, 32].contains(bytes[cursor]) { cursor += 1 }
    }

    private mutating func consume(_ byte: UInt8) throws {
        whitespace()
        guard cursor < bytes.count && bytes[cursor] == byte else {
            throw DecisionError.invalidRequest("Malformed JSON.")
        }
        cursor += 1
    }

    private mutating func string() throws -> String {
        whitespace()
        let start = cursor
        try consume(34)
        while cursor < bytes.count {
            if bytes[cursor] == 92 {
                cursor += 2
                continue
            }
            if bytes[cursor] == 34 {
                cursor += 1
                return try JSONDecoder().decode(String.self, from: Data(bytes[start..<cursor]))
            }
            cursor += 1
        }
        throw DecisionError.invalidRequest("Unterminated JSON string.")
    }

    private mutating func value(path: [String], depth: Int) throws {
        guard depth <= 32 else { throw DecisionError.invalidRequest("JSON nesting exceeds 32 levels.") }
        whitespace()
        guard cursor < bytes.count else { throw DecisionError.invalidRequest("Incomplete JSON.") }
        if bytes[cursor] == 123 {
            cursor += 1
            var keys = Set<String>()
            whitespace()
            if cursor < bytes.count && bytes[cursor] == 125 {
                cursor += 1
                return
            }
            while true {
                let key = try string()
                guard keys.insert(key).inserted else {
                    throw DecisionError.invalidRequest("Duplicate JSON key: \(key)")
                }
                if path == ["schema"] { schemaOrder.append(key) }
                if path.count == 2, path[0] == "schema",
                    !["type", "description", "choices", "choice_descriptions"].contains(key)
                {
                    throw DecisionError.invalidRequest("Unsupported field definition key: \(key)")
                }
                try consume(58)
                try value(path: path + [key], depth: depth + 1)
                whitespace()
                guard cursor < bytes.count else { throw DecisionError.invalidRequest("Incomplete JSON object.") }
                if bytes[cursor] == 125 {
                    cursor += 1
                    break
                }
                try consume(44)
            }
        } else if bytes[cursor] == 91 {
            cursor += 1
            whitespace()
            if cursor < bytes.count && bytes[cursor] == 93 {
                cursor += 1
                return
            }
            while true {
                try value(path: path + ["[]"], depth: depth + 1)
                whitespace()
                guard cursor < bytes.count else { throw DecisionError.invalidRequest("Incomplete JSON array.") }
                if bytes[cursor] == 93 {
                    cursor += 1
                    break
                }
                try consume(44)
            }
        } else if bytes[cursor] == 34 {
            _ = try string()
        } else {
            let start = cursor
            while cursor < bytes.count && ![9, 10, 13, 32, 44, 93, 125].contains(bytes[cursor]) { cursor += 1 }
            guard cursor > start else { throw DecisionError.invalidRequest("Malformed JSON value.") }
            _ = try JSONSerialization.jsonObject(with: Data(bytes[start..<cursor]), options: .fragmentsAllowed)
        }
    }
}
