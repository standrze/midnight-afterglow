import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// Exercises the served model's real chat template and tool parser through chat completions.
public struct EvaluationHTTPTransport: EvaluationTransport {
    public var endpoint: URL
    public var model: String
    public var maximumTokens: Int
    public var timeout: Double
    private let token: String?
    private let session: URLSession

    public init(
        endpoint: URL, model: String, maximumTokens: Int = 2048, timeout: Double = 120, token: String? = nil,
        session: URLSession = .shared
    ) {
        self.endpoint = endpoint
        self.model = model
        self.maximumTokens = maximumTokens
        self.timeout = timeout
        self.token = token
        self.session = session
    }

    public func complete(messages: [EvaluationMessage], tools: [EvaluationTool]) async throws -> EvaluationResponse {
        struct Tool: Encodable {
            let type = "function"
            let function: EvaluationTool
        }
        struct Request: Encodable {
            let model: String
            let messages: [EvaluationMessage]
            let tools: [Tool]?
            let temperature = 0
            let stream = false
            let maxTokens: Int
            enum CodingKeys: String, CodingKey {
                case model, messages, tools, temperature, stream
                case maxTokens = "max_tokens"
            }
        }
        struct Response: Decodable {
            let model: String
            let choices: [Choice]
            let usage: Usage?
            struct Choice: Decodable {
                let message: EvaluationMessage
                let finishReason: String
                enum CodingKeys: String, CodingKey {
                    case message
                    case finishReason = "finish_reason"
                }
            }
            struct Usage: Decodable {
                let completionTokens: Int
                enum CodingKeys: String, CodingKey { case completionTokens = "completion_tokens" }
            }
        }
        var request = URLRequest(url: endpoint, timeoutInterval: timeout)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        request.httpBody = try JSONEncoder().encode(
            Request(
                model: model, messages: messages, tools: tools.isEmpty ? nil : tools.map { Tool(function: $0) },
                maxTokens: maximumTokens))
        let start = ContinuousClock.now
        let (data, raw) = try await session.data(for: request)
        let elapsed = start.duration(to: .now)
        guard let http = raw as? HTTPURLResponse, http.statusCode == 200 else {
            let status = (raw as? HTTPURLResponse)?.statusCode ?? 0
            throw EvaluationFailure.transport("Chat completion failed with HTTP \(status).")
        }
        guard data.count <= 8 * 1024 * 1024 else { throw EvaluationFailure.transport("Response exceeds 8 MiB.") }
        let decoded = try JSONDecoder().decode(Response.self, from: data)
        guard decoded.choices.count == 1, decoded.model == model else {
            throw EvaluationFailure.transport("Expected one choice and the requested served model identity.")
        }
        return .init(
            message: decoded.choices[0].message, finishReason: decoded.choices[0].finishReason,
            completionTokens: decoded.usage?.completionTokens,
            elapsedSeconds: Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18)
    }
}
