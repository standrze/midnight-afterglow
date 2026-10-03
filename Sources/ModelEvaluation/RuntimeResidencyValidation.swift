import Foundation

/// Setter history is required: a zero reported limit alone does not prove restoration.
struct RuntimeWiredState: Decodable {
    var activeBaselineBytes: UInt64?
    var activeTicketCount: Int
    var ticketCount: Int
    var backendFailureCount: Int
    var backendSuccessCount: Int
    var backendSupported: Bool
    var baselineBytes: UInt64
    var currentLimitBytes: UInt64
    var lastAttemptSucceeded: Bool?
    var lastAttemptedLimitBytes: UInt64?
    var lastConfirmedRestoredBaselineBytes: UInt64?
    var lastSuccessfulBackendLimitBytes: UInt64?

    var restored: Bool {
        backendSupported && backendFailureCount == 0 && activeTicketCount == 0 && ticketCount == 0
            && activeBaselineBytes == nil && baselineBytes == 0 && currentLimitBytes == 0
            && lastAttemptSucceeded == true && lastAttemptedLimitBytes == 0
            && lastSuccessfulBackendLimitBytes == 0 && lastConfirmedRestoredBaselineBytes == 0
    }

    func active(limit: UInt64, tickets: Int) -> Bool {
        limit > 0 && backendSupported && backendFailureCount == 0 && activeBaselineBytes == 0
            && baselineBytes == 0 && currentLimitBytes == limit && activeTicketCount == tickets
            && ticketCount == tickets && lastAttemptSucceeded == true && lastAttemptedLimitBytes == limit
            && lastSuccessfulBackendLimitBytes == limit
    }
}

struct RuntimeWiredRequest: Decodable {
    var activeState: RuntimeWiredState
    var requestedLimitBytes: UInt64
    var startReturnedLimitBytes: UInt64
}

struct RuntimeWiredSession: Decodable {
    var before: RuntimeWiredState
    var started: RuntimeWiredState
    var ended: RuntimeWiredState
    var requestedLimitBytes: UInt64
    var startReturnedLimitBytes: UInt64
    var endReturnedLimitBytes: UInt64
}

extension RuntimeNativeReport {
    /// Validate the fixed eight-warmup/eight-trial residency diagnostic independently of native eligibility flags.
    func validateResidency(model: String, workload: RuntimeEvaluationPlan.Workload, scope: String) throws {
        try validate(model: model, workload: workload, expectedMeasuredTrials: 8)
        guard ["request", "session"].contains(scope), wiredMemoryScope == scope,
            wiredMemorySessionFailures == [], let requests = wiredMemoryRequests,
            requests.count == warmups.count + trials.count, let after = wiredMemoryAfterMeasured, after.restored
        else { throw EvaluationFailure.transport("Missing residency evidence, incorrect scope or failed restoration.") }

        for (index, request) in requests.prefix(warmups.count).enumerated() {
            guard request.startReturnedLimitBytes == request.requestedLimitBytes,
                request.activeState.active(limit: request.requestedLimitBytes, tickets: 1),
                request.activeState.backendSuccessCount == index * 2 + 1
            else { throw EvaluationFailure.transport("Warmup request lacks confirmed request-scoped residency.") }
        }
        if scope == "session" {
            guard wiredMemorySessionEligible == true, let session = wiredMemorySession,
                session.before.restored, session.before.backendSuccessCount == warmups.count * 2,
                session.ended.restored,
                session.started.active(limit: session.requestedLimitBytes, tickets: 1),
                session.startReturnedLimitBytes == session.requestedLimitBytes, session.endReturnedLimitBytes == 0,
                session.started.backendSuccessCount == session.before.backendSuccessCount + 1,
                session.ended.backendSuccessCount == session.started.backendSuccessCount + 1,
                after.backendSuccessCount == session.ended.backendSuccessCount
            else { throw EvaluationFailure.transport("Session capacity or setter restoration history is invalid.") }
            for request in requests.suffix(trials.count) {
                guard request.requestedLimitBytes == session.requestedLimitBytes,
                    request.startReturnedLimitBytes == session.requestedLimitBytes,
                    request.activeState.active(limit: session.requestedLimitBytes, tickets: 2),
                    request.activeState.backendSuccessCount == session.started.backendSuccessCount
                else { throw EvaluationFailure.transport("Measured request changed the session capacity.") }
            }
        } else {
            guard wiredMemorySession == nil, wiredMemorySessionEligible == nil else {
                throw EvaluationFailure.transport("Request-scoped diagnostic unexpectedly contains a session.")
            }
            var previousSuccessCount = 0
            for request in requests {
                guard request.startReturnedLimitBytes == request.requestedLimitBytes,
                    request.activeState.active(limit: request.requestedLimitBytes, tickets: 1),
                    request.activeState.backendSuccessCount == previousSuccessCount + 1
                else { throw EvaluationFailure.transport("Request setter history or active capacity is invalid.") }
                previousSuccessCount = request.activeState.backendSuccessCount + 1
            }
            guard after.backendSuccessCount == previousSuccessCount else {
                throw EvaluationFailure.transport("Final request restoration setter history is incomplete.")
            }
        }
    }
}
