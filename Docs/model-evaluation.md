# Automated model evaluation

The evaluator, scorers and batch controller are native Swift, exposed through
ArgumentParser subcommands in one executable. `midnight-afterglow evaluate model` exercises a running
model's real `/v1/chat/completions` path, including its chat template and tool-call
parser. It does not start or download a model. Evaluation belongs in this package;
serving stays in Midnight.

```sh
midnight-afterglow evaluate model \
  --suite EvaluationSuites/coding-cyber-tools-v2.json \
  --model gemma-4-26B-A4B-it-midnight \
  --revision SOURCE_REVISION \
  --python-image 'python@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea' \
  --output EvaluationRuns/a4b-baseline-v1
```

The default endpoint is `http://127.0.0.1:8080/v1/chat/completions`; use `--endpoint`
for another listener. Authentication comes only from `AFTERGLOW_EVAL_TOKEN` in
the environment. The Docker image must already exist locally; the runner never
pulls it. Supply a suitable digest-pinned Python image if using another platform.

The initial, authored **development** suite has 14 cases: seven tool-use cases,
three coding cases and four cybersecurity cases. It covers typed arguments,
unnecessary calls, tool errors, untrusted tool-result instructions, dependent
calls, unrequested side effects, sequence/interval/event behavior, owner-based
authorization, archive paths, parameterized SQL and evidence-based triage.
Cyber correctness includes executed remediation contracts rather than only
vulnerability classification. This is a starting regression suite, not a broad
benchmark or independent holdout. Keep final holdout families separate from
calibration, development and recipe selection.

## Scoring and reports

Each case has explicit public prompts/tools and separate private expectations.
Expected arguments and final answers never enter requests. After a correct call,
its fixed mock result is returned using the model's actual call ID. Mock tools
never execute real actions. Wrong tools, extra calls/arguments, wrong argument
types, malformed JSON, missing answers and output truncation fail. Tool calls
in a multi-call turn are currently scored in declared order; use separate turns
when order matters. No language model judges the output.

Structured answers must match the specified JSON exactly, independent of object
key order. Python answers are source or one code fence. Private Python tests run
in a disposable Docker container: no network, non-root user, read-only input and
root filesystem, bounded CPU/memory/process count and a timeout. No generated
code executes on the host. These are behavioral correctness checks, not an
adversarial proof against generated code intentionally attacking its test oracle.

Output contains:

- `manifest.json`: suite SHA-256, model/revision, generation settings and sandbox image.
- `suite.json`: exact input suite, including private tests for later audit.
- `case-*.json`: each completed case, responses, timing and failure reason, saved immediately.
- `report.json`: all results and separate category pass/fail/error counts.

Existing output directories are rejected. Infrastructure errors remain in each
category's denominator and are reported separately; subsequent cases continue.
The command exits nonzero if any case fails or errors. Interrupted runs preserve
completed case files but do not constitute complete results. Review the outputs,
not just the aggregate score, before selecting a recipe.

HTTP wall time includes queuing, prefill and generation; it is not native decode
throughput. Source revision and optional `--artifact-sha256` are explicitly
operator-declared, not verified against server weights. These reports alone do
not qualify native speed, serving-memory budgets, device capacity or promotion.
Run the same immutable suite/settings on each checkpoint and retain the existing
native timing/memory identity checks for the quantization tradeoff decision.

## Extend the suite

Use schema version 1, unique case IDs and `sourceFamily` values. Each case includes
`prompt`, `tools` (possibly empty), and `turns`. A tool turn has nonempty `calls`,
each with `name`, typed JSON `arguments` and string `result`. The final turn has
empty `calls` plus either `answer` (JSON) or `pythonTests` (Python importing
`solution`). A case must end with a scored answer. Suites declare `split` as
`development` or `holdout`; the label itself does not prove independence.

## Native Swift batch runs

```sh
midnight-afterglow evaluate batch \
  --plan EvaluationRuns/gemma-quantization-suite-v2/plan.json \
  --output EvaluationRuns/gemma-quantization-suite-v2
```

The plan declares local model paths and immutable source revisions, a digest-pinned
sandbox image, SHA-256/size pins for the evaluator, suite, server, Metal libraries,
configuration and memory observer, and optional predecessor progress/identity pins.
The batch never downloads weights. It evaluates each recipe serially with its own
loopback Midnight server, using fixed generation settings. Full checkpoint file
hashes before and after each recipe detect changed weights or metadata.

`progress.json` records owned process IDs and completion; each recipe retains its
model report, checkpoint identities, server log and memory observation. Final
`RESULTS.md` separates coding, cybersecurity and tool-use scores, and
`paired-comparisons.json` lists individual gains, regressions and infrastructure
errors against each family's incumbent. Errors never become accuracy gains.
Existing progress and recipe outputs are protected. Cancellation stops only owned
processes/containers and preserves incomplete evidence. The plan requires at least
100 GiB free, with another 1 GiB of early-stop headroom.

The Swift regression tests cover typed tool arguments, prompt/expectation separation,
truncation, infrastructure failures, actual Docker pass/fail/timeout behavior,
batch validation/cancellation/server failures, and a native HTTP fixture exercising
all 14 cases. Python appears as generated-code test input executed in Docker; the
new evaluation framework and its tests do not depend on a Python controller.
Run `./test.sh --filter ModelEvaluationTests` with `AFTERGLOW_TEST_DOCKER_IMAGE`
set to the installed digest-pinned image to include Docker fixtures. Fixture scores
verify the harness; they are not model accuracy results.

## Versioned contract repair

Use `coding-cyber-tools-v2.json` for new full-suite runs. The v1 owner authorization
case is quarantined: its public missing-fields wording was ambiguous about an
optional admin flag, while private tests permitted owners without that flag. Its
reference also allowed two explicitly null IDs to match. Original v1 reports remain
preserved and their apparent Q5 gain on this case must not guide recipe selection.

V2 explicitly defines authentication identity, optional admin behavior, non-null
owner IDs and absent resources, with added boundary tests. The Swift regression
test passes the reference and rejects three defects. The single-case
`coding-cyber-tools-v2-owner-correction.json` allows rechecking only this repaired
case; the other thirteen cases are byte-for-byte unchanged. This is development
repair after reviewing outputs, not an independent holdout.

Strict final JSON scores include formatting: a correct value inside a Markdown
fence still violates the raw JSON contract. Review format-only failures separately
from tool-name/argument errors. An earlier tool-turn expectation must pass before
the evaluator reaches the final-answer check; a fenced correct final value is not
proof that the model chose the wrong tool. Original strict scores stay unchanged.

## Native runtime evaluation

`midnight-afterglow evaluate runtime --plan runtime-plan.json --output new-runtime-run`
adds native speed testing to the same Swift CLI. It launches a pinned, locally
available Midnight native benchmark executable, rather than treating HTTP wall
time as decode throughput. It does not download or change a model. The historical
Python campaign already running remains frozen; this command is for future runs.

Runtime plan version 1 uses these JSON fields (camelCase):

- `runtime` and `metal`: `{ "path": "absolute-path", "bytes": 123, "sha256": "64-hex-digits" }`.
  The Metal artifact must be named `mlx.metallib` beside the executable.
- `baseline` and `candidate`: each has `checkpoint` with `family`, `recipe`, `path`,
  `source_revision`; plus pinned `qualityIdentity`, `qualityReport` and `qualityManifest` artifacts.
  Identity is the complete `checkpoint-before.json` produced by `evaluate batch`;
  report and manifest are that recipe's `evaluation/report.json` and `manifest.json`.
  Both accuracy runs must use the same suite hash and case inventory, with no
  infrastructure errors. Failed accuracy cases stay visible and do not prevent
  measuring speed; speed qualification does not establish accuracy superiority.
- `workloads`: `{ "name": "coding-short", "prompt": "public prompt",
  "expectedPromptTokens": 128, "contextLength": 8192 }`. Exact prompt text, rendered
  token fingerprints and actual token counts are checked. Use multiple workloads
  to cover both coding and long-context cybersecurity.
- `minimumFreeBytes`: at least 107374182400; the controller adds 1 GiB early-stop headroom.
- `exclusiveProcessIDs`: positive PIDs of existing model jobs that must have exited
  before execution. This is an explicit exclusion check, not automatic discovery of
  every GPU process on the host. Run physically serially.
- `timeoutSeconds`: fixed per-arm deadline, 1 through 7200 seconds.

Each workload runs eight AB/BA pairs in three phases: same-model pre-controls,
candidate comparison, and same-model post-controls. Every process uses eight fixed
warmups, one measured trial, 256 output tokens, Metal, greedy target-only generation,
512-token prefill, no KV compression and no prompt reuse. No outlier removal or
adaptive warmup selection is performed. Checkpoints are fully hashed against
quality identities before running and after every phase. Raw native JSON, command
records, logs and progress are preserved in a protected output directory. Existing
runs cannot be overwritten; cancellation stops the owned native process.

Decode, prefill and TTFT use the 10% drift/order gates. Same-model median ratios
must lie in [0.9, 1.1], with bootstrap intervals containing one; control output
hashes must match. A failed phase stops the campaign and returns a failure exit.
Failed post-controls leave candidate measurements unqualified. The versioned Swift
bootstrap uses 2,000 paired resamples and a fixed PRNG; its exact intervals are not
claimed to be identical to the historical Python implementation. All measurements
remain in the report.

`performanceQualified` means the complete timing protocol passed, independently of
accuracy or memory selection. `normalSpeedBudgetPassed` additionally requires each
workload's decode-ratio lower 95% bound to be at least 0.75. A 40% exception is
never applied automatically: it still needs substantial quality evidence and the
owner's review of absolute speed. `defaultsChanged` always remains false. This
command measures warmed native rates; it does not measure cold loading or replace
the separate HTTP serving-memory screen. It has not yet run a full Gemma model
campaign; process fixtures verify orchestration, rejection and cleanup only.


## Native residency diagnostic

```sh
midnight-afterglow evaluate runtime-diagnostic --plan execution-plan.json --output new-diagnostic-run
```

This command investigates unstable timing; it cannot qualify a recipe or change a
default. The execution plan contains `version: 1`, a pinned `preregistration`
artifact, `waitFor` predecessor records (progress path, PID, pinned plan and
controller, explicit terminal statuses), and `exclusiveProcessIDs`. Existing
predecessor controllers and their reported children must exit with declared
terminal evidence before model execution. Missing terminal evidence fails the run.
These exclusions are explicit inputs; they do not discover every host process.

The pinned preregistration fixes workloads, checkpoint and quality identities,
runtime/Metal artifacts, provenance sources, and the full protocol. Version one
runs request/session/session/request for each workload, with eight warmups and
eight measured trials per process. It uses 256 output tokens, full prefill, greedy
Metal target-only generation, disabled prompt reuse and no KV compression. Fixed
protocol changes, cached work, missing residency observations, unsuccessful setter
history, changed session capacity or unconfirmed restoration invalidate evidence.

Every warmup and trial is retained alongside output hashes, native JSON, commands,
logs, full checkpoint hashes before/after each process and first/last-half drift
for decode, prefill and TTFT. No trials are removed or retried. Disk reserve and
cancellation checks remain active while waiting and during inference. Existing
output directories are protected. `performanceQualified` and `defaultsChanged`
always remain false, including successful diagnostics. The full Gemma residency
diagnostic has not run yet; fixture results only validate the command's behavior.

### Small runtime screening

Set `"mode": "screening"` in a new pinned runtime plan and run
`midnight-afterglow evaluate runtime --plan <plan.json> --output <new-directory>`.
Screening runs two AB/BA pairs (four workers) per workload, one warmup and one
measured 256-token trial per worker. It keeps checkpoint/quality identity checks,
full prompt prefill with reuse disabled, and the 100 GiB reserve plus 1 GiB headroom.
Reports preserve both paired ratios, absolute decode rates, prefill and TTFT.
They end with `complete_screening_only`; performance qualification, speed-budget
acceptance and default changes remain false. There is no confidence interval.
Use the existing coding/cyber/tool suite to eliminate recipes without useful
quality gains, then screen speed on a short workload. Reserve long-context and
full bracketing controls for finalists. Omit `mode` or use `qualification` for
the original eight-pair protocol. Never relabel interrupted legacy runs as screens.
