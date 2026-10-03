# midnight-afterglow MLX quantization standard v2

Recorded 2 October 2026. This is the active research and measurement contract for Gemma 4 26B-A4B and 31B. No newly tested recipe is promoted by this document. Historical Wick sources, campaigns and the v1 specification remain preserved. New fitting, calibration and quantization belong to midnight-afterglow; Midnight owns native serving, runtime inspection and HTTP evaluation. The package must remain standalone. The working folder is currently `afterglow` because live frozen campaigns use that absolute path.

## Owner objective and budgets

Select for generated coding and cybersecurity correctness. Math scores do not drive selection. Choose each model's precision separately. Normal exploration allows up to 25% lower decode tokens/s and 25% more measured serving memory, within device capacity. An exceptional recipe may be up to 40% slower only for a substantial coding/cyber gain and owner review of the remaining absolute tokens/s. The owner has not chosen an absolute speed floor or numerical definition of exceptional accuracy; do not invent one or silently apply the exception. Ask for budgets before extending these model-specific limits to another model.

Keep at least 100 GB free. Controllers enforce the stricter 100 GiB reserve and stop owned work at 101 GiB. Account for the complete output, private staging and captured activations before starting. Preserve compact evidence and retire rejected or interrupted owned payloads only after their consuming processes exit. Protect serving models, original sources, unrelated data and existing runs. No publishing is authorized.

## Recipe identity and native compatibility

Record storage, fitting and execution independently:

| Field | Required evidence |
| --- | --- |
| Source | Original publisher, immutable revision, complete source weight hashes, tokenizer/config identity, target/assistant pairing where applicable |
| Storage | Native tensor names/index; per-module format, bits, group size, packed geometry, scale/bias dtype; protected tensors and ties; actual weight and sidecar bytes |
| Fitting | Named algorithm/version and code hashes; fit-corpus families, capture identity, source-bound inputs, coverage, damping, scale factors, output-row coverage and stored-grid validation |
| Execution | Exact native executable/Metal hashes, runtime settings, model path/full identity, activation dtype, KV policy, prefill size, cache policy and assistant status |
| Selection | Per-task generated results, gains/regressions, qualified timing, observed serving memory, workload/device scope, uncertainty and owner tradeoff decision |

ScaleSearch is offline fitting and can accompany either Q4 or Q5 when that native format/geometry is supported. Activation weighting can also be combined with scale search. The current covariance method adds correlated-input error compensation to native-affine scale selection; it does not introduce an inference-time correction operator. An affine adaptation must be named distinctly from published FP4 scale-search algorithms. Equal bits or weight bytes do not establish equal generated accuracy, memory or speed: outputs and expert routing can change.

Keep ordinary quantization as a control. Compare incumbent affine LS2 Q4, ordinary Q4/Q5, Q5 plus ScaleSearch, diagonal activation-fitted Q4, covariance plus ScaleSearch Q4, and selective Q8 promotions where compatible. A rejected implementation stays rejected for its recorded recipe and workload; its outcome is not a universal claim about that bit width or method.

Fit the actual stored scale/bias dtype, currently BF16 for these native Gemma paths. Preserve native per-module quantization metadata, untouched template payloads, source architecture protections, expert order, fused gate/up semantics, modalities and tied weights. Do not mix formats within a stacked expert tensor without a separately validated loader/kernel path. Unsupported dimensions, under-covered experts and unvalidated assistant pairing must be reported explicitly. Never silently substitute a community checkpoint or alter assistant precision. Existing maker-source research copies are pinned data, not a new download or publication policy.

Conversion must use a new transactional destination. Verify every patched payload and every untouched template byte; record deliberate sidecar edits separately. A synthetic dense/expert fixture proves the tested layout and operation only. Require complete-checkpoint native reload and generated evaluation before asserting full-model compatibility or quality.

## Calibration and quality selection

Fit uses fit inputs only. Development reconstruction can screen a fixed recipe but must not be relabeled a final holdout. Capture actual routed-expert inputs, report coverage, bound retained arrays and full fitting memory, and release statistics between targets. Record rank-deficient covariance and damping. A sampled projection error reduction is a proxy, not a prediction of generated accuracy. Full output-row fitting still covers only the selected modules/experts.

Use identical native generation settings and exact task/token identities for baseline and candidate. Current coding/cyber quality runs use greedy target-only generation, 2048 output tokens, 8192 context, prefill step 512, no assistant, no compressed KV and no prompt reuse. Supply public tasks only to generation. Private answer keys and reference implementations remain scoring inputs. Execute generated code only inside the pinned Docker sandbox with no network access.

Report each suite separately: executable remediation, coding/cyber pilot, vulnerability classification, repository edits and long-context tasks measure different behavior. Include task-level gains and regressions, valid-output counts, false positives/negatives when relevant, truncation and uncertainty. Do not sum unrelated suites into a population accuracy claim. A matched-settings qualification permits comparison; it does not establish a meaningful accuracy improvement.

Freeze new evaluation prompts/tests before generating candidate outputs. Validate reference and plausible defective implementations before scoring. Record whether tasks were authored with knowledge of earlier outcomes. Convenience-authored function tasks are limited evidence, even if unused for calibration; they do not become a representative independent benchmark. After evaluation exposure, treat them as development data for any subsequent recipe tuning. Repository edits and long-context behavior remain required before broad promotion.

## Timing qualification

Previous same-model controls failed stability. Preserve those results. The preregistered replacement uses the unchanged short coding and long cyber prompts, eight fixed warmups per process, eight measured AB/BA pairs per workload, fixed 256-token work, no prompt reuse, no KV compression and no assistant. Warmups are unscored; every scheduled measurement remains in the report. More warmups are an experiment, not evidence that stability is solved.

Run physically serially after other model compute exits. Pin the executable, Metal library, helper, probe, manifests and full checkpoints before and after. Decode, prefill and TTFT must pass the existing 10% drift/order gates. For same-model controls, each accepted metric's paired ratio must lie in [0.9, 1.1] and its recorded interval must include one. A failed control stops candidate timing under that protocol. No outlier removal, adaptive warmup choice, selective retry or retrospective gate relaxation.

Candidate timing is queued after repository/context completion in `selective-covariance-paired-timing-v6`. Fresh controls bracket each eligible model’s comparison, and every timing checkpoint file must match the complete checkpoint identity from its successful generated-quality evidence. A model that failed initial controls is held under that unchanged protocol; a different model not reached by the earlier queue may run its own first controls. Candidate timing requires passing controls before and after the comparison. Failed post-controls invalidate intervening speed claims. Report absolute baseline/candidate tokens/s, paired ratios/intervals, prefill and TTFT separately. Native warmed rates exclude checkpoint loading; report cold/loading behavior separately. A normal decode ratio must be at least 0.75; an exceptional ratio must be at least 0.60 and still requires the accuracy/absolute-speed review. No throughput claim follows from two HTTP clients.

## Serving memory and device capacity

Weight-file bytes are a storage measure. Measure real HTTP server physical footprint under matched requests. Use identical short coding and long cyber prompts with warmup, solo and two-client phases. Bracket each candidate with initial/repeated baselines. Verify full checkpoint/runtime identities before and after. Authentication uses ephemeral environment/header secrets, never logged arguments.

Require complete HTTP status/stream/finish/model identity and matching actual prompt, completion and cached-token counts. Preserve admission failures, early stops and probe errors; incomplete or smaller work never establishes savings. Observe peak footprint while the listener remains alive after requests, including loading and retained allocations. Baseline drift must be at most 10%; compare the candidate against the lower matched baseline peak. Every required phase must be within 1.25 times that baseline. Verify owned listener/probe exit and the disk reserve.

These are empirical workload/device capacity observations, not a guarantee at every context or concurrency. Record host memory/swap state and failures. Two connections may serialize or reject; do not describe them as simultaneous model forwards without evidence. Assistants, different KV policies and broader serving concurrency need separate pairing/capacity measurements.

## Current evidence and remaining work

Both selective covariance checkpoints completed conversion and byte verification. A4B fits nine selected attention/expert targets; 31B fits three attention projections. All selected output rows are covered, and other native weights retain the incumbent recipe. Their weight files have identical shard sizes to the incumbents: A4B 14,194,825,714 bytes; 31B 17,269,383,727 bytes. Conversion provenance adds about 1.1 MB and 1.7 MB respectively; Both candidates passed the completed serving-memory screen. Native speed remains unqualified.

All eight matched development comparisons are complete (16 native runs, 320 generated responses). A4B passes pilot 17/18 versus 16/18, remediation 2/6 versus 2/6, OWASP 41/48 versus 39/48 and earlier contracts 6/8 versus 6/8. It gains one owner-authorization coding task and two vulnerable SQL cases, with no task-level regressions on these slices; seven OWASP false positives remain. 31B passes 18/18, 3/6, 40/48 and 8/8 in both recipes, with identical task-level pass/fail outcomes throughout. These exposed development results do not establish promotion or an exceptional slowdown justification. The development controller exited. Initial A4B same-model timing controls completed all 32 arms but failed the preregistered drift gates; their TPS observations cannot establish a qualified slowdown. The unchanged A4B protocol is held rather than blindly rerun. 31B/bracketed timing is live with its original qualification gates after the owner-contract repair; A4B remains held.

Fresh six-contract comparisons completed with matched settings and checkpoint/runtime identities: A4B incumbent 1/6 versus covariance 1/6; 31B incumbent 2/6 versus covariance 3/6. The sole 31B gain passes private behavior tests while the incumbent fails the single-code-block response contract. That pair does not prove the incumbent's code would fail behavioral tests. These six authored tasks do not establish broad accuracy or independence. The pre-generation correction to long leading-zero header reference cases remains preserved in the fresh-corpus history.

A4B's serving-memory comparison completed 24 requests across initial incumbent, covariance and repeated incumbent. Every required phase passed stable-baseline and ≤1.25 memory checks; the largest candidate/lower-baseline footprint ratio was 1.005482 (about +0.55%). Full before/after checkpoint identities match, and owned listeners/probes exited. 31B completed all 24 requests across candidate and initial/repeated incumbents; every phase passed, with largest candidate/lower-baseline footprint ratio 1.002425 (about +0.24%). Both screens total 48 completed requests and verified unchanged checkpoints/runtimes, with owned servers/probes exited. Repository and long-context comparisons completed, using 62 candidate generations against 62 integrity-verified earlier baselines. Both recipes have identical task-level outcomes: A4B repository 2/3 and context 31/32; 31B repository 1/3 and context 24/24. These observations are scoped to the matched workload/device, not a universal capacity guarantee.

The built-in evaluator, batch controller and runtime controller are native Swift ArgumentParser subcommands: `midnight-afterglow evaluate model`, `evaluate batch` and `evaluate runtime`. Eighteen evaluation tests passed in the completed package workflow with zero failures and zero skips, including actual HTTP/Docker fixtures, tool-argument checks, batch cleanup/failure handling, the repaired owner-contract check, and six runtime identity/control/statistical/cancellation fixtures. The actual release executable built and its runtime command registration was verified. The new runtime command has not yet run a full Gemma campaign; the active v6 campaign retains its frozen historical scheduler and native executable.

All eight recipes completed the original 14-case authored development suite (112 cases), with zero infrastructure errors and unchanged full checkpoints. The apparent Q5 gain is quarantined: the original owner prompt ambiguously said missing fields deny while a private test allowed an owner without optional is_admin; the original reference also allowed explicitly None IDs to match. Do not treat that case as accuracy evidence. V2 makes those rules explicit and adds boundary tests. A separate native Swift one-case batch tested the repair on all eight unchanged checkpoints: every recipe passed.

Composite development evidence reuses thirteen exactly unchanged case inputs/outputs plus the corrected case from a separate matched run. It is not a single full v2 invocation or an independent holdout. All recipes have coding 3/3 and cybersecurity 3/4; strict tool scores are A4B 4/7 and 31B 2/7, identical within each family. All failed structured answers are matching JSON inside Markdown fences; preceding tool/argument expectations passed. These remain response-format failures, not wrong tool choices. No within-family gain/regression is demonstrated by this small suite. See [automated evaluation](model-evaluation.md) and the source-bound `native-suite-repaired-composite-audit-v1.json` for limitations.

The owned timing v5 run was intentionally stopped to repair the case and preserve partial diagnostics. V6 resumes after all model repairs and package tests completed, with unchanged statistical gates/runtime identities. This interruption is not a statistical-control failure and its partial arms do not qualify speed. Frozen active manifests and paths remain immutable.


Evidence roots:

- Preserved calibration/reconstruction: `../wick/benchmark-results/gemma-sources-20261001/covariance-inputs-v1` and `covariance-reconstruction-v2`.
- Current 31B conversion: `Benchmarks/GemmaQuantization20261002/31b-selective-conversion-v1`.
- Native quality and storage: `../midnight/benchmark-results/gemma-quantization-20261001/selective-covariance-quality-v2`.
- Initial timing preregistration: `../midnight/benchmark-results/gemma-quantization-20261001/steady-state-timing-protocol-v1`.
- Candidate timing and bracketing controls: `../midnight/benchmark-results/gemma-quantization-20261001/selective-covariance-paired-timing-v6`.
- New authored evaluation: `../midnight/benchmark-results/gemma-quantization-20261001/fresh-code-cyber-v3` and `selective-covariance-fresh-quality-v2`.
- Native Swift evaluation: `EvaluationRuns/gemma-quantization-suite-v2`, `EvaluationRuns/gemma-owner-contract-correction-v1`, `EvaluationRuns/harness-validation-native-swift-v2` and `EvaluationSuites/coding-cyber-tools-v2.json`.
- Serving memory: `../midnight/benchmark-results/gemma-quantization-20261001/selective-covariance-serving-memory-v2`.
- Repository/context checks: `../midnight/benchmark-results/gemma-quantization-20261001/selective-covariance-repository-context-v2`.

No new default is qualified. Before selection, finish the assembled-model comparisons, fresh/repository/long-context evidence, stable candidate timing with controls, serving-memory screen and relevant assistant pairing. Present a per-model accuracy/speed/memory table with practical limits and per-task regressions. Call the result the best tested tradeoff within the declared candidate set; never claim global optimality. Failed/rejected payloads should be retired after preserving their evidence and confirming no active consumers.

The older A4B pilot authorization gain was separately reviewed using unchanged retained outputs: with all actor fields present and authenticated equal to integer 1, the incumbent permits the matching owner while covariance denies as the public exact-True rule requires. Both outputs were executed only in the pinned sandbox. This post-exposure diagnostic is not a new model generation or holdout; see `pilot-authentication-gain-review-v1/findings.json`.

A materially different A4B residency diagnostic is preregistered in `Benchmarks/GemmaQuantization20261002/a4b-runtime-residency-diagnostic-preregistration-v1.json`. It fixes request/session/session/request process order, eight warmups and eight measured trials per process, with no cache and no score pruning. It is not started. The native Swift `midnight-afterglow evaluate runtime-diagnostic` entrypoint passed twenty package tests with no failures or skips and its release CLI was verified. The frozen executable is queued in a tracked process session, waiting for the 31B campaign and owned model processes to exit; the package compiler has exited. No diagnostic model generation has started. Diagnostic rates cannot qualify serving speed or change existing selection gates. The separate two-resident paired prototype remains limited to Gemma 3 270M; its architecture and capacity guards are unchanged.

### Owner-directed smaller screening (2026-10-02)

The owner requested fewer rounds to narrow recipes before expensive validation.
The legacy 31B control campaign was stopped with SIGINT at 26 completed arms;
its partial results are retained and do not qualify performance. The prior queued
A4B residency controller exited before any model work; its stale waiting record
is retained and it is not restarted.

The Swift `midnight-afterglow evaluate runtime` plan now accepts
`"mode": "screening"`: two AB/BA pairs, four workers per workload, one warmup
and one measured 256-token trial per worker, candidate phase only. Screening
reports both ratios and absolute decode rates, prefill and TTFT, with no confidence
interval, no speed-budget acceptance and no performance qualification. Existing
quality and checkpoint pins, no-cache/full-prefill work, capacity and disk reserve
checks remain enforced. Qualification plans retain the original protocol.

Use retained coding/cyber/tool results for initial elimination; start short runtime
screening with the A4B selective-covariance recipe, which has a limited exposed
authentication improvement. The repaired 14-case composite showed no recipe
gain; this is not evidence of a broad improvement. Its runtime plan binds to the
matching one-case repaired quality runs, not an independent holdout. Reserve
long-context tests and full repeat controls for promising finalists.

### Completed small recipe screen (2026-10-02)

All eight recipes retained at scheduling time completed the same four new
development cases once (32 cases, zero infrastructure errors). A4B Q5 gains
redirect validation but regresses on required input-type handling; 31B selective
Q8 regresses on redirect whitespace rejection. Category totals do not establish
equal task outcomes. The shared nested Gemma-tool symptom reproduces a current
local structured-decoder limitation and is not isolated quantization accuracy.

Six challengers completed two paired short runtime rounds apiece through native
Swift (24 measured arms): A4B Q5 about 15–16% slower, activation-fitted Q4 roughly
unchanged, selective Q8 6–9% slower, covariance Q4 18–23% slower with substantial
rate variation; 31B selective Q8 16–20% slower and covariance roughly unchanged.
These are short uncached target-only screening observations, not qualified
confidence intervals or long-context evidence. Full source-bound review is
`../midnight/benchmark-results/gemma-quantization-20261001/tradeoff-snapshot-v6.json`
relative to the Afterglow project root. Earlier serving-memory comparisons remain
separate. The 31B selective-Q8 owned payload was rejected and deleted after full
hash verification and consumer shutdown, retaining compact metadata/reports.

Keep installed ScaleSearch Q4 defaults on both families. A4B activation-fitted
ScaleSearch Q4 remains the strongest exploratory candidate from earlier exposed
development coverage, with roughly unchanged short speed and little tested
memory increase, but the four new cases establish no additional gain. No 31B
challenger has a demonstrated useful quality win without a relevant regression.
The 40% exception is unused. Further adoption requires stronger quality evidence
and finalist performance validation; no failed or interrupted legacy run is
reclassified as qualification.

### Experimental smaller-group Q4 follow-up

Affine Q4 ScaleSearch now supports G32 as well as G64/G128 in the native
Afterglow converter. Q5 ScaleSearch remains G64/G128. Eight native package
fixture tests passed with no failures or skips, including CPU/Metal per-group
stored-grid fallback, batched expert packing, CLI conversion and checkpoint
reload. These checks establish implementation compatibility, not model quality.

The single A4B experiment is preregistered in
`Benchmarks/GemmaQuantization20261002/a4b-q4-g32-small-screen-preregistration-v1.json`.
Its bounded-memory source conversion uses the cached immutable maker revision
and a native Swift controller enforcing the 100 GiB disk reserve plus 1 GiB
early-stop headroom. Export packing and module geometry are checked before
transaction commit. The four-case development suite runs once per model on the
same parser-fixed serving binary; quality survivors alone receive two paired
speed rounds and serving-memory measurement. This does not claim an independent
holdout, guaranteed accuracy gain, qualified speed, or adoption. Installed
G64 defaults and the model-specific 25%/40% review budgets remain unchanged.

The A4B/G32 follow-up completed the four-case screen: both incumbent and
candidate passed 3/4 with the same case outcomes and no infrastructure errors.
G32 showed no new gain and was rejected without timing or serving-memory rounds.
Its owned 15.77 GB of weight shards were hash-verified and deleted after consumers
exited. Review and compact retirement proof are in
`EvaluationRuns/gemma-a4b-g32-small-quality-screen-v1/`. G32 implementation support
remains tested; no model default or claim of broad accuracy equivalence changed.

### Strengthened four-case screen v2

Two native sandbox oracle fixtures passed before a single pass per retained
model (20 case executions, 25 assistant responses, zero infrastructure errors).
The same pinned parser-fixed server served all models, and all full checkpoint
identities matched before/after. A4B incumbent and activation-fitted Q4 tied at
3/4; Q5 scored 2/4, gaining redirect validation but regressing on recursive
configuration deletion and required scanner error handling. 31B incumbent and
covariance-fitted Q4 both passed 4/4. This is development evidence, not a broad
accuracy score or independent holdout; v1 runs are not relabelled.

No new performance rounds were warranted. Installed ScaleSearch Q4 remains
selected for both models; A4B activation fitting is exploratory only. The owned
A4B Q5 and 31B covariance payloads were freshly rehashed and deleted after
consumer shutdown, recovering 34.62 GB of weight data. Compact reports, metadata
and proof remain in `EvaluationRuns/gemma-small-quality-screen-v2/`. Historical
plans referencing deleted checkpoints must not be rerun. No stronger-accuracy
claim or default change follows from these results.

### Precision reference and isolated embedding override

A native standard-Q8 A4B diagnostic reference passed the single unchanged v2
redirect case that the Q4 incumbent failed. This is one development response,
not BF16 ground truth, a broad accuracy estimate, or an eligible full-model
memory-budget candidate. Source revision and full before/after identity proof
are retained in `EvaluationRuns/gemma-a4b-q8-redirect-reference-v1/`.

The smaller embedding-only Q8 / otherwise ScaleSearch Q4 export differed from
the incumbent in exactly two tensors: embedding packed weights and scales. All
other 1,337 tensors, including embedding bias, were byte-identical. Its four-case
screen passed 2/4 versus incumbent 3/4, still failing redirect validation and
regressing on recursive configuration deletion. It was rejected before timing
or serving-memory tests and its freshly hashed 14.56 GB of weight shards were
deleted after consumers exited. Proof is retained in
`EvaluationRuns/gemma-a4b-q8-embedding-small-screen-v1/`.

Attention-only Q8 completed its native export and one four-case screen. The
115 Q8 attention matrices matched the higher-precision reference exactly; all
994 other arrays matched the incumbent. It tied the incumbent at 3/4 with the
same case outcomes and zero infrastructure errors. No timing or serving-memory
rounds were run. Its freshly verified 14.75 GB of weight shards were deleted
after consumers exited; compact proof remains in
`EvaluationRuns/gemma-a4b-q8-attention-small-screen-v1/`.

Screen candidates with one pass of the four development cases and no repeated
attempts on failed cases. Reject ties and regressions before performance work.
A regression-free quality gain receives two paired speed rounds; broader
accuracy and budget validation is reserved for survivors before adoption.
These small screens narrow candidates, rather than establish broad accuracy.
The installed profiles and 25% normal speed/memory budgets remain unchanged;
the owner-reviewed 40% speed exception is unused.

### Combined embedding and attention precision screen

The combined A4B Q8/G64 embedding and attention override, with the remaining
210 matrices at Q4/G64 ScaleSearch, passed 3/4 with exactly the incumbent's
case outcomes. One native Swift pass produced five responses and zero
infrastructure errors. Its redirect response still omitted literal-space
rejection despite describing it in a comment. All 348 selected projection and
embedding arrays matched the Q8 reference; the other 991 arrays matched Q4.
The additional 924 MB of tensor storage did not establish an accuracy gain
and was not a serving-memory measurement.

No timing or serving-memory rounds were run. The candidate's freshly verified
15.12 GB of weight shards were deleted after all consumers exited. Compact
proof remains in
`EvaluationRuns/gemma-a4b-q8-embedding-attention-small-screen-v1/`.
Neither isolated nor combined embedding/attention overrides reproduced the
full-Q8 diagnostic's redirect pass. Installed defaults remain unchanged.

### A4B covariance strengthened-screen retirement

A single native Swift v2 pass scored A4B covariance-fitted Q4 at 2/4 versus
incumbent 3/4, with zero infrastructure errors and unchanged full checkpoint
identities. New dictionary children were copied without applying nested None
deletion markers, a coding regression. Both recipes failed redirect validation
and passed scanner arguments and typed tool roundtrip. No new performance rounds
were run. Earlier authentication gains and variable short speed evidence remain
preserved; they do not offset the newly observed regression. Prior serving-memory
growth was approximately 0.55%, and short decode screens were 18–23% slower,
without performance qualification.

The rejected owned checkpoint's freshly verified 14.19 GB of weight shards were
deleted after consumer shutdown. Compact metadata, conversion provenance and
full identities remain in `EvaluationRuns/gemma-a4b-covariance-small-screen-v2/`.
Installed defaults and publisher sources remain unchanged. Historical plans
pointing to the retired checkpoint must not be rerun.
