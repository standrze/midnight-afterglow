# Compact INT4 proof of concept

This is an experimental Afterglow format and converter, not a new default or a
native MLX checkpoint. Disaggregated quantization is out of scope. No full model
has been exported, published, or qualified with this format.

## Stored arithmetic

For row `n`, column `k`:

```
W[n,k] = BF16_gain[n] * (
    2^(E8M0[n,k/32] - 127) * signed_int4[n,k]
    + BF16_offset[n,k/64])
```

Codes are two's complement -8 through 7. The even-indexed code occupies the low
nibble of each byte. K is contiguous, positive, and divisible by 64. E8M0 bytes
0 through 254 encode powers of two (0 means 2^-127, not zero); 255 is rejected.
Offsets are finite. Row gains are finite and positive. BF16 uses nearest-even
rounding. Reconstructed values must be finite. Version 1 always stores row gains.
The numeric formula is independent of an MLX BF16 dequantization rounding step.
CPU accumulation and GPU output use FP32; the GPU proof accepts FP16 inputs.
Reassociation and device subnormal behavior preclude a blanket bitwise guarantee.

Payload size is `N*K/2 + N*K/32 + 2*N*K/64 + 2*N` bytes: 4.5 + 16/K
bits per weight. This is 4.5625 at K=256, before the 24-byte artifact overhead.

The bounded fitter searches six row-gain seeds across an octave and alternates
code/exponent selection, least-squares shared-offset fitting, and row-gain fitting.
It scores stored metadata and retains the best iteration. It currently optimizes
weight reconstruction error, not activation covariance or model quality. Joint
optimization here means alternating discrete and continuous updates; it is not a
global optimum claim. Compare against ordinary affine Q4/G64 and symmetric
Q4/G32, both 4.5 payload bits per weight with BF16 metadata.

## File contract: AGQ4E001

All multibyte integers are little-endian. Layout:

1. Eight ASCII bytes `AGQ4E001`.
2. UInt32 N, UInt32 K.
3. Row-major packed code bytes.
4. Row-major E8M0 bytes.
5. Row-major BF16 offset bit patterns (UInt16).
6. BF16 gain bit patterns (UInt16), one per row.
7. UInt64 FNV-1a checksum of all preceding bytes (accidental corruption only).

The reader validates exact size before model-sized allocations, overflow,
geometry, checksum, reserved metadata and reconstructed finiteness. The writer
stages a complete artifact in the destination directory and publishes via an
exclusive hard link. Existing outputs are never replaced. This ensures complete
file visibility; it does not promise persistence through sudden power loss.

## Execution

`CompactInt4Matrix.multiply` computes group dot products directly from packed
nibbles. Offset corrections use group input sums. It never expands the whole
weight matrix. The independent dense oracle reconstructs the weights first.

`CompactInt4Metal` exposes two explicit backends:

- `packed`: direct packed Metal shader, FP32 dot accumulation.
- `nativeInt4`: Metal TensorOps FP16 × signed INT4 for each G32, then per-row
  scale/offset corrections and gain. This requires macOS 26.4 and Apple GPU
  family 10. It does not require macOS 27 scale planes, silently fall back, or
  expand the weight matrix. The small-group epilogue is a correctness prototype;
  it is not yet a tuned prefill implementation.

The source ships as a resource and is compiled through Metal at runtime. A
successful `compile-metal` run proves compilation only, not kernel execution.
`metal` explicitly dispatches both backends and compares them with the CPU.

## Reproduce

From Afterglow:

```sh
swift build --product afterglow-int4-proof
# Every invocation needs a fresh output filename.
$(swift build --show-bin-path)/afterglow-int4-proof /tmp/example.agq4
$(swift build --show-bin-path)/afterglow-int4-proof /tmp/example-compiled.agq4 compile-metal
$(swift build --show-bin-path)/afterglow-int4-proof /tmp/example-metal.agq4 metal
./test.sh --filter CompactInt4Tests
```

An optional fourth argument supplies a bounded JSON matrix containing `rows`,
`columns`, flattened row-major `weights`, and a descriptive `source` string. The
caller is responsible for preserving provenance. No model is downloaded.

```sh
$(swift build --show-bin-path)/afterglow-int4-proof /tmp/slice.agq4 cpu MATRIX.json
python3 Scripts/validate-compact-int4.py MATRIX.json /tmp/slice.agq4
```

The Python validator selects MLX CPU and independently parses the payload to
compare weight error with the two ordinary Q4 controls. It is a bounded research
check, not a checkpoint loader or substitute for the strict Swift reader.

## ScaleSearch v2

`Scripts/gemma_covariance_scale_search.py` now jointly refits scale and bias
under the conditional inverse-Cholesky objective, then searches exact adjacent
BF16 metadata values. It resimulates the codes and error compensation for each
candidate, with at most two refinement rounds by default (bounded 0...8).

It also runs the unchanged v1 algorithm and compares complete output rows under
`||(W - Q) U^-1||²`, where `Uᵀ U = H^-1`. This retains cross-group effects and
keeps v1 rows on ties or regressions. The guarantee is for that calibration
objective on the stored affine grid in FP32, not held-out quality or rounded
BF16 GEMM. Cost is additional conversion time; packed weights, metadata types,
group geometry and serving kernels stay the same.

`refinement_steps=0` reproduces v1. The preregistered checkpoint converter
requires `"refinement_steps": 2` in a new plan to opt into v2; plans omitting
that field retain v1. The standalone primitive defaults to v2, while the probe
keeps its GPTQ control unrefined. Reports preserve legacy group diagnostics,
separately label refined-pass diagnostics, and report final per-row acceptance
and objective totals. Frozen v1 campaign code and existing checkpoints remain
unchanged. Native activation-weighted LS2 already had joint fitting; this change
fills the gap specifically in the covariance search branch.

## Measured evidence (2026-10-02)

The CPU proof passed four Swift tests covering signed packing, adjacent G32
scales, shared offsets, a dense oracle, malformed inputs, corruption and exclusive
artifact publication. All 27 Gemma Python research tests passed, including the
selective checkpoint converter and source/template preservation checks.

For a bounded BF16 slice (first 17 rows, first 256 columns) of
`model.language_model.layers.0.self_attn.q_proj.weight` from
`google/gemma-4-31B-it@842da3794eaa0b77d5f08bae87a17459d91ff475`:

| Format | Payload bits/weight | Relative squared weight error |
| --- | ---: | ---: |
| Compact INT4 proof | 4.5625 | 0.01024347 |
| MLX affine Q4/G64 | 4.5 | 0.00996362 |
| Symmetric Q4/G32, max-abs scale | 4.5 | 0.01110586 |

The compact format is about 2.8% worse than native affine on this slice; it is
not an accuracy win. Packed CPU execution agreed with the independently expanded
dense oracle within 6.54e-8 maximum absolute error for the tested inputs.

On that same slice with **synthetic** correlated calibration inputs (seed 629,
512 samples, 256 channels), covariance ScaleSearch v2 reduced its objective from
0.01059661 to 0.01008562 (about 4.82%), accepting 15 of 17 rows. Both results
occupied 2448 tensor bytes. One concurrent-workload CPU run took 0.52 s for v1 and
1.30 s for v2; these are conversion observations, not isolated performance claims.
There is no held-out model-quality result from this fixture.

Both Metal pipelines compiled on macOS 26.6.2. GPU execution validation is pending
coordination with a separate active timing campaign. Compilation is not execution
proof. The format remains an experimental candidate pending native execution,
representative calibration and runtime comparisons.


### Actual activation screen

The full package `./test.sh --filter CompactInt4Tests` passed on 2026-10-02.
ScaleSearch was then tested on 16 deterministic full-width rows from each of
Gemma-4-31B layers 0, 29 and 59, with 4096 captured fit inputs and 4096 captured
development inputs. Fit families were cpython-stdlib/swift-nio; development
families were httpx/swift-huggingface. These are previously used development
corpora, not a fresh generated-quality holdout.

| Layer | Q4/G64 v1 dev relative output MSE | v2 | Reduction |
| --- | ---: | ---: | ---: |
| 0 | 0.0000852631 | 0.0000762210 | 10.60% |
| 29 | 0.00464842 | 0.00418314 | 10.01% |
| 59 | 0.000872139 | 0.000784741 | 10.02% |

Each G64 result used exactly 48,384 tensor bytes for the selected rows.
G128 improved development error by 7.23–10.01%, also at unchanged size.
Measured process RSS was about 1.34 GB. Reports, input hashes, selected source
payload hashes, an installed-checkpoint control, and frozen code are in
`Benchmarks/ScaleSearchV2ActualInputs20261002`. This supports the fitting change;
it does not qualify a complete model or promote a serving default.


## E4M4 candidate and lightweight option screens

Following the request to favor many options over broad testing, the next stage
used four full-width rows of layer 29 and 256 evenly subsampled actual positions
per fit/development split. Eight encoding/weighting combinations, nine local
covariance/code-update combinations, six gain granularities, two full covariance
encodings, and three matched damping choices for both custom/native candidates
were screened. Duplicate baselines are controls, not independent successes.
These are idea-selection measurements, not a fresh quality holdout.

The best simple option was activation-weighted unsigned E4M4. More local gain
parameters did not justify their extra bytes with the tested fitter. Local
covariance/code updates often lowered fitting loss but worsened development
loss. Full covariance did not establish a win over native ScaleSearch v2.
At that stage, the best screened custom development error was 0.005038 versus 0.004406 for the
best equally tuned native control (about 14.3% higher). No runtime advantage has
been measured, so the custom format is **not yet established as viable**.

### Implemented experimental option: AGQ4M001

The explicit `e4m4` option retains the INT4/G32/shared-offset-G64/row-gain layout,
replacing only the scale-byte interpretation:

```
scale(byte) = (1 + (byte & 15)/16) * 2^((byte >> 4) - 7)
```

All 256 byte values are finite positive scales, from 1/128 to 496. The scale
has no sign bit. It can be decoded by constructing the FP32 exponent and
mantissa directly. Artifact magic is `AGQ4M001`; payload order, checksum and
byte counts are unchanged. E8M0 `AGQ4E001` artifacts remain readable, and E8M0
remains the default option. The public Swift `scaleBytes` field names the raw
bytes without incorrectly implying that every encoding contains only exponents.

The calibrated fitter searches three BF16 row-gain seeds, alternately fits two
G32 slopes and one shared G64 offset using diagonal activation moments, and
scores stored metadata/codes. This is a bounded local search, not a global
optimum. JSON fixtures can supply a nonnegative `importance` vector of K second
moments, specific to the source weights and calibration inputs.

```sh
swift build --configuration release --product afterglow-int4-proof
$(swift build --configuration release --show-bin-path)/afterglow-int4-proof   /tmp/new-e4m4.agq4 cpu MATRIX.json e4m4
```

Five focused codec/CPU checks passed. The canonical release product built and
reproduced the screened artifact byte-for-byte. Both E4M4 Metal pipelines compile;
GPU execution and comparative speed remain unverified. The initial E8M0 Swift
source is preserved under `Benchmarks/CompactInt4Proof20261002/OriginalSwiftCode`.
Current screens and code live under `Benchmarks/CompactInt4ActualInputs20261002`.
An initial full-covariance screen was invalidated after detecting a source-array
alias in the exploratory script; its replacement copies the source and checks
immutability. Do not use the invalidated report.


## Signed-scale candidate: AGQ4S001

Further cheap screens retained the same four rows, 256 positions per split and
source-family separation. They explored scale-bit allocation, signed scale
orientation, local covariance shrinkage, offset recentering, integer-code updates,
independent G32 fractional offsets, ScaleSearch initialization, and full-covariance
metadata refitting. Most extra fitting lowered fit loss without a development gain.
Independent fractional offsets did not justify replacing the shared BF16 offset.
The best signed-scale candidate used ten local fitting rounds, three BF16 row-gain
seeds, all four initial G32 sign pairs, and covariance off-diagonals shrunk to 0.3
plus a 0.01 diagonal ridge. Selection within the fitter uses only fit inputs.

| Candidate | Dev relative output MSE | Payload bits/weight |
| --- | ---: | ---: |
| Native ScaleSearch v2, best prior damping screen | 0.004405899 | 4.5 |
| Signed E3M4 + local covariance shrinkage 0.3 | 0.004379417 | 4.502976 |
| Signed E3M4 + fixed native codes, nine gain phases | 0.004482630 | 4.502976 |
| Signed E3M4 + full-covariance metadata refit, ridge 0.3 | 0.004455352 | 4.502976 |

The roughly 0.6% edge of the best signed result is too small, on a repeatedly used
idea-selection fixture, to establish a quality win. It supports implementing the
packed arithmetic candidate for an eventual runtime comparison, not a broad model
campaign. The format remains **experimental; viability is not yet established**.

`AGQ4S001` retains exactly the existing payload order, signed INT4 codes, shared
BF16 offset per G64, BF16 row gain, and checksum. Only the scale byte changes:

```
scale(byte) = (byte & 128 ? -1 : 1)
              * (1 + (byte & 15)/16)
              * 2^(((byte >> 4) & 7) - 3)
```

All 256 values are finite nonzero signed scales, with magnitudes 1/8 through 31.
This is a custom scale encoding, not a standard FP8 format. Negative scales permit
either orientation of the asymmetric signed INT4 code range. CPU and Metal decode
by constructing the FP32 sign, exponent and mantissa bits. E8M0 remains the default;
all three artifact variants are explicitly identified and readable.

The covariance fitter is currently the frozen Python arithmetic prototype
`Benchmarks/CompactInt4ActualInputs20261002/Code/export_signed_candidate.py`.
It exports packed codes and exact stored metadata, and asserts that the artifact
grid equals the screened grid. The convenience Swift fitter explicitly rejects
`.e3m4`; it does not substitute its diagonal-only algorithm for the experimental
local covariance fitter. Swift supplies strict artifact reading/writing and packed
CPU/Metal execution for this encoding.

```sh
# Requires the existing pinned source weights and activation fixtures; no download.
python3 Benchmarks/CompactInt4ActualInputs20261002/Code/export_signed_candidate.py /tmp/new-signed-candidate
swift build --configuration release --product afterglow-int4-proof
$(swift build --configuration release --show-bin-path)/afterglow-int4-proof verify /tmp/new-signed-candidate/weights.agq4 compile-metal Benchmarks/CompactInt4ActualInputs20261002/signed-candidate/matrix.json
```

`verify` reads an existing artifact without modifying it. Optional `MATRIX.json`
supplies the original weights for reporting reconstruction error; without it, the
report uses null for that metric. It checks packed CPU arithmetic against a dense
Double oracle. `compile-metal` compiles both backends without GPU dispatch; `metal`
requests explicit execution. Six focused CPU codec checks, including all 256
signed scale values, passed in an isolated package. The real four-row packed
artifact's CPU maximum absolute oracle error was 9.82e-7. Both signed Metal
pipelines compiled and dispatched successfully through the canonical release
product. Each backend agreed with the FP16-input CPU reference within 8.35e-7
maximum absolute error. The separate timing process was confirmed absent before
this brief dispatch. Comparative runtime speed remains unmeasured.

Screens, packed artifacts, source hashes and proof output are preserved in
`Benchmarks/CompactInt4ActualInputs20261002`. The prior E4M4 Swift source is frozen
under `e4m4-candidate/SwiftCode` so its earlier hash manifest remains reproducible.
The Python `validate-compact-int4.py` helper still accepts only the original E8M0
format; use the Swift `verify` command for other encodings.


## Runtime option screens, 2 October 2026

The packed-word MLX NAX loader was the first promising runtime option. At N=2048, K=5376 and M=128, its two tile variants beat the paired current-project MLX control by 1.4–1.7 times. At representative projection geometry N=16384, that advantage disappeared. The best M=128 variant remained about 5% slower. Runtime viability is not established.

| Inputs M | Tile BM | Candidate median ms | Project control median ms |
| --- | --- | --- | --- |
| 16 | 32 | 1.136 | 0.974 |
| 16 | 64 | 1.695 | 0.944 |
| 128 | 32 | 3.209 | 2.382 |
| 128 | 64 | 1.226 | 1.171 |

Each screen used one warmup and three paired resident samples in AB, BA, AB order. Four fitted rows were physically repeated to reach the stated geometry. The candidate passed its FP16-weight dense CPU oracle; the full-shape control was not independently numerically qualified. These screens establish neither model quality nor cold-cache or serving speed. An earlier Python MLX 0.31.2 control was substantially slower than the current project backend and cannot support a claim of improvement over the project.

The project backend uses pinned mlx-swift with local dependency patches. It must not be described as unmodified stock MLX. The final screen records dependency revisions, dirty state and current source hashes, with that recording scope stated explicitly. MLX NAX helper imports retain their MIT license and provenance under runtime-screen-v10/Code/mlx-import.

The packed-word loader reconstructs only a threadgroup weight tile and reuses scale, offset and gain across eight packed values. It remains a research kernel. The library backend, serving defaults and published formats were not promoted. Decode performance, broader calibration quality and a reliable runtime benefit still need convincing evidence.

Reproduce the final screen from the Afterglow root:

    swift build --configuration release --product afterglow-int4-runtime-screen --jobs 2 -Xswiftc -enable-testing
    $(swift build --configuration release --show-bin-path)/afterglow-int4-runtime-screen Benchmarks/CompactInt4ActualInputs20261002/runtime-fixture Benchmarks/CompactInt4ActualInputs20261002/runtime-screen-v12/Code/CompactInt4.metal --packed-nax-options-only --rows 16384

Evidence: Benchmarks/CompactInt4ActualInputs20261002/runtime-screen-v12/summary.json. Fixture construction and control scripts are preserved alongside the shader; the unchanged small fixture is in runtime-fixture. The next step is further inexpensive option exploration, not a full-model campaign.


## Fixed normal codebook option

Eight cheap arithmetic options compared symmetric integer, FP4 E2M1, normal-midpoint and sinh codebooks with diagonal or shrunken local covariance fitting. The normal-midpoint option led on the reused four-row middle-layer fixture, but its local-only fit lost to native ScaleSearch on one preselected layer0 transfer fixture. This motivated a full covariance error-carrying fit with the same fixed codebook and damping .03.

The full covariance option uses 16 fixed FP16 values, unsigned E4M4 scales per 32 weights, a BF16 offset per 64 weights and a BF16 row gain. Its packed tensor cost is 4.5 + 16/K bits per weight, with a fixed 32-byte codebook constant. The exact exported-grid middle-layer error matches the arithmetic screen.

| Tiny fixture | Normal16 output MSE | Native v2 output MSE | Reduction |
| --- | --- | --- | --- |
| Layer29, original four rows | 0.003694 | 0.004406 | 16.2% |
| Layer0, four preselected other rows | 0.0005683 | 0.0008010 | 29.1% |

These are relative output MSEs on 256 reused development positions, with family-disjoint fit/dev captures. The layer0 option used the already selected approach, with no parameter sweep on that fixture. It remains a small transfer screen, not a new quality holdout or model-quality proof. The normal codebook itself is not claimed as a novel invention.

The research runtime harness accepts this option via --normal16 --packed-nax-options-only. It reads normal16.json, a research payload with no published/library artifact format, and decodes a packed word through the fixed codebook before loading a small FP16 threadgroup tile. It does not relabel the normal payload as an INT4 artifact. Both tile variants passed the dense CPU oracle at full projection geometry, with maximum absolute FP16 output difference 0.00006104.

The one-warmup/three-pair timing screen showed large within-run settling changes. BM32 medians cannot establish a speed win; the BM64 M=128 median was about 11% slower than the paired current-project control. No reliable runtime advantage is established. Production CompactInt4 codecs, serving defaults and native INT4 backends remain unchanged. Next useful work is loader options and one reliable timing screen for this promising accuracy candidate, not a full-model campaign yet.

Evidence: codebook-option-screen.json, codebook-transfer-screen.json, codebook-full-cov-layer29.json, codebook-full-cov-layer0.json and runtime-screen-v13-normal16 under Benchmarks/CompactInt4ActualInputs20261002. Reproducible fitting/export scripts live in its Code folder. Keep output paths new; the exporter currently supports only the screened layer29 runtime fixture.


## Targeted runtime options and MLX integration

The next small round tried three NAX tile layouts and three SIMD decode layouts. Each option retained only three measured paired samples. A 100ms unmeasured alternating settling pass addressed the previous run's clock ramp; it is recorded separately from the measured samples. Both the candidate and the project control now receive numerical checks at the stated full projection geometry. The control relative RMSE against its float-grid reference is about 0.000217 for M=128 and 0.000758 for M=1, below the 0.003 screen bound.

At M=128, BM128/BN64 was close to parity in the settled direct-queue screen: candidate 0.745ms versus control 0.781ms. Wider BN128 was slower. These tiny timing differences are insufficient to establish a reliable win. The best direct-queue decode layouts remained about 20% slower in wall time; row layouts 2 and 4 beat layout 8.

The harness now also runs the packed codebook through MLXFast.metalKernel, using MLX's own array allocation, encoder and synchronization path. Decode uses the same word loader and fixed lookup table; prefill imports the same licensed MLX NAX helpers and reconstructs only a small FP16 threadgroup tile. These paths are research options outside the production library and serving defaults.

Both MLX-integrated decode layouts and both prefill layouts passed their CPU numerical oracles. Decode FP16 output error was zero on the fixed input; prefill maximum absolute difference was 0.00006104. A separate model conversion was active during the integrated screens, so their timings remain exploratory and cannot establish runtime viability. The conversion was observed running at roughly 66–109% CPU; GPU overlap was not measured. Do not report those screens as uncontended benchmarks.

Research flags, used with --normal16 --packed-nax-options-only --rows 16384 --settle-gpu:

* --nax-wide-options-only: three full-geometry M=128 tile layouts.
* --normal-decode-options-only: three direct-queue M=1 row layouts.
* --mlx-normal-decode-options-only: two decode layouts through MLXFast.
* --mlx-normal-prefill-options-only: two prefill layouts through MLXFast.

The normal16 fixture is runtime-screen-v13-normal16/fixture. Wide and MLX prefill screens use the frozen wide shader; direct decode requires the shader containing compact_normal16_decode. Full sources and raw samples are preserved under runtime-screen-v14-wide through runtime-screen-v17-mlx-prefill. An initial decode compilation failure is recorded separately and contains no timing evidence. Earlier binary identities were not retained; the final MLX-prefill executable and current dependency state are recorded with their explicit scope.

The remaining proof is still open: a stable competitive runtime result and quality evidence beyond the reused tiny projection fixtures. This round establishes that the promising packed representation can execute through MLX itself, not just through an isolated Metal command queue.


## Canonical normal16 artifact and additional small checks

The stronger accuracy candidate now has a canonical research envelope, AGQ4N001. It uses the same row-major low-nibble-first layout and checksum as the earlier envelopes, with a distinct fixed FP16 codebook. Unsigned E4M4 scales belong to G32, BF16 offsets to G64, and each row has one positive BF16 gain. The codebook's 16 half-bit patterns are fixed by the format version, not stored per group. Payload cost remains 4.5 + 16/K bits per weight; the file envelope adds 24 bytes.

Swift construction selects CodeEncoding.normal16 and ScaleEncoding.e4m4. Other scale encodings are rejected for this codebook. CPU packed multiplication, strict artifact reading/writing and the explicit packed Metal backend support the format. Native signed-INT4 TensorOps cannot decode it and is rejected explicitly; the proof command reports that unsupported backend. Existing integer formats retain their defaults. The Python covariance exporter now emits the canonical artifact directly through Scripts/compact_normal16_artifact.py; a frozen payload produces exactly the same bytes as the verified Swift envelope. Cubic codebooks cannot be mislabeled AGQ4N001.

Seven focused codec checks passed in 0.045 seconds, including all 16 normal levels, independent scales, shared offset/gain, artifact round trip and rejection of an incompatible native-INT4 backend. The actual four-row canonical candidate passed CPU and packed Metal oracles with maximum absolute errors approximately 0.000000562 and 0.000000596. The artifact has 12,104 payload bytes and 12,128 file bytes. Its fixed metadata is the same candidate previously screened, with no refitting during packaging.

One frozen-candidate check used activation positions 1 modulo 16 rather than the positions 0 modulo 16 used in option selection. Relative output MSE was 0.004599 for normal16 versus 0.005490 for native v2, about 16.2% lower; FP16 tile rounding gave 0.004602. All four per-row errors improved. Those positions share the original capture corpus and neighboring tokens are correlated, so they are not an independent corpus holdout.

One additional transfer check used preselected upper-half rows [9169,10231,13779,15907] from layer59, absent from earlier option selection, and positions 1 modulo 16. With fixed codebook, damping .03 and fitting procedure, normal16 error was 0.001337 versus 0.001838 for native v2, about 27.3% lower. Its FP16 tile error was 0.001336. This extends the small projection evidence to three layers, without claiming full-model quality.

The targeted decode round tried packed register lookup, SIMD shuffle lookup, compiling both candidate and control, cubic codebook arithmetic and multiple SIMD groups per threadgroup. No reliable latency win emerged. A cubic codebook retained about 12% lower error than native on the original tiny fixture, but its arithmetic decoder did not close the gap. The first cubic selector accidentally ran lookup-only cases; that result is scoped separately and never used as formula evidence.

The canonical artifact was also read directly into the MLX prefill harness, so JSON is no longer required for the normal format. No model process was observed at the pre/post checks. At M128,N16384,K5376, BM64 median was 1.230ms versus native 1.002ms; BM128 was 0.864ms versus 0.718ms. Numerical oracles passed, but this approximately 20–23% latency cost leaves runtime viability unproven. Physical geometry still repeats four fitted rows; no full-model or cold-cache claim is implied.

Canonical proof:

    swift build --configuration release --product afterglow-int4-proof --jobs 2 -Xswiftc -enable-testing
    $(swift build --configuration release --show-bin-path)/afterglow-int4-proof verify Benchmarks/CompactInt4ActualInputs20261002/normal16-candidate/weights.agq4 metal Benchmarks/CompactInt4ActualInputs20261002/normal16-candidate/matrix.json

Canonical MLX path:

    swift build --configuration release --product afterglow-int4-runtime-screen --jobs 2 -Xswiftc -enable-testing
    $(swift build --configuration release --show-bin-path)/afterglow-int4-runtime-screen Benchmarks/CompactInt4ActualInputs20261002/normal16-candidate/runtime-fixture Benchmarks/CompactInt4ActualInputs20261002/normal16-candidate/Code/CompactInt4Wide.metal --normal16 --packed-nax-options-only --mlx-normal-prefill-options-only --rows 16384 --settle-gpu

Evidence and exact sources are under normal16-candidate, normal16-unseen-positions.json, normal16-new-layer-screen.json, polynomial-codebook-screen.json and runtime-screen-v18-lookup through runtime-screen-v21-cubic-groups. Serving defaults and model checkpoints were not promoted or published. The next runtime lead is direct NAX fragment loading to avoid the current threadgroup weight-tile barriers and copies.


## Direct NAX fragment options

Two further small options removed the threadgroup weight tile and its barriers. Both load the packed codebook values directly into MLX NAX fragments using BaseNAXFrag.get_coord and the licensed helper's fragment layout. The K32 option decodes each fragment directly. The K64 option caches row gain and reuses shared offset and scale metadata across four K fragments.

Both passed the existing dense FP16-weight oracle, with maximum absolute FP16 output difference 0.00006104. Neither improved latency. K32 BM64 median was 0.955ms versus native 0.764ms; BM128 was 1.103ms versus 0.765ms. K64 metadata reuse was slower still: 1.734ms versus 0.992ms at BM64 and 2.605ms versus 1.467ms at BM128. Each option retained only three measured pairs. These are repeated-row microbenchmarks, not model-level latency evidence.

The results rule out the simple assumption that removing the temporary weight tile necessarily helps. Direct loading repeats decode work across M partitions; larger K fragments also increase register demand. Those are plausible explanations, not profiler-confirmed causes. Both options remain research evidence under runtime-screen-v22-direct-fragments and runtime-screen-v23-direct-fragments-k64, and were not promoted over the earlier tile loader.

The next useful direction is a broader but still small frozen-candidate accuracy check and a reusable MLX layer integration. The codebook/metadata tradeoff may be worthwhile for quality even if it does not win raw kernel latency; its model-level value remains unproven.
