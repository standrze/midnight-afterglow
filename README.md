# midnight-afterglow

Native Swift/MLX model development for Midnight: decision training, calibration,
checkpoint conversion, quantization, evaluation, and verified export.

midnight-afterglow is a standalone package. It builds without Midnight, Wick, or Training
checkouts. Quantization code was copied from Wick; optimizer/checkpoint patterns
were adapted from Training. Originals and existing experiments remain in place.
See `Docs/import-manifest.json` for origin hashes and `LICENSE` for attribution.

## Build and test

Apple Silicon, macOS 15+, Swift 6.4+, and Xcode's Metal compiler are required.

```sh
./build.sh
.build/release/midnight-afterglow --help
./test.sh --filter Decision
```

The pinned dependencies and training overlays are prepared by the build script.
New training uses the operations-based Qwen3.5 recurrence with activation
checkpointing; inference keeps the fused Metal path. CUDA is not yet validated.

## Simple terminal interface

Run `midnight-afterglow` with no arguments in a terminal, or select it explicitly
with `midnight-afterglow ui`. It uses loom for rendering and weft for input,
resize handling, and terminal restoration. Without a terminal, no arguments
prints help; existing subcommands remain scriptable.

Press `1` for decision training or `2` for quantization. Arrows or Tab select a
field; Enter edits it. Type or paste one line, then Enter saves and Esc discards
the edit. Press `r` to start, `c` to cancel, or `q` to quit and cancel the owned
job. Status, elapsed time, and recent native command output appear below the form.
Only one job runs at a time. Source models and existing outputs are preserved.

Training asks for a base model, decision contract, train/development JSONL,
new run folder, and optional initial adapter. Defaults are batch size 1,
one epoch, and 512 prompt tokens. Set resume to yes to use an existing run;
the native trainer still verifies the checkpoint, data, and configuration.
Quantization offers affine 4-bit or 8-bit weights with group size 64, standard
calibration, and optional ScaleSearch for 4-bit weights. Architecture support
and full data validation remain the responsibility of the native commands.

Cancellation terminates only the UI's child. Completed atomic checkpoints are
preserved; the current update may be lost. A child that does not stop within five
seconds is killed. Partial quantization output is not a completed artifact.
For advanced settings, use the regular CLI subcommands.

Verify with `./test.sh --filter AfterglowConsoleTests` and
`python3 Tests/terminal-smoke.py .build/release/midnight-afterglow`.

## Official Nimble bundle

Download with Midnight's native Swift service, which resolves immutable revisions
and preserves existing Hugging Face authentication:

```sh
midnight download nimble-9b
```

This scoped bundle contains the official full-precision Qwen3.5-9B base and
Bespoke's matching adapter, approximately 19.52 GB at the recorded release.
It requires preparation before decision serving:

```sh
midnight-afterglow convert --source ~/.midnight/models/nimble-9b/publisher-adapter \
  --revision bd792f44ec8e265be861bfcdf4e05967ffe0e858 \
  --output ~/.midnight/models/nimble-9b/adapter
midnight --model ~/.midnight/models/nimble-9b/base \
  --adapter ~/.midnight/models/nimble-9b/adapter --name nimble-9b
```

The converter verifies the publisher adapter hash, prompt implementation, LoRA
scaling, and complete tensor mapping. It preserves tokenizer and notices.
A successful download or conversion alone does not establish model quality.

## Train decisions

JSONL examples contain `id`, `source_family`, `context`, `schema`, and `labels`.
Schema fields use `type: boolean` or `type: enum`, `description`, optional
`choices` and `choice_descriptions`. Optional `score_fields` names numeric rubric
enums. Labels never enter the prompt. Use disjoint source families for training,
development, calibration, and final testing.

```sh
midnight-afterglow validate --contract adapter/decision-model.json train.jsonl dev.jsonl
midnight-afterglow train decision --model base --contract adapter/decision-model.json \
  --adapter adapter --data train.jsonl --development dev.jsonl --output runs/pilot
```

Omit `--adapter` to initialize a fresh decision adapter. Defaults are rank 16,
learning rate 5e-5, effective batch size 8 (sequential gradient accumulation),
one epoch, seed 17, and a 2,048-token training limit. The preserved linear
learning-rate schedule spans three epochs. Only candidate logits contribute to
loss. Exported trained adapters reset temperature to 1 pending recalibration.

Use `--stop-after-updates N` for an intentional interruption and `--resume`
with the same paths/options to continue. Resume restores optimizer moments,
schedule position, shuffle order/cursor and per-step seed. Changed model, data,
contract or options are rejected. Checkpoints publish an atomic pointer only
after their immutable tensor/state files are complete.

```sh
midnight-afterglow evaluate decision --model base --adapter runs/pilot/adapter-step-N \
  --contract runs/pilot/adapter-step-N/decision-model.json \
  --data test.jsonl --output results.json
midnight-afterglow calibrate decision --model base --adapter runs/pilot/adapter-step-N \
  --contract runs/pilot/adapter-step-N/decision-model.json --data calibration.jsonl \
  --exclude train.jsonl dev.jsonl test.jsonl --output calibrated.json
midnight-afterglow export --model base --adapter runs/pilot/adapter-step-N \
  --contract calibrated.json --probe request.json --output exports/pilot
```

The export includes base weights, adapter, decision contract and reload probe.
Do not promote a candidate without the documented inference, training, resume,
reload and held-out quality checks. Probabilities are preferences among allowed
answers; test application thresholds on your own labeled data.

## Quantization

```sh
midnight-afterglow formats
midnight-afterglow quantize /path/to/unquantized /path/to/q4 --standard-q4 --dry-run
midnight-afterglow quantize /path/to/unquantized /path/to/q4 --standard-q4
```

Affine, MXFP4, MXFP8 and NVFP4 preparation retain Wick's measured policies.
Quantization is an in-process command. Existing `wick`/`facet` product aliases
remain for imported checks; the public model-development command is `midnight-afterglow`.
Quantized decision models need their own accuracy and calibration validation.


## Gemma covariance research

Bounded source row reading and transactional selective covariance Q4 conversion
are available as research scripts in `Scripts/`. The converter takes a frozen
JSON recipe with source, template, captured fit/development inputs and a new
output directory:

```sh
python3 Scripts/convert-gemma-selective-covariance.py --plan PLAN.json --model 31b --progress NEW-PROGRESS.json
```

Plans protect existing outputs, retain native per-module MLX layouts and bind
full source identity. Conversion is not generated-quality qualification or a
new default. See `Docs/gemma-covariance-research-v1.md` for measured scope,
recovery evidence and active campaign paths.

The active [MLX quantization standard](Docs/mlx-quantization-standard-v2.md) records the Gemma coding/cyber budgets, native compatibility, quality/timing/memory gates and current evidence limits. Historical Wick campaigns remain preserved.

## Automated model evaluation

`midnight-afterglow evaluate model` runs the versioned coding, cybersecurity and
tool-use suite, saves every model response and failure reason, and reports each
category separately. Generated Python runs in bounded Docker containers; tools
use deterministic mocks. See [the evaluation guide](Docs/model-evaluation.md) for
the command, suite contract and report limits.

`midnight-afterglow evaluate batch` runs the suite serially across pinned local
checkpoints. `midnight-afterglow evaluate runtime` adds a native speed campaign
with same-model controls before and after each comparison, full checkpoint
identities, no caching, and fixed work. Runtime plans bind to completed accuracy
reports; speed, serving memory and accuracy remain separate selection gates.
The runtime command has passed process fixtures; its full Gemma campaign is pending.
