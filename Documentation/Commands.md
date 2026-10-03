# Afterglow commands

## Optional publisher-adapter conversion

The current `convert` implementation supports Bespoke's Nimble decision-adapter
format. It reads an existing local publisher adapter; no model is downloaded or
loaded merely by running Afterglow. Other PEFT formats are not accepted by this
converter. Native training can initialize its own decision adapter without this
conversion step.

```sh
midnight-afterglow convert --source /path/to/publisher-adapter \
  --revision PUBLISHER_COMMIT_SHA --output /path/to/native-adapter
```

Supply the immutable 40-character publisher revision. The converter checks the
adapter tensor hash against `temperature_config.json`, imports `schema_config.json`,
and maps supported adapter tensors to the native MLX layout. Keep the matching
base checkpoint and publisher notices with the resulting adapter.

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
Quantization runs in process through `midnight-afterglow quantize`.
Quantized decision models need their own accuracy and calibration validation.
