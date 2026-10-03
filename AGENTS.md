# midnight-afterglow

Own training, calibration, checkpoint conversion, quantization, evaluation and
model exports here. Serving and HTTP API code belong in Midnight. This package
must build without sibling checkouts. Preserve pinned dependencies, notices,
import provenance, standard MLX checkpoint layouts and per-module quantization.

Follow Docs/SwiftProjectStyle.md. Format only touched Swift files and verify them
with swift format. Run ./test.sh for meaningful fixture checks. Full-model
validation must record source revisions, precision, memory and quality evidence.

Keep labels outside decision prompts, data splits disjoint by source family,
and calibration specific to the exported weights. Checkpoint/export writes are
transactional; never overwrite source weights or existing runs. Do not publish
weights without explicit authorization. Never log tokens or private raw datasets.
