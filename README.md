# midnight-afterglow

A native Swift command-line app for decision training, calibration, model
conversion, quantization, evaluation, and verified export for Midnight.

This repository ships one executable: `midnight-afterglow`. Its default interface
runs in the terminal using loom and weft. The app builds independently of Midnight
and Studio; it does not include a web UI, model weights, or research campaigns.

Apple Silicon, macOS 15+, Swift 6.4+, and Xcode's Metal compiler are required.
Evaluation of generated Python code additionally requires Docker. Model operations
run in native Swift/MLX; build-time overlays use versioned patches.

```sh
./build.sh
.build/debug/midnight-afterglow --help
./test.sh
./Scripts/check-swift-format.sh
```

Run `midnight-afterglow` in a terminal for the interactive interface. Press `1` for
training or `2` for quantization, Tab to select fields, Enter to edit, `r` to run,
`c` to cancel, and `q` to quit. Without a terminal, no arguments prints help.

```sh
midnight-afterglow formats
midnight-afterglow quantize /path/to/source /path/to/q4 --standard-q4 --dry-run
```

See [command documentation](Documentation/Commands.md),
[model evaluation](Documentation/model-evaluation.md),
[Swift conventions](Documentation/SwiftProjectStyle.md), and
[source provenance](Documentation/import-manifest.json).
Model-specific adapter conversion is optional; no Nimble download is required.
