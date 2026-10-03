# Pinned dependency overlays

Wick owns the dependency patches in this directory. `../prepare-dependencies.sh`
verifies the exact host-specific MLX, MLX-C, MLX Swift, and MLX Swift LM
revisions before applying them. Re-running preparation recognizes already
applied overlays; an unexpected revision or conflicting edit stops the build.

The patch set covers affine ScaleSearch (centered grids, bias refinement,
joint fitting, and group-size support), portable checkpoint metadata,
Mistral/Gemma compatibility, MLX compile-cache lifetime and Metal kernel
corrections, and the existing Linux CUDA integration. It does not include
Midnight's server, speech, client streaming, or release publishing overlays.

Patch ordering matters: use the preparation script rather than applying files
alphabetically. A plain SwiftPM build without preparation does not supply the
ScaleSearch conversion APIs.

[Extraction provenance](../Docs/extraction-manifest.json) records original
source hashes. [Third-party notices](../THIRD_PARTY_NOTICES.md) preserve upstream
license terms. Review and test overlays again when changing pinned revisions.

- `mlx-swift-lm-bounded-conversion.patch`: opt-in lazy conversion loading, per-module replacement and 512-row ScaleSearch batches. Apply after the group-size and self-contained conversion patches. Default runtime weight loading remains eager.

- `mlx-swift-lm-generic-scale-search-g128.patch`: pass the selected G64/G128
  geometry through generic conversion validation, Linear/SwitchLinear fitting,
  quantized module construction and metadata. Apply after bounded conversion.
  The primitive already supported G128; its callers previously fixed G64.
  Default G64 behavior is unchanged. `GemmaSelectiveGroupConversionTests` covers
  ordinary/searched mixed groups, unchanged norms/other matrices, tied-head export
  and native reload on CPU.

- `mlx-swift-lm-gemma-calibration-access.patch`: expose the existing Gemma 4
  decoder initializer through `GemmaEncoder` SPI and allow subclassing the native
  BF16 `SwitchLinear` call. Instrumentation forwards unchanged native arithmetic.
  Apply after generic G128 support. Metal fixture tests compare observed and native
  dense/MoE outputs exactly, including sorted and unsorted expert routing.

- `mlx-swift-lm-q4-scale-search-g32.patch`: permit affine Q4 ScaleSearch G32 in the existing generic packing/fitting routine. Apply after Q5 support. Q5 geometry remains G64/G128. CPU and Metal fixtures check exact stored-grid fallback, row batching, expert packing, CLI conversion, and reload. This is compatibility support; full-model accuracy and speed remain experimental.
