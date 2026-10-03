#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")" && pwd)"
# Wick owns these pinned overlays and its own dependency checkouts.
DEPENDENCY_PACKAGE_ROOT="$PACKAGE_ROOT"
export MODEL_RUNNER_SCRATCH_PATH="${WICK_SCRATCH_PATH:-${FACET_SCRATCH_PATH:-${MODEL_RUNNER_SCRATCH_PATH:-}}}"
if [[ ! -f "$DEPENDENCY_PACKAGE_ROOT/Package.swift" ]]; then
  echo "Dependency package root must contain Package.swift: $DEPENDENCY_PACKAGE_ROOT" >&2
  exit 2
fi
DEPENDENCY_PACKAGE_ROOT="$(cd "$DEPENDENCY_PACKAGE_ROOT" && pwd -P)"
source "$PACKAGE_ROOT/Scripts/swiftpm-scratch-path.sh"
source "$PACKAGE_ROOT/Scripts/optional-dependency-patch.sh"
source "$PACKAGE_ROOT/Scripts/compile-cache-lifetime-patch.sh"
source "$PACKAGE_ROOT/Scripts/gate-up-slices-patch.sh"
HOST_OS="$(uname -s)"
model_runner_configure_swiftpm_scratch "$DEPENDENCY_PACKAGE_ROOT" "$HOST_OS"
MLX_SWIFT_CHECKOUT="$MODEL_RUNNER_SWIFTPM_SCRATCH_PATH/checkouts/mlx-swift"
MLX_SWIFT_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-cuda-linux.patch"
MLX_SWIFT_GENERATED_HEADER_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-cuda-generated-header.patch"
MLX_SWIFT_MLX32_LINK_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-mlx32-cuda-link.patch"
MLX_SWIFT_CROSS_THREAD_STREAM_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-cross-thread-stream.patch"
MLX_SWIFT_CLEAR_STREAMS_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-clear-streams.patch"
MLX_SWIFT_AFFINE_Q4_QMV_JIT_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-affine-q4-qmv-jit.patch"
MLX_SWIFT_SORTED_GATHER_QMM_NAX_JIT_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-sorted-gather-qmm-nax-row-bounds-jit.patch"
MLX_SWIFT_DARWIN_EXPECTED_REVISION="72f3c3ad8aeee39bfc94f8fbeb446cac89e3a798"
MLX_SWIFT_LINUX_EXPECTED_REVISION="2d2724006b62855c6c2a71df633baf4ee4ad8a0f"
MLX_SOURCE_CHECKOUT="$MLX_SWIFT_CHECKOUT/Source/Cmlx/mlx"
MLX_SOURCE_PATCH="$PACKAGE_ROOT/Patches/mlx-cuda-half-fmod.patch"
MLX_SOURCE_GLOBAL_STREAM_CLEANUP_PATCH="$PACKAGE_ROOT/Patches/mlx-global-stream-cleanup.patch"
MLX_SOURCE_AFFINE_Q4_QMV_PATCH="$PACKAGE_ROOT/Patches/mlx-affine-q4-qmv-specialization.patch"
MLX_SOURCE_SORTED_GATHER_QMM_NAX_PATCH="$PACKAGE_ROOT/Patches/mlx-sorted-gather-qmm-nax-row-bounds.patch"
MLX_SOURCE_DARWIN_EXPECTED_REVISION="1f8e74e3f12f31365464a6867c6579f0e9b29d85"
MLX_SOURCE_LINUX_EXPECTED_REVISION="7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247"
MLX_C_SOURCE_CHECKOUT="$MLX_SWIFT_CHECKOUT/Source/Cmlx/mlx-c"
MLX_C_SOURCE_CLEAR_STREAMS_PATCH="$PACKAGE_ROOT/Patches/mlx-c-clear-streams.patch"
MLX_C_SOURCE_CLEAR_GLOBAL_STREAMS_PATCH="$PACKAGE_ROOT/Patches/mlx-c-clear-global-streams.patch"
MLX_C_SOURCE_DARWIN_EXPECTED_REVISION="c74db5307cc8ce122f48d97ef951b30578674e7f"
MLX_C_SOURCE_LINUX_EXPECTED_REVISION="fba4470b89073180056c9ea46c443051375f7399"
SWIFT_TRANSFORMERS_CHECKOUT="$MODEL_RUNNER_SWIFTPM_SCRATCH_PATH/checkouts/swift-transformers"
SWIFT_TRANSFORMERS_EXPECTED_REVISION="2fa33e1f5e7131a7fc64c28e6d161dcec0d24820"
MLX_SWIFT_LM_CHECKOUT="$MODEL_RUNNER_SWIFTPM_SCRATCH_PATH/checkouts/mlx-swift-lm"
MLX_SWIFT_LM_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-ignore-readmes.patch"
MLX_SWIFT_LM_COREFOUNDATION_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-corefoundation-linux.patch"
MLX_SWIFT_LM_GEMMA4_CACHE_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-gemma4-nonrotating-cache.patch"
MLX_SWIFT_LM_Q4_AFFINE_SCALE_SEARCH_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-q4-affine-scale-search.patch"
MLX_SWIFT_LM_Q4_AFFINE_CENTERED_SCALE_SEARCH_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-q4-affine-centered-scale-search.patch"
MLX_SWIFT_LM_Q4_AFFINE_BIAS_REFINEMENT_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-q4-affine-bias-refinement.patch"
MLX_SWIFT_LM_Q4_AFFINE_JOINT_FIT_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-q4-affine-joint-fit.patch"
MLX_SWIFT_LM_Q4_AFFINE_GROUP_SIZE_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-q4-affine-group-size.patch"
MLX_SWIFT_LM_MISTRAL_HYBRID_ATTENTION_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-mistral-hybrid-attention.patch"
MLX_SWIFT_LM_EXPECTED_REVISION="14414441fa44f45eee35a61e9fa0bab577cf9734"

case "$HOST_OS" in
  Darwin)
    MLX_SWIFT_EXPECTED_REVISION="$MLX_SWIFT_DARWIN_EXPECTED_REVISION"
    MLX_SOURCE_EXPECTED_REVISION="$MLX_SOURCE_DARWIN_EXPECTED_REVISION"
    MLX_C_SOURCE_EXPECTED_REVISION="$MLX_C_SOURCE_DARWIN_EXPECTED_REVISION"
    APPLY_LINUX_DEPENDENCY_PATCHES=0
    MLX_SWIFT_CROSS_THREAD_STREAM_OVERLAY_MODE="$(
      model_runner_mlx_cross_thread_stream_overlay_mode "$HOST_OS"
    )"
    ;;
  Linux)
    MLX_SWIFT_EXPECTED_REVISION="$MLX_SWIFT_LINUX_EXPECTED_REVISION"
    MLX_SOURCE_EXPECTED_REVISION="$MLX_SOURCE_LINUX_EXPECTED_REVISION"
    MLX_C_SOURCE_EXPECTED_REVISION="$MLX_C_SOURCE_LINUX_EXPECTED_REVISION"
    APPLY_LINUX_DEPENDENCY_PATCHES=1
    MLX_SWIFT_CROSS_THREAD_STREAM_OVERLAY_MODE="$(
      model_runner_mlx_cross_thread_stream_overlay_mode "$HOST_OS"
    )"
    ;;
  *)
    echo "Unsupported host for dependency preparation: $HOST_OS" >&2
    exit 1
    ;;
esac

cd "$DEPENDENCY_PACKAGE_ROOT"
# Bash 3.2 treats an empty array expansion as unbound under `set -u`; keep the
# no-scratch path explicit so this preparation step also remains usable on the
# Mac client.
if [[ -n "${MODEL_RUNNER_SWIFT_PACKAGE_SCRATCH_ARGS+configured}" ]]; then
  swift package "${MODEL_RUNNER_SWIFT_PACKAGE_SCRATCH_ARGS[@]}" resolve --quiet
else
  swift package resolve --quiet
fi

verify_checkout_revision() {
  local label="$1"
  local checkout="$2"
  local expected_revision="$3"
  local actual_revision

  if [[ ! -d "$checkout" ]]; then
    echo "$label checkout was not created at $checkout" >&2
    return 1
  fi
  actual_revision="$(git -C "$checkout" rev-parse HEAD)"
  if [[ "$actual_revision" != "$expected_revision" ]]; then
    echo "Refusing to patch unexpected $label revision: $actual_revision" >&2
    echo "Expected: $expected_revision" >&2
    return 1
  fi
}

apply_dependency_patch() {
  local label="$1"
  local checkout="$2"
  local patch_file="$3"

  # G32 changes the Q4 guard inside earlier overlays. Prove the complete follow-up
  # and retain the Q5 implementation before accepting those superseded hunks.
  if [[ "$label" == "mlx-swift-lm Q4 affine group size" || "$label" == "mlx-swift-lm bounded conversion" || "$label" == "mlx-swift-lm generic ScaleSearch G128" || "$label" == "mlx-swift-lm Q5 affine scale search" ]] \
    && git -C "$checkout" apply --reverse --check "$PACKAGE_ROOT/Patches/mlx-swift-lm-q4-scale-search-g32.patch" >/dev/null 2>&1 \
    && grep -Fq 'public func q5AffineScaleSearchQuantized(' "$checkout/Libraries/MLXLMCommon/ModelConversion.swift"; then
    echo "$label patch already applied (Q4 G32 follow-up)."
    return 0
  fi

  # Q5 extends bounded/G128 call sites while preserving the Q4 fitter.
  if [[ "$label" == "mlx-swift-lm Q4 affine group size" || "$label" == "mlx-swift-lm bounded conversion" || "$label" == "mlx-swift-lm generic ScaleSearch G128" ]] \
    && git -C "$checkout" apply --reverse --check "$PACKAGE_ROOT/Patches/mlx-swift-lm-q5-affine-scale-search.patch" >/dev/null 2>&1; then
    echo "$label patch already applied (Q5 ScaleSearch follow-up)."
    return 0
  fi

  # Generic G128 conversion follows the bounded-conversion call sites. A complete
  # reverse check proves this overlay is present before accepting its ancestors.
  if [[ "$label" == "mlx-swift-lm Q4 affine group size" || "$label" == "mlx-swift-lm bounded conversion" ]] \
    && git -C "$checkout" apply --reverse --check "$PACKAGE_ROOT/Patches/mlx-swift-lm-generic-scale-search-g128.patch" >/dev/null 2>&1; then
    echo "$label patch already applied (generic G128 follow-up)."
    return 0
  fi

  # The bounded-conversion follow-up extends the group-size function signature.
  if [[ "$label" == "mlx-swift-lm Q4 affine group size" ]] \
    && git -C "$checkout" apply --reverse --check "$PACKAGE_ROOT/Patches/mlx-swift-lm-bounded-conversion.patch" >/dev/null 2>&1; then
    echo "$label patch already applied (bounded-conversion follow-up)."
    return 0
  fi

  # This checkout may also carry the CUDA diagnostic overlay used on the
  # Linux hosts. That overlay preserves the cache fix while changing its
  # explanatory comment, so a byte-for-byte reverse patch check is not a
  # reliable idempotence test for this one semantic change.
  if [[ "$label" == "mlx-swift-lm Gemma 4 non-rotating cache" ]] \
    && grep -Fq 'try makeAttentionKVCache(parameters: parameters)' \
      "$checkout/Libraries/MLXLLM/Models/Gemma4Text.swift"; then
    echo "$label patch already applied."
    return 0
  fi

  # Early Linux bring-up used the same safe CUDA decode boundary without the
  # later macOS-only async branch. Accept that equivalent state so existing
  # server checkouts can converge without reverting generated dependencies.
  if [[ "$label" == "mlx-swift-lm backend-aware token evaluation" ]] \
    && grep -Fq 'eval([token] + cache.flatMap { $0.state })' \
      "$checkout/Libraries/MLXLMCommon/Evaluate.swift" \
    && ! grep -Fq 'if tokenCount % 256 == 0' \
      "$checkout/Libraries/MLXLMCommon/Evaluate.swift"; then
    echo "$label patch already applied."
    return 0
  fi

  # The centered-grid follow-up intentionally refines the original Q4 affine
  # scale-search hunk. Recognize both semantic states directly because the
  # follow-up means reverse-applying the first patch is no longer byte-exact.
  if [[ "$label" == "mlx-swift-lm Q4 affine scale search" ]] \
    && grep -Fq 'case q4AffineScaleSearch' \
      "$checkout/Libraries/MLXLMCommon/ModelConversion.swift"; then
    echo "$label patch already applied."
    return 0
  fi
  if [[ "$label" == "mlx-swift-lm Q4 affine centered scale search" ]] \
    && grep -Fq 'q4AffineScaleSearchFactors' \
      "$checkout/Libraries/MLXLMCommon/ModelConversion.swift"; then
    echo "$label patch already applied."
    return 0
  fi
  if [[ "$label" == "mlx-swift-lm Q4 affine bias refinement" ]] \
    && grep -Fq 'least-squares affine bias' \
      "$checkout/Libraries/MLXLMCommon/ModelConversion.swift"; then
    echo "$label patch already applied."
    return 0
  fi
  if [[ "$label" == "mlx-swift-lm Q4 affine joint fit" ]] \
    && grep -Fq 'let secondCodeValues' \
      "$checkout/Libraries/MLXLMCommon/ModelConversion.swift"; then
    echo "$label patch already applied."
    return 0
  fi

  # The diagnostic follow-up inserts calls inside the scheduling patch's
  # verifier hunk. Recognize both durable semantic states directly so an
  # already fully overlaid checkout remains idempotent in forward order.
  if [[ "$label" == "mlx-swift-lm MTP decode scheduling" ]] \
    && grep -Fq 'if processor == nil, sampler is ArgMaxSampler {' \
      "$checkout/Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift" \
    && grep -Fq 'private var passthroughPipelinePrimed = false' \
      "$checkout/Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift"; then
    echo "$label patch already applied."
    return 0
  fi
  if [[ "$label" == "mlx-swift-lm MTP first-rejection diagnostic" ]] \
    && grep -Fq 'MODEL_RUNNER_DFLASH_FIRST_REJECTION_DIAGNOSTIC' \
      "$checkout/Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift"; then
    echo "$label patch already applied."
    return 0
  fi

  # The Foundation Models follow-up extends the reasoning-stream switch with
  # richer protocol routing. Its extra cases make reverse-applying the original
  # patch non-exact even though the public Generation.reasoning event and its
  # decoder emission are already present.
  if [[ "$label" == "mlx-swift-lm reasoning stream events" ]] \
    && grep -Fq 'case reasoning(String)' \
      "$checkout/Libraries/MLXLMCommon/Evaluate.swift" \
    && grep -Fq 'emit(.reasoning(text))' \
      "$checkout/Libraries/MLXLMCommon/Evaluate.swift" \
    && grep -Fq 'ReasoningEventEmitter' \
      "$checkout/Libraries/MLXLMCommon/Tool/TokenStreamDecoder.swift"; then
    echo "$label patch already applied."
    return 0
  fi

  # The global-cleanup follow-up intentionally extends the same mlx-c function
  # hunk, so reverse-applying the first patch is no longer a valid idempotence
  # test after both are present. Recognize each exported behavior directly.
  if [[ "$label" == "mlx-c clear streams API" ]] \
    && grep -Fq 'extern "C" int mlx_clear_streams(void)' \
      "$checkout/mlx/c/stream.cpp"; then
    echo "$label patch already applied."
    return 0
  fi
  if [[ "$label" == "mlx-c clear global streams API" ]] \
    && grep -Fq 'mlx::core::gpu::clear_global_streams();' \
      "$checkout/mlx/c/stream.cpp"; then
    echo "$label patch already applied."
    return 0
  fi

  if git -C "$checkout" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    echo "$label patch already applied."
    return 0
  fi
  if ! git -C "$checkout" apply --check "$patch_file" >/dev/null 2>&1; then
    echo "Could not apply $label patch cleanly: $patch_file" >&2
    return 1
  fi
  git -C "$checkout" apply "$patch_file"
  echo "Applied $label patch."
}

# Check pinned upstream revisions before applying local overlays.
verify_checkout_revision "mlx-swift" "$MLX_SWIFT_CHECKOUT" "$MLX_SWIFT_EXPECTED_REVISION"
verify_checkout_revision "mlx-swift-lm" "$MLX_SWIFT_LM_CHECKOUT" "$MLX_SWIFT_LM_EXPECTED_REVISION"
verify_checkout_revision "mlx source" "$MLX_SOURCE_CHECKOUT" "$MLX_SOURCE_EXPECTED_REVISION"
verify_checkout_revision "mlx-c source" "$MLX_C_SOURCE_CHECKOUT" "$MLX_C_SOURCE_EXPECTED_REVISION"

if [[ "$APPLY_LINUX_DEPENDENCY_PATCHES" == "1" ]]; then
  apply_dependency_patch \
    "mlx-swift CUDA Linux integration" "$MLX_SWIFT_CHECKOUT" "$MLX_SWIFT_PATCH"
  apply_dependency_patch \
    "mlx-swift CUDA generated header" "$MLX_SWIFT_CHECKOUT" "$MLX_SWIFT_GENERATED_HEADER_PATCH"
  apply_dependency_patch \
    "mlx-swift MLX 0.32 CUDA link" "$MLX_SWIFT_CHECKOUT" "$MLX_SWIFT_MLX32_LINK_PATCH"
  model_runner_reconcile_optional_dependency_patch \
    "mlx-swift cross-thread stream" \
    "$MLX_SWIFT_CHECKOUT" \
    "$MLX_SWIFT_CROSS_THREAD_STREAM_PATCH" \
    "$MLX_SWIFT_CROSS_THREAD_STREAM_OVERLAY_MODE"
  apply_dependency_patch \
    "mlx-swift clear streams API" \
    "$MLX_SWIFT_CHECKOUT" \
    "$MLX_SWIFT_CLEAR_STREAMS_PATCH"
  apply_dependency_patch \
    "mlx CUDA half fmod" "$MLX_SOURCE_CHECKOUT" "$MLX_SOURCE_PATCH"
  apply_dependency_patch \
    "mlx global stream terminal cleanup" \
    "$MLX_SOURCE_CHECKOUT" \
    "$MLX_SOURCE_GLOBAL_STREAM_CLEANUP_PATCH"
  apply_dependency_patch \
    "mlx-c clear streams API" \
    "$MLX_C_SOURCE_CHECKOUT" \
    "$MLX_C_SOURCE_CLEAR_STREAMS_PATCH"
  apply_dependency_patch \
    "mlx-c clear global streams API" \
    "$MLX_C_SOURCE_CHECKOUT" \
    "$MLX_C_SOURCE_CLEAR_GLOBAL_STREAMS_PATCH"
fi

if [[ "$HOST_OS" == "Darwin" ]]; then
  model_runner_prepare_compile_cache_lifetime "$HOST_OS" "$PACKAGE_ROOT" "$MLX_SWIFT_CHECKOUT"
  model_runner_prepare_gate_up_slices "$HOST_OS" "$PACKAGE_ROOT" "$MLX_SWIFT_LM_CHECKOUT"
  # Upstream d73eb752: clamp large sorted expert-row counts before narrowing.
  # Keep compiled Metal headers and generated JIT shader sources synchronized.
  apply_dependency_patch \
    "mlx sorted gather QMM NAX row bounds" \
    "$MLX_SOURCE_CHECKOUT" \
    "$MLX_SOURCE_SORTED_GATHER_QMM_NAX_PATCH"
  apply_dependency_patch \
    "mlx-swift sorted gather QMM NAX generated JIT" \
    "$MLX_SWIFT_CHECKOUT" \
    "$MLX_SWIFT_SORTED_GATHER_QMM_NAX_JIT_PATCH"
  apply_dependency_patch \
    "mlx affine Q4 QMV specialization" \
    "$MLX_SOURCE_CHECKOUT" \
    "$MLX_SOURCE_AFFINE_Q4_QMV_PATCH"
  apply_dependency_patch \
    "mlx-swift affine Q4 QMV generated JIT" \
    "$MLX_SWIFT_CHECKOUT" \
    "$MLX_SWIFT_AFFINE_Q4_QMV_JIT_PATCH"
fi

if [[ "$APPLY_LINUX_DEPENDENCY_PATCHES" == "1" ]]; then
  apply_dependency_patch "mlx-swift-lm CoreFoundation import" "$MLX_SWIFT_LM_CHECKOUT" "$MLX_SWIFT_LM_COREFOUNDATION_PATCH"
fi
apply_dependency_patch "mlx-swift-lm README warning fix" "$MLX_SWIFT_LM_CHECKOUT" "$MLX_SWIFT_LM_PATCH"
apply_dependency_patch "mlx-swift-lm Gemma 4 non-rotating cache" "$MLX_SWIFT_LM_CHECKOUT" "$MLX_SWIFT_LM_GEMMA4_CACHE_PATCH"
apply_dependency_patch "mlx-swift-lm Mistral hybrid attention" "$MLX_SWIFT_LM_CHECKOUT" "$MLX_SWIFT_LM_MISTRAL_HYBRID_ATTENTION_PATCH"
apply_dependency_patch "mlx-swift-lm Q4 affine scale search" "$MLX_SWIFT_LM_CHECKOUT" "$MLX_SWIFT_LM_Q4_AFFINE_SCALE_SEARCH_PATCH"
apply_dependency_patch "mlx-swift-lm Q4 affine centered scale search" "$MLX_SWIFT_LM_CHECKOUT" "$MLX_SWIFT_LM_Q4_AFFINE_CENTERED_SCALE_SEARCH_PATCH"
apply_dependency_patch "mlx-swift-lm Q4 affine bias refinement" "$MLX_SWIFT_LM_CHECKOUT" "$MLX_SWIFT_LM_Q4_AFFINE_BIAS_REFINEMENT_PATCH"
apply_dependency_patch "mlx-swift-lm Q4 affine joint fit" "$MLX_SWIFT_LM_CHECKOUT" "$MLX_SWIFT_LM_Q4_AFFINE_JOINT_FIT_PATCH"
apply_dependency_patch "mlx-swift-lm Q4 affine group size" "$MLX_SWIFT_LM_CHECKOUT" "$MLX_SWIFT_LM_Q4_AFFINE_GROUP_SIZE_PATCH"
apply_dependency_patch "mlx-swift-lm self-contained conversion metadata" "$MLX_SWIFT_LM_CHECKOUT" "$PACKAGE_ROOT/Patches/mlx-swift-lm-self-contained-conversion.patch"

apply_dependency_patch "mlx-swift-lm bounded conversion" "$MLX_SWIFT_LM_CHECKOUT" "$PACKAGE_ROOT/Patches/mlx-swift-lm-bounded-conversion.patch"

apply_dependency_patch "mlx-swift-lm generic ScaleSearch G128" "$MLX_SWIFT_LM_CHECKOUT" "$PACKAGE_ROOT/Patches/mlx-swift-lm-generic-scale-search-g128.patch"

apply_dependency_patch "mlx-swift-lm Q5 affine scale search" "$MLX_SWIFT_LM_CHECKOUT" "$PACKAGE_ROOT/Patches/mlx-swift-lm-q5-affine-scale-search.patch"
apply_dependency_patch "mlx-swift-lm Q4 ScaleSearch G32" "$MLX_SWIFT_LM_CHECKOUT" "$PACKAGE_ROOT/Patches/mlx-swift-lm-q4-scale-search-g32.patch"

apply_dependency_patch "mlx-swift-lm Gemma calibration access" "$MLX_SWIFT_LM_CHECKOUT" "$PACKAGE_ROOT/Patches/mlx-swift-lm-gemma-calibration-access.patch"

python3 "$PACKAGE_ROOT/Scripts/decision-training-patch.py" "$MLX_SWIFT_LM_CHECKOUT"
