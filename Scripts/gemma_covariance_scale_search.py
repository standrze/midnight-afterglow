"""Research primitive: native affine Q4 scale search with GPTQ error compensation.

Consumes actual linear input samples, including expert-conditioned samples when
used for MoE. No checkpoint loader, model capture, or default conversion policy.
Search uses the inverse-Hessian sequential error objective, not held-out quality.
"""
import math

FACTORS = (1.0, 0.75, 0.8125, 0.875, 0.9375, 1.0625, 1.125, 1.1875, 1.25)


def _finite(mx, value, label):
    if not bool(mx.all(mx.isfinite(value)).item()):
        raise ValueError(f"non-finite {label}")


def inverse_factor(inputs, damping=0.01):
    import mlx.core as mx
    if inputs.ndim != 2 or min(inputs.shape) == 0:
        raise ValueError("inputs must be a nonempty samples-by-channels matrix")
    if not math.isfinite(damping) or damping <= 0:
        raise ValueError("damping must be positive and finite")
    x = inputs.astype(mx.float32)
    _finite(mx, x, "inputs")
    h = x.T @ x / x.shape[0]
    _finite(mx, h, "covariance")
    indices = mx.arange(h.shape[0])
    observed = mx.diag(h) > 0
    h[indices, indices] = mx.where(observed, mx.diag(h), 1)
    absolute = damping * mx.mean(mx.diag(h))
    h[indices, indices] += absolute
    lower = mx.linalg.cholesky(h)
    factor = mx.linalg.cholesky(mx.linalg.cholesky_inv(lower), upper=True)
    mx.eval(factor)
    _finite(mx, factor, "inverse covariance factor")
    return factor, {"samples": x.shape[0], "channels": x.shape[1],
                    "unobserved_channels": int(mx.sum(~observed).item()),
                    "relative_damping": damping, "absolute_damping": float(absolute.item())}


def _simulate(mx, source, factor, scale, bias):
    work = source.astype(mx.float32)
    codes = mx.zeros(source.shape, dtype=mx.uint32)
    errors = mx.zeros_like(work)
    s, b = scale.astype(mx.float32), bias.astype(mx.float32)
    safe = mx.where(s != 0, s, 1)
    for column in range(source.shape[1]):
        value = work[:, column:column + 1]
        code = mx.where(s != 0, mx.clip(mx.round((value - b) / safe), 0, 15), 0).astype(mx.uint32)
        error = (value - (code.astype(mx.float32) * s + b)) / factor[column, column]
        codes[:, column:column + 1] = code
        errors[:, column:column + 1] = error
        work[:, column:] -= error @ factor[column:column + 1, column:]
        mx.eval(work, codes, errors)
    return codes, errors, mx.sum(errors * errors, axis=1, keepdims=True)


def _quantize_pass(weight, factor, group_size=64, factors=FACTORS, refinement_steps=0):
    """Greedy group search; no guarantee of improved whole-layer/model quality.

    Factor 1 preserves native scales/biases exactly. Alternative scales and
    centered biases are rounded to the source storage dtype before code fitting.
    Each output row chooses independently; groups remain contiguous and uniform.
    """
    import mlx.core as mx
    if weight.ndim != 2 or min(weight.shape) == 0 or weight.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        raise ValueError("weight must be a nonempty BF16/F16/F32 matrix")
    if group_size not in (64, 128) or weight.shape[1] % group_size:
        raise ValueError("native group geometry mismatch")
    if factor.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("factor/weight geometry mismatch")
    factors = tuple(factors)
    if not factors or factors[0] != 1.0 or any(not math.isfinite(f) or f <= 0 for f in factors):
        raise ValueError("factors must begin with 1 and be positive and finite")
    _finite(mx, weight, "weight")
    _finite(mx, factor, "factor")
    if not bool(mx.all(mx.diag(factor) > 0).item()):
        raise ValueError("factor diagonal must be positive")
    if not bool(mx.all(mx.tril(factor, -1) == 0).item()):
        raise ValueError("factor must be upper triangular")
    corrected = weight.astype(mx.float32)
    all_codes, scales, biases, decisions = [], [], [], []
    for start in range(0, weight.shape[1], group_size):
        end = start + group_size
        group = corrected[:, start:end]
        local_factor = factor[start:end, start:end].astype(mx.float32)
        _, base_s, base_b = mx.quantize(group.astype(weight.dtype), group_size=group_size, bits=4)
        center = base_b.astype(mx.float32) + base_s.astype(mx.float32) * 7.5
        best_codes, best_errors, best_loss = _simulate(mx, group, local_factor, base_s, base_b)
        original_loss = best_loss
        best_s, best_b = base_s, base_b
        chosen = mx.zeros((weight.shape[0], 1), dtype=mx.int32)
        for index, multiplier in enumerate(factors[1:], 1):
            s = (base_s.astype(mx.float32) * multiplier).astype(weight.dtype)
            b = (center - s.astype(mx.float32) * 7.5).astype(weight.dtype)
            codes, errors, loss = _simulate(mx, group, local_factor, s, b)
            improve = loss < best_loss
            best_codes = mx.where(improve, codes, best_codes)
            best_errors = mx.where(improve, errors, best_errors)
            best_s, best_b = mx.where(improve, s, best_s), mx.where(improve, b, best_b)
            chosen, best_loss = mx.where(improve, index, chosen), mx.where(improve, loss, best_loss)
            mx.eval(best_codes, best_errors, best_s, best_b, chosen, best_loss)
        refined_rows = mx.zeros((weight.shape[0], 1), dtype=mx.bool_)
        if refinement_steps:
            # residual = errors @ U, so ||residual @ inv(U)||² is the
            # conditional objective. Solve once for this group's transform.
            transform = mx.linalg.solve_triangular(
                local_factor, mx.eye(group_size), upper=True, stream=mx.cpu)
            target = group @ transform
            ones = mx.ones((1, group_size)) @ transform
            for _ in range(refinement_steps):
                feature = best_codes.astype(mx.float32) @ transform
                cc = mx.sum(feature * feature, axis=1, keepdims=True)
                ct = mx.sum(feature * ones, axis=1, keepdims=True)
                tt = mx.sum(ones * ones)
                cy = mx.sum(feature * target, axis=1, keepdims=True)
                ty = mx.sum(ones * target, axis=1, keepdims=True)
                determinant = cc * tt - ct * ct
                valid = determinant > 1e-6 * mx.maximum(cc * tt, 1e-30)
                denominator = mx.where(valid, determinant, 1)
                fitted_s = mx.where(valid, (cy * tt - ty * ct) / denominator, best_s)
                fitted_b = mx.where(valid, (ty * cc - cy * ct) / denominator, best_b)
                candidates = [(fitted_s.astype(weight.dtype), fitted_b.astype(weight.dtype))]
                # Exact adjacent stored BF16 values, including signed scales,
                # zeros and subnormals; never approximate a ULP by a percentage.
                if weight.dtype == mx.bfloat16:
                    for direction in (-1, 1):
                        candidates.append((_bf16_neighbor(mx, best_s, direction), best_b))
                        candidates.append((best_s, _bf16_neighbor(mx, best_b, direction)))
                any_improvement = False
                for s, b in candidates:
                    finite = mx.isfinite(s) & mx.isfinite(b)
                    s, b = mx.where(finite, s, best_s), mx.where(finite, b, best_b)
                    codes, errors, loss = _simulate(mx, group, local_factor, s, b)
                    improve = finite & (loss < best_loss)
                    any_improvement |= bool(mx.any(improve).item())
                    best_codes = mx.where(improve, codes, best_codes)
                    best_errors = mx.where(improve, errors, best_errors)
                    best_s, best_b = mx.where(improve, s, best_s), mx.where(improve, b, best_b)
                    best_loss = mx.where(improve, loss, best_loss)
                    refined_rows |= improve
                    mx.eval(best_codes, best_errors, best_s, best_b, best_loss)
                if not any_improvement:
                    break
        corrected[:, end:] -= best_errors @ factor[start:end, end:].astype(mx.float32)
        all_codes.append(best_codes)
        scales.append(best_s)
        biases.append(best_b)
        decisions.append({"group_start": start, "native_conditional_error": float(mx.sum(original_loss).item()),
                          "selected_conditional_error": float(mx.sum(best_loss).item()),
                          "refined_rows": int(mx.sum(refined_rows).item()),
                          "factor_counts": [int(mx.sum(chosen == i).item()) for i in range(len(factors))]})
        mx.eval(corrected)
    codes = mx.concatenate(all_codes, axis=1)
    shifts = mx.arange(0, 32, 4, dtype=mx.uint32)
    packed = mx.sum(codes.reshape(weight.shape[0], -1, 8) << shifts, axis=-1).astype(mx.uint32)
    result = packed, mx.concatenate(scales, axis=1), mx.concatenate(biases, axis=1)
    mx.eval(result)
    for value in result:
        _finite(mx, value, "quantized result")
    return result, {"algorithm": "greedy-native-affine-q4-covariance-scale-search-v1",
                    "bits": 4, "group_size": group_size, "factors": list(factors),
                    "metadata_dtype": str(weight.dtype), "tensor_bytes": sum(v.nbytes for v in result),
                    "decisions": decisions, "model_quality_measured": False}


def _bf16_neighbor(mx, value, direction):
    bits = value.view(mx.uint16).astype(mx.int32)
    magnitude = bits & 0x7fff
    negative = (bits & 0x8000) != 0
    step = mx.where(negative, -direction, direction)
    adjacent = mx.where(magnitude == 0, 1 if direction > 0 else 0x8001, bits + step)
    return adjacent.astype(mx.uint16).view(mx.bfloat16)


def _row_objective(mx, weight, factor, result, group_size):
    packed, scales, biases = result
    shifts = mx.arange(0, 32, 4, dtype=mx.uint32)
    codes = ((packed[..., None] >> shifts) & 15).reshape(weight.shape).astype(mx.float32)
    residual = weight.astype(mx.float32) - (
        codes * mx.repeat(scales.astype(mx.float32), group_size, axis=1)
        + mx.repeat(biases.astype(mx.float32), group_size, axis=1))
    # Avoid materializing another K-by-K inverse. E U = residual.
    errors = mx.linalg.solve_triangular(
        factor.astype(mx.float32).T, residual.T, upper=False, stream=mx.cpu).T
    return mx.sum(errors * errors, axis=1, keepdims=True)


def quantize(weight, factor, group_size=64, factors=FACTORS, *, refinement_steps=2):
    """Refine stored metadata and retain the legacy full-row candidate.

    The no-worse guard is per row on the supplied damped covariance objective,
    evaluated on the exact stored affine grid in FP32. It is not a claim about
    held-out inputs, BF16 matmul rounding, or assembled model quality.
    Set refinement_steps=0 to reproduce v1 exactly.
    """
    import mlx.core as mx
    if isinstance(refinement_steps, bool) or not isinstance(refinement_steps, int) or not 0 <= refinement_steps <= 8:
        raise ValueError("refinement_steps must be an integer from 0 through 8")
    factors = tuple(factors)
    baseline, baseline_report = _quantize_pass(weight, factor, group_size, factors)
    if refinement_steps == 0:
        return baseline, baseline_report
    refined, report = _quantize_pass(weight, factor, group_size, factors, refinement_steps)
    original_loss = _row_objective(mx, weight, factor, baseline, group_size)
    proposed_loss = _row_objective(mx, weight, factor, refined, group_size)
    accept = mx.isfinite(proposed_loss) & (proposed_loss < original_loss)
    result = tuple(mx.where(accept, new, old) for new, old in zip(refined, baseline))
    mx.eval(result)
    report.update(algorithm="greedy-native-affine-q4-covariance-scale-search-v2",
                  refinement_steps=refinement_steps,
                  refined_pass_decisions=report.pop("decisions"),
                  decisions=baseline_report["decisions"],
                  decisions_scope="legacy pass; refined pass diagnostics are separate from final row selection",
                  baseline_row_objective=float(mx.sum(original_loss).item()),
                  selected_row_objective=float(mx.sum(mx.where(accept, proposed_loss, original_loss)).item()),
                  refined_rows_accepted=int(mx.sum(accept).item()),
                  row_selection=accept.reshape(-1).tolist(),
                  objective="FP32 stored grid; supplied inverse Cholesky factor; calibration only")
    return result, report
