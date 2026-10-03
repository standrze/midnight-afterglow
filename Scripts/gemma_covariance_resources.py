"""Allocation planning and guarded entry point for experimental covariance Q4.

Estimates are conservative accounting, not a proven bound on MLX/linalg scratch
or total process footprint. Callers must separately enforce device memory limits
and source/spool disk reserves, and serialize layers/experts.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CovarianceResources:
    samples: int
    input_channels: int
    output_rows: int
    covariance_work_bytes: int
    activation_work_bytes: int
    fitting_work_bytes: int
    planned_work_bytes: int
    budget_bytes: int


def preflight(samples, input_channels, output_rows, budget_bytes, *, metadata_bytes=2):
    for name, value in [('samples', samples), ('input_channels', input_channels),
                        ('output_rows', output_rows), ('budget_bytes', budget_bytes)]:
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if metadata_bytes not in (2, 4):
        raise ValueError('metadata bytes must be 2 or 4')
    if input_channels % 64:
        raise ValueError('input width must support contiguous native G64')
    # Eight FP32 square buffers account for covariance, damping, Cholesky,
    # inverse/factor operations and temporary copies; backend scratch is extra.
    covariance = 8 * input_channels * input_channels * 4
    # Original stored inputs plus a FP32 copy and one further FP32 work buffer.
    activations = samples * input_channels * (metadata_bytes + 8)
    # Source and v1/v2 outputs, corrected weights, code/error work, whitened
    # features and full-row guard residuals. Linalg/backend scratch remains extra.
    fitting = output_rows * input_channels * (metadata_bytes + 16 * 4)
    total = covariance + activations + fitting
    plan = CovarianceResources(samples, input_channels, output_rows,
                               covariance, activations, fitting, total, budget_bytes)
    if total > budget_bytes:
        raise MemoryError(f'planned covariance/fitting work {total} exceeds budget {budget_bytes}; '
                          'reduce captured rows/samples or explicitly choose a larger validated budget; '
                          'input correlations will not be silently replaced with diagonal statistics')
    return plan


def fit(weight, inputs, *, budget_bytes, damping=0.01, group_size=64):
    """Preflight before forming any covariance; caller owns already-loaded arrays."""
    if weight.ndim != 2 or inputs.ndim != 2 or weight.shape[1] != inputs.shape[1]:
        raise ValueError('weight and actual inputs must be compatible matrices')
    # Imports are deferred so policy tests never initialize MLX or its GPU backend.
    import mlx.core as mx
    if weight.dtype not in (mx.bfloat16, mx.float16, mx.float32) or inputs.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        raise ValueError('require finite floating point source weights and inputs')
    if group_size not in (64, 128) or weight.shape[1] % group_size:
        raise ValueError('unsupported native group geometry')
    plan = preflight(inputs.shape[0], inputs.shape[1], weight.shape[0], budget_bytes,
                     metadata_bytes=4 if mx.float32 in (weight.dtype, inputs.dtype) else 2)
    from gemma_covariance_scale_search import inverse_factor, quantize
    factor, coverage = inverse_factor(inputs, damping)
    result, report = quantize(weight, factor, group_size)
    report['resources'] = asdict(plan)
    report['coverage'] = coverage
    report['resource_estimate_is_process_peak_bound'] = False
    return result, report
