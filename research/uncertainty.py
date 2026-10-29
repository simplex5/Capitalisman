"""Fixed-seed paired circular-block bootstrap for sequential error differences."""

import math

import numpy as np


def paired_mae_interval(model_errors, baseline_errors, *, horizon, step, seed=1729, repetitions=2000):
    """Positive differences favor the model; intervals are exploratory, not adjusted.

    Resample paired differences in contiguous circular blocks to retain local
    dependence. Block length covers overlapping targets and a cube-root sample
    heuristic. Too few observations/blocks yields an explicit missing interval.
    """
    model = np.asarray(model_errors, dtype=float)
    baseline = np.asarray(baseline_errors, dtype=float)
    if model.ndim != 1 or model.shape != baseline.shape or not len(model):
        raise ValueError("Paired errors must be nonempty matching one-dimensional arrays")
    if not np.isfinite(model).all() or not np.isfinite(baseline).all():
        raise ValueError("Paired errors must be finite")
    if horizon < 1 or step < 1 or repetitions < 100:
        raise ValueError("Positive horizon/step and at least 100 repetitions required")
    differences = baseline - model
    n = len(differences)
    block = max(math.ceil(horizon / step), math.ceil(n ** (1 / 3)))
    result = {
        "mean_improvement": float(differences.mean()), "observations": n,
        "method": "paired_circular_block_bootstrap", "block_length": block,
        "seed": seed, "repetitions": repetitions, "confidence_level": 0.95,
        "lower": None, "upper": None, "status": "insufficient_observations",
        "multiplicity_adjusted": False,
    }
    if n < 10 or n < 3 * block:
        return result
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions)
    # Bound temporary memory even for long histories.
    for i in range(repetitions):
        starts = rng.integers(0, n, size=math.ceil(n / block))
        indices = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        means[i] = differences[indices].mean()
    result.update(status="estimated", lower=float(np.quantile(means, 0.025)),
                  upper=float(np.quantile(means, 0.975)))
    return result
