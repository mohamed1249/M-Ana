"""Lightweight Bayesian comparisons for experiment proportions."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy import stats


def _validate_count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    return int(value)


def bayesian_proportion_ab_test(
    successes_a: int,
    trials_a: int,
    successes_b: int,
    trials_b: int,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    samples: int = 100_000,
    credible_mass: float = 0.95,
    random_state: Optional[int] = 42,
) -> Dict[str, Any]:
    """Compare two binomial rates using independent Beta posteriors.

    Variant B is treated as the candidate treatment. Individual posterior
    intervals are exact Beta quantiles; the difference interval and probability
    that B is better are estimated by reproducible Monte Carlo sampling.
    """
    successes_a = _validate_count("successes_a", successes_a)
    trials_a = _validate_count("trials_a", trials_a)
    successes_b = _validate_count("successes_b", successes_b)
    trials_b = _validate_count("trials_b", trials_b)
    samples = _validate_count("samples", samples)

    if trials_a <= 0 or trials_b <= 0:
        raise ValueError("Both trial counts must be positive.")
    if not (0 <= successes_a <= trials_a and 0 <= successes_b <= trials_b):
        raise ValueError("Success counts must be between zero and their trial counts.")
    if not np.isfinite(prior_alpha) or prior_alpha <= 0:
        raise ValueError("prior_alpha must be a positive finite value.")
    if not np.isfinite(prior_beta) or prior_beta <= 0:
        raise ValueError("prior_beta must be a positive finite value.")
    if samples < 2:
        raise ValueError("samples must be at least 2.")
    if not 0 < credible_mass < 1:
        raise ValueError("credible_mass must be between 0 and 1.")

    posterior_alpha_a = prior_alpha + successes_a
    posterior_beta_a = prior_beta + trials_a - successes_a
    posterior_alpha_b = prior_alpha + successes_b
    posterior_beta_b = prior_beta + trials_b - successes_b
    posterior_mean_a = posterior_alpha_a / (posterior_alpha_a + posterior_beta_a)
    posterior_mean_b = posterior_alpha_b / (posterior_alpha_b + posterior_beta_b)

    rng = np.random.default_rng(random_state)
    draws_a = rng.beta(posterior_alpha_a, posterior_beta_a, samples)
    draws_b = rng.beta(posterior_alpha_b, posterior_beta_b, samples)
    difference = draws_b - draws_a
    tail = (1 - credible_mass) / 2
    quantiles = [tail, 1 - tail]

    return {
        "rate_a": float(successes_a / trials_a),
        "rate_b": float(successes_b / trials_b),
        "posterior_mean_a": float(posterior_mean_a),
        "posterior_mean_b": float(posterior_mean_b),
        "posterior_interval_a": tuple(
            float(value)
            for value in stats.beta.ppf(
                quantiles, posterior_alpha_a, posterior_beta_a
            )
        ),
        "posterior_interval_b": tuple(
            float(value)
            for value in stats.beta.ppf(
                quantiles, posterior_alpha_b, posterior_beta_b
            )
        ),
        "probability_b_better": float(np.mean(difference > 0)),
        "probability_a_better": float(np.mean(difference < 0)),
        "expected_difference": float(posterior_mean_b - posterior_mean_a),
        "difference_credible_interval": tuple(
            float(value) for value in np.quantile(difference, quantiles)
        ),
        "credible_mass": float(credible_mass),
        "samples": samples,
        "posterior_parameters": {
            "a": (float(posterior_alpha_a), float(posterior_beta_a)),
            "b": (float(posterior_alpha_b), float(posterior_beta_b)),
        },
    }


__all__ = ["bayesian_proportion_ab_test"]
