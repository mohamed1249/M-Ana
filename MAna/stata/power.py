"""Power and sample-size helpers for two independent sample means."""

from __future__ import annotations

import numpy as np
from scipy import stats


def _validate_probability(name: str, value: float) -> None:
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be between 0 and 1.")


def _validate_sample_size(sample_size_per_group: int) -> int:
    if isinstance(sample_size_per_group, bool) or not isinstance(
        sample_size_per_group, (int, np.integer)
    ):
        raise TypeError("sample_size_per_group must be an integer.")
    sample_size_per_group = int(sample_size_per_group)
    if sample_size_per_group < 2:
        raise ValueError("sample_size_per_group must be at least 2.")
    return sample_size_per_group


def two_sample_ttest_power(
    effect_size: float,
    sample_size_per_group: int,
    *,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """Calculate exact power for an equal-size independent two-sample t-test.

    ``effect_size`` is Cohen's d for group A minus group B. Its sign therefore
    matters for one-sided alternatives.
    """
    _validate_probability("alpha", alpha)
    sample_size_per_group = _validate_sample_size(sample_size_per_group)
    if not np.isfinite(effect_size):
        raise ValueError("effect_size must be finite.")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'.")

    degrees_of_freedom = 2 * sample_size_per_group - 2
    noncentrality = float(effect_size) * np.sqrt(sample_size_per_group / 2)

    if alternative == "two-sided":
        critical = stats.t.ppf(1 - alpha / 2, degrees_of_freedom)
        power = stats.nct.cdf(
            -critical, degrees_of_freedom, noncentrality
        ) + stats.nct.sf(critical, degrees_of_freedom, noncentrality)
    elif alternative == "greater":
        critical = stats.t.ppf(1 - alpha, degrees_of_freedom)
        power = stats.nct.sf(critical, degrees_of_freedom, noncentrality)
    else:
        critical = stats.t.ppf(alpha, degrees_of_freedom)
        power = stats.nct.cdf(critical, degrees_of_freedom, noncentrality)

    return float(np.clip(power, 0.0, 1.0))


def required_sample_size_two_sample(
    effect_size: float,
    *,
    power: float = 0.80,
    alpha: float = 0.05,
    two_sided: bool = True,
    max_sample_size: int = 1_000_000,
) -> int:
    """Find the smallest observations per group reaching the target power.

    The calculation uses the noncentral t distribution. For a one-sided test,
    the sign of ``effect_size`` selects ``greater`` or ``less``.
    """
    _validate_probability("power", power)
    _validate_probability("alpha", alpha)
    if not np.isfinite(effect_size) or effect_size == 0:
        raise ValueError("effect_size must be finite and non-zero.")
    if not isinstance(two_sided, bool):
        raise TypeError("two_sided must be a boolean.")
    max_sample_size = _validate_sample_size(max_sample_size)

    alternative = "two-sided"
    if not two_sided:
        alternative = "greater" if effect_size > 0 else "less"

    def achieved(sample_size: int) -> float:
        return two_sample_ttest_power(
            effect_size,
            sample_size,
            alpha=alpha,
            alternative=alternative,
        )

    lower, upper = 2, 2
    if achieved(lower) >= power:
        return lower

    while upper < max_sample_size and achieved(upper) < power:
        upper = min(upper * 2, max_sample_size)
    if achieved(upper) < power:
        raise ValueError(
            "Target power was not reached before max_sample_size; increase the "
            "limit or use a larger effect size."
        )

    while lower + 1 < upper:
        midpoint = (lower + upper) // 2
        if achieved(midpoint) >= power:
            upper = midpoint
        else:
            lower = midpoint
    return upper


__all__ = ["required_sample_size_two_sample", "two_sample_ttest_power"]
