"""
Statistical Utilities Module

Helper functions for statistical analysis, effect size calculations,
multiple testing corrections, confidence intervals, and bootstrapping.

Functions included:
- Effect size calculators (Cohen's d, h, g, Cramér's V, etc.)
- Multiple testing corrections (Bonferroni, Holm, FDR, etc.)
- Confidence interval calculators
- Bootstrap methods
- Power and sample size helpers
- Data transformation utilities
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, List, Callable, Dict, Any
from scipy import stats


# ==================== EFFECT SIZE CALCULATORS ====================

def cohens_d(
    group1: Union[np.ndarray, pd.Series, List],
    group2: Union[np.ndarray, pd.Series, List],
    paired: bool = False,
    correction: bool = False
) -> float:
    """
    Calculate Cohen's d effect size for comparing means.

    Cohen's d is a standardized measure of the difference between two means.

    Interpretation:
    - Small effect: d = 0.2
    - Medium effect: d = 0.5
    - Large effect: d = 0.8

    Parameters:
    -----------
    group1 : array-like
        First group of observations.
    group2 : array-like
        Second group of observations.
    paired : bool
        If True, calculates for paired samples.
    correction : bool
        If True, applies Hedges' g correction for small samples.

    Returns:
    --------
    float
        Cohen's d effect size.

    Examples:
    ---------
    >>> control = [23, 25, 27, 22, 24, 26, 28]
    >>> treatment = [28, 30, 32, 29, 31, 33, 30]
    >>> d = cohens_d(control, treatment)
    >>> print(f"Cohen's d: {d:.3f}")

    >>> # Paired samples
    >>> before = [120, 135, 128, 140, 132]
    >>> after = [115, 128, 122, 135, 125]
    >>> d = cohens_d(before, after, paired=True)
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    # Remove NaN
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    if paired:
        # For paired samples
        diff = group1 - group2
        d = np.mean(diff) / np.std(diff, ddof=1)
    else:
        # For independent samples
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = (mean1 - mean2) / pooled_std

    # Hedges' g correction for small samples
    if correction:
        n = len(group1) + len(group2) if not paired else len(group1)
        correction_factor = 1 - (3 / (4 * n - 9))
        d = d * correction_factor

    return d


def cohens_h(p1: float, p2: float) -> float:
    """
    Calculate Cohen's h effect size for comparing proportions.

    Interpretation:
    - Small effect: h = 0.2
    - Medium effect: h = 0.5
    - Large effect: h = 0.8

    Parameters:
    -----------
    p1 : float
        Proportion in group 1 (0 to 1).
    p2 : float
        Proportion in group 2 (0 to 1).

    Returns:
    --------
    float
        Cohen's h effect size.

    Examples:
    ---------
    >>> # Compare conversion rates: 10% vs 15%
    >>> h = cohens_h(0.10, 0.15)
    >>> print(f"Cohen's h: {h:.3f}")
    """
    # Arcsine transformation
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))

    return phi1 - phi2


def cramers_v(
    chi2: float,
    n: int,
    rows: int,
    cols: int
) -> float:
    """
    Calculate Cramér's V effect size for chi-square tests.

    Interpretation (for df = 1):
    - Small effect: V = 0.1
    - Medium effect: V = 0.3
    - Large effect: V = 0.5

    Parameters:
    -----------
    chi2 : float
        Chi-square statistic.
    n : int
        Total sample size.
    rows : int
        Number of rows in contingency table.
    cols : int
        Number of columns in contingency table.

    Returns:
    --------
    float
        Cramér's V effect size.

    Examples:
    ---------
    >>> # From chi-square test result
    >>> chi2_stat = 15.3
    >>> n_total = 200
    >>> v = cramers_v(chi2_stat, n_total, rows=2, cols=3)
    >>> print(f"Cramér's V: {v:.3f}")
    """
    min_dim = min(rows - 1, cols - 1)
    return np.sqrt(chi2 / (n * min_dim))


def glass_delta(
    group1: Union[np.ndarray, pd.Series, List],
    group2: Union[np.ndarray, pd.Series, List],
    control_group: int = 2
) -> float:
    """
    Calculate Glass's Δ (delta) effect size.

    Uses only the control group's standard deviation (more appropriate when
    groups have very different variances).

    Parameters:
    -----------
    group1 : array-like
        First group (typically treatment).
    group2 : array-like
        Second group (typically control).
    control_group : int
        Which group is control: 1 or 2 (default: 2).

    Returns:
    --------
    float
        Glass's delta effect size.

    Examples:
    ---------
    >>> treatment = [28, 30, 32, 29, 31, 33, 30]
    >>> control = [23, 25, 27, 22, 24, 26, 28]
    >>> delta = glass_delta(treatment, control, control_group=2)
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    mean1, mean2 = np.mean(group1), np.mean(group2)

    if control_group == 1:
        std_control = np.std(group1, ddof=1)
    else:
        std_control = np.std(group2, ddof=1)

    return (mean1 - mean2) / std_control


def eta_squared(
    ss_between: float,
    ss_total: float
) -> float:
    """
    Calculate eta-squared (η²) effect size for ANOVA.

    Represents the proportion of variance explained.

    Interpretation:
    - Small effect: η² = 0.01
    - Medium effect: η² = 0.06
    - Large effect: η² = 0.14

    Parameters:
    -----------
    ss_between : float
        Between-groups sum of squares.
    ss_total : float
        Total sum of squares.

    Returns:
    --------
    float
        Eta-squared effect size.

    Examples:
    ---------
    >>> ss_between = 450.5
    >>> ss_total = 1200.0
    >>> eta2 = eta_squared(ss_between, ss_total)
    >>> print(f"η² = {eta2:.3f} ({eta2*100:.1f}% variance explained)")
    """
    return ss_between / ss_total


def omega_squared(
    f_statistic: float,
    df_between: int,
    df_within: int,
    n_groups: int
) -> float:
    """
    Calculate omega-squared (ω²) effect size for ANOVA.

    Less biased estimate than eta-squared, especially for small samples.

    Parameters:
    -----------
    f_statistic : float
        F-statistic from ANOVA.
    df_between : int
        Degrees of freedom between groups.
    df_within : int
        Degrees of freedom within groups.
    n_groups : int
        Number of groups.

    Returns:
    --------
    float
        Omega-squared effect size.
    """
    ms_between = f_statistic * (df_within / df_between)
    return (df_between * (ms_between - 1)) / (df_between * ms_between + df_within + 1)


def interpret_effect_size(
    effect_size: float,
    effect_type: str = 'cohens_d'
) -> str:
    """
    Interpret effect size magnitude using Cohen's conventions.

    Parameters:
    -----------
    effect_size : float
        Calculated effect size.
    effect_type : str
        Type of effect size: 'cohens_d', 'cohens_h', 'cramers_v',
        'eta_squared', 'r' (correlation/rank-biserial).

    Returns:
    --------
    str
        Interpretation: 'negligible', 'small', 'medium', or 'large'.

    Examples:
    ---------
    >>> d = 0.65
    >>> print(interpret_effect_size(d, 'cohens_d'))  # 'medium'

    >>> v = 0.42
    >>> print(interpret_effect_size(v, 'cramers_v'))  # 'large'
    """
    abs_effect = abs(effect_size)

    if effect_type in ['cohens_d', 'cohens_h', 'glass_delta']:
        if abs_effect < 0.2:
            return 'negligible'
        elif abs_effect < 0.5:
            return 'small'
        elif abs_effect < 0.8:
            return 'medium'
        else:
            return 'large'

    elif effect_type == 'cramers_v':
        if abs_effect < 0.1:
            return 'negligible'
        elif abs_effect < 0.3:
            return 'small'
        elif abs_effect < 0.5:
            return 'medium'
        else:
            return 'large'

    elif effect_type in ['eta_squared', 'omega_squared']:
        if abs_effect < 0.01:
            return 'negligible'
        elif abs_effect < 0.06:
            return 'small'
        elif abs_effect < 0.14:
            return 'medium'
        else:
            return 'large'

    elif effect_type == 'r':  # Correlation or rank-biserial
        if abs_effect < 0.1:
            return 'negligible'
        elif abs_effect < 0.3:
            return 'small'
        elif abs_effect < 0.5:
            return 'medium'
        else:
            return 'large'

    else:
        # Generic interpretation
        if abs_effect < 0.2:
            return 'negligible'
        elif abs_effect < 0.5:
            return 'small'
        elif abs_effect < 0.8:
            return 'medium'
        else:
            return 'large'


# ==================== MULTIPLE TESTING CORRECTIONS ====================

def bonferroni_correction(
    p_values: Union[List[float], np.ndarray],
    alpha: float = 0.05
) -> Tuple[np.ndarray, float]:
    """
    Apply Bonferroni correction for multiple comparisons.

    Most conservative correction. Divides alpha by number of tests.

    Parameters:
    -----------
    p_values : array-like
        List of p-values from multiple tests.
    alpha : float
        Family-wise error rate.

    Returns:
    --------
    tuple
        (corrected_p_values, adjusted_alpha)

    Examples:
    ---------
    >>> p_values = [0.01, 0.04, 0.03, 0.08, 0.002]
    >>> corrected, alpha_adj = bonferroni_correction(p_values)
    >>> significant = corrected < alpha_adj
    >>> print(f"Adjusted alpha: {alpha_adj:.4f}")
    >>> print(f"Significant tests: {sum(significant)}")
    """
    p_values = np.asarray(p_values)
    n_tests = len(p_values)

    # Adjusted alpha
    alpha_adjusted = alpha / n_tests

    # Corrected p-values (multiply by number of tests, cap at 1.0)
    corrected_p = np.minimum(p_values * n_tests, 1.0)

    return corrected_p, alpha_adjusted


def holm_bonferroni_correction(
    p_values: Union[List[float], np.ndarray],
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Holm-Bonferroni (step-down) correction for multiple comparisons.

    Less conservative than Bonferroni, more powerful while controlling FWER.

    Parameters:
    -----------
    p_values : array-like
        List of p-values from multiple tests.
    alpha : float
        Family-wise error rate.

    Returns:
    --------
    tuple
        (reject_array, corrected_p_values)
        reject_array: Boolean array indicating which nulls to reject
        corrected_p_values: Adjusted p-values

    Examples:
    ---------
    >>> p_values = [0.01, 0.04, 0.03, 0.08, 0.002]
    >>> reject, corrected = holm_bonferroni_correction(p_values)
    >>> print(f"Reject null: {reject}")
    >>> print(f"Corrected p-values: {corrected}")
    """
    p_values = np.asarray(p_values)
    n_tests = len(p_values)

    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # Calculate adjusted p-values
    adjusted_p = np.zeros(n_tests)
    for i in range(n_tests):
        adjusted_p[i] = min((n_tests - i) * sorted_p[i], 1.0)

    # Ensure monotonicity (adjusted p-values should be non-decreasing)
    for i in range(1, n_tests):
        adjusted_p[i] = max(adjusted_p[i], adjusted_p[i-1])

    # Restore original order
    corrected_p = np.zeros(n_tests)
    corrected_p[sorted_indices] = adjusted_p

    # Determine which to reject
    reject = corrected_p < alpha

    return reject, corrected_p


def fdr_correction(
    p_values: Union[List[float], np.ndarray],
    alpha: float = 0.05,
    method: str = 'bh'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply False Discovery Rate (FDR) correction for multiple comparisons.

    Controls the expected proportion of false discoveries among rejections.
    Less conservative than FWER methods, good for exploratory analysis.

    Parameters:
    -----------
    p_values : array-like
        List of p-values from multiple tests.
    alpha : float
        False discovery rate.
    method : str
        'bh' for Benjamini-Hochberg (default) or 'by' for Benjamini-Yekutieli.

    Returns:
    --------
    tuple
        (reject_array, corrected_p_values)

    Examples:
    ---------
    >>> p_values = [0.01, 0.04, 0.03, 0.08, 0.002]
    >>> reject, corrected = fdr_correction(p_values, alpha=0.05)
    >>> print(f"Number of rejections: {sum(reject)}")
    """

    p_values = np.asarray(p_values)
    n_tests = len(p_values)

    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    if method == 'bh':
        # Benjamini-Hochberg procedure
        critical_values = (np.arange(1, n_tests + 1) / n_tests) * alpha

        # Find largest i where p(i) <= (i/m) * alpha
        comparisons = sorted_p <= critical_values
        if np.any(comparisons):
            max_idx = np.where(comparisons)[0][-1]
            reject_sorted = np.zeros(n_tests, dtype=bool)
            reject_sorted[:max_idx + 1] = True
        else:
            reject_sorted = np.zeros(n_tests, dtype=bool)

        # Adjusted p-values
        adjusted_p = np.minimum.accumulate(
            (sorted_p * n_tests / np.arange(1, n_tests + 1))[::-1]
        )[::-1]
        adjusted_p = np.minimum(adjusted_p, 1.0)

    elif method == 'by':
        # Benjamini-Yekutieli (more conservative, works under dependency)
        c_m = np.sum(1 / np.arange(1, n_tests + 1))
        critical_values = (np.arange(1, n_tests + 1) / (n_tests * c_m)) * alpha

        comparisons = sorted_p <= critical_values
        if np.any(comparisons):
            max_idx = np.where(comparisons)[0][-1]
            reject_sorted = np.zeros(n_tests, dtype=bool)
            reject_sorted[:max_idx + 1] = True
        else:
            reject_sorted = np.zeros(n_tests, dtype=bool)

        # Adjusted p-values
        adjusted_p = np.minimum.accumulate(
            (sorted_p * n_tests * c_m / np.arange(1, n_tests + 1))[::-1]
        )[::-1]
        adjusted_p = np.minimum(adjusted_p, 1.0)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Restore original order
    corrected_p = np.zeros(n_tests)
    corrected_p[sorted_indices] = adjusted_p

    reject = np.zeros(n_tests, dtype=bool)
    reject[sorted_indices] = reject_sorted

    return reject, corrected_p


def sidak_correction(
    p_values: Union[List[float], np.ndarray],
    alpha: float = 0.05
) -> Tuple[np.ndarray, float]:
    """
    Apply Šidák correction for multiple comparisons.

    Less conservative than Bonferroni, assumes independence of tests.

    Parameters:
    -----------
    p_values : array-like
        List of p-values from multiple tests.
    alpha : float
        Family-wise error rate.

    Returns:
    --------
    tuple
        (corrected_p_values, adjusted_alpha)

    Examples:
    ---------
    >>> p_values = [0.01, 0.04, 0.03, 0.08, 0.002]
    >>> corrected, alpha_adj = sidak_correction(p_values)
    """
    p_values = np.asarray(p_values)
    n_tests = len(p_values)

    # Adjusted alpha using Šidák formula
    alpha_adjusted = 1 - (1 - alpha) ** (1 / n_tests)

    # Corrected p-values
    corrected_p = 1 - (1 - p_values) ** n_tests
    corrected_p = np.minimum(corrected_p, 1.0)

    return corrected_p, alpha_adjusted


# ==================== CONFIDENCE INTERVALS ====================

def confidence_interval_mean(
    data: Union[np.ndarray, pd.Series, List],
    confidence: float = 0.95,
    method: str = 'parametric'
) -> Tuple[float, float]:
    """
    Calculate confidence interval for the mean.

    Parameters:
    -----------
    data : array-like
        Sample data.
    confidence : float
        Confidence level (e.g., 0.95 for 95% CI).
    method : str
        'parametric' (t-distribution) or 'bootstrap'.

    Returns:
    --------
    tuple
        (lower_bound, upper_bound)

    Examples:
    ---------
    >>> data = [23, 25, 27, 22, 24, 26, 28, 25, 23, 26]
    >>> ci = confidence_interval_mean(data, confidence=0.95)
    >>> print(f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")

    >>> # Bootstrap CI (non-parametric)
    >>> ci_boot = confidence_interval_mean(data, method='bootstrap')
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    if method == 'parametric':
        # Parametric CI using t-distribution
        mean = np.mean(data)
        se = stats.sem(data)
        ci = stats.t.interval(confidence, len(data) - 1, loc=mean, scale=se)
        return ci

    elif method == 'bootstrap':
        # Bootstrap CI
        n_bootstrap = 10000
        bootstrap_means = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

        return (lower, upper)

    else:
        raise ValueError(f"Unknown method: {method}")


def confidence_interval_proportion(
    successes: int,
    n: int,
    confidence: float = 0.95,
    method: str = 'wilson'
) -> Tuple[float, float]:
    """
    Calculate confidence interval for a proportion.

    Parameters:
    -----------
    successes : int
        Number of successes.
    n : int
        Total number of trials.
    confidence : float
        Confidence level.
    method : str
        'wilson' (default, recommended), 'normal', or 'agresti_coull'.

    Returns:
    --------
    tuple
        (lower_bound, upper_bound)

    Examples:
    ---------
    >>> # 48 conversions out of 500 visitors
    >>> ci = confidence_interval_proportion(48, 500, confidence=0.95)
    >>> print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    >>> print(f"95% CI: [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")
    """
    p = successes / n
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)

    if method == 'wilson':
        # Wilson score interval (more accurate, especially for small samples)
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt((p * (1 - p) / n + z**2 / (4 * n**2))) / denominator

        lower = center - margin
        upper = center + margin

    elif method == 'normal':
        # Normal approximation (Wald interval)
        se = np.sqrt(p * (1 - p) / n)
        margin = z * se

        lower = p - margin
        upper = p + margin

    elif method == 'agresti_coull':
        # Agresti-Coull interval
        n_tilde = n + z**2
        p_tilde = (successes + z**2 / 2) / n_tilde
        se_tilde = np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
        margin = z * se_tilde

        lower = p_tilde - margin
        upper = p_tilde + margin

    else:
        raise ValueError(f"Unknown method: {method}")

    # Ensure bounds are within [0, 1]
    lower = max(0, lower)
    upper = min(1, upper)

    return (lower, upper)


def confidence_interval_difference(
    group1: Union[np.ndarray, pd.Series, List],
    group2: Union[np.ndarray, pd.Series, List],
    confidence: float = 0.95,
    paired: bool = False,
    equal_var: bool = True
) -> Tuple[float, float]:
    """
    Calculate confidence interval for the difference between two means.

    Parameters:
    -----------
    group1 : array-like
        First group.
    group2 : array-like
        Second group.
    confidence : float
        Confidence level.
    paired : bool
        Whether samples are paired.
    equal_var : bool
        Assume equal variances (ignored if paired=True).

    Returns:
    --------
    tuple
        (lower_bound, upper_bound)

    Examples:
    ---------
    >>> control = [23, 25, 27, 22, 24, 26, 28]
    >>> treatment = [28, 30, 32, 29, 31, 33, 30]
    >>> ci = confidence_interval_difference(control, treatment)
    >>> print(f"95% CI for difference: [{ci[0]:.2f}, {ci[1]:.2f}]")
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    alpha = 1 - confidence

    if paired:
        # Paired samples
        diff = group1 - group2
        mean_diff = np.mean(diff)
        se = stats.sem(diff)
        df = len(diff) - 1

        margin = stats.t.ppf(1 - alpha / 2, df) * se

        lower = mean_diff - margin
        upper = mean_diff + margin

    else:
        # Independent samples
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        mean_diff = mean1 - mean2

        if equal_var:
            # Pooled variance
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            se = np.sqrt(pooled_var * (1/n1 + 1/n2))
            df = n1 + n2 - 2
        else:
            # Welch's t-test (unequal variances)
            se = np.sqrt(var1/n1 + var2/n2)
            # Welch-Satterthwaite degrees of freedom
            df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2 / (n1-1) + (var2/n2)**2 / (n2-1))

        margin = stats.t.ppf(1 - alpha / 2, df) * se

        lower = mean_diff - margin
        upper = mean_diff + margin

    return (lower, upper)


# ==================== BOOTSTRAP METHODS ====================

def bootstrap(
    data: Union[np.ndarray, pd.Series, List],
    statistic: Callable,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform bootstrap resampling to estimate sampling distribution.

    Parameters:
    -----------
    data : array-like
        Original sample data.
    statistic : callable
        Function to compute statistic (e.g., np.mean, np.median).
    n_bootstrap : int
        Number of bootstrap samples.
    confidence : float
        Confidence level for CI.
    random_state : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    dict
        Dictionary containing:
        - 'statistic': Original statistic value
        - 'bootstrap_distribution': Array of bootstrap statistics
        - 'mean': Mean of bootstrap distribution
        - 'std': Standard error (std of bootstrap distribution)
        - 'ci': Confidence interval (lower, upper)

    Examples:
    ---------
    >>> data = [23, 25, 27, 22, 24, 26, 28, 25, 23, 26]
    >>> result = bootstrap(data, statistic=np.mean, n_bootstrap=10000)
    >>> print(f"Mean: {result['statistic']:.2f}")
    >>> lower, upper = result['ci']
    >>> print(f"95% CI: [{lower:.2f}, {upper:.2f}]")
    >>> print(f"Standard Error: {result['std']:.2f}")

    >>> # Bootstrap for median
    >>> result_median = bootstrap(data, statistic=np.median)

    >>> # Custom statistic
    >>> result_iqr = bootstrap(data, statistic=lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    """
    if random_state is not None:
        np.random.seed(random_state)

    data = np.asarray(data)
    data = data[~np.isnan(data)]

    # Original statistic
    original_stat = statistic(data)

    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Calculate confidence interval
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return {
        'statistic': original_stat,
        'bootstrap_distribution': bootstrap_stats,
        'mean': np.mean(bootstrap_stats),
        'std': np.std(bootstrap_stats),
        'ci': (ci_lower, ci_upper),
        'n_bootstrap': n_bootstrap
    }


def bootstrap_test(
    group1: Union[np.ndarray, pd.Series, List],
    group2: Union[np.ndarray, pd.Series, List],
    statistic: Callable = lambda x, y: np.mean(x) - np.mean(y),
    n_bootstrap: int = 10000,
    alternative: str = 'two-sided',
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform permutation/bootstrap hypothesis test.

    Non-parametric alternative to traditional tests.

    Parameters:
    -----------
    group1 : array-like
        First group.
    group2 : array-like
        Second group.
    statistic : callable
        Function to compute test statistic (default: difference in means).
    n_bootstrap : int
        Number of permutations.
    alternative : str
        'two-sided', 'greater', or 'less'.
    random_state : int, optional
        Random seed.

    Returns:
    --------
    dict
        Test results including p-value and bootstrap distribution.

    Examples:
    ---------
    >>> control = [23, 25, 27, 22, 24, 26, 28]
    >>> treatment = [28, 30, 32, 29, 31, 33, 30]
    >>> result = bootstrap_test(control, treatment, n_bootstrap=10000)
    >>> print(f"P-value: {result['p_value']:.4f}")
    >>> print(f"Observed difference: {result['observed_statistic']:.2f}")
    """
    if random_state is not None:
        np.random.seed(random_state)

    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    # Observed test statistic
    observed_stat = statistic(group1, group2)

    # Combine data for permutation
    combined = np.concatenate([group1, group2])
    n1 = len(group1)

    # Permutation test
    permuted_stats = []
    for _ in range(n_bootstrap):
        # Shuffle combined data
        np.random.shuffle(combined)

        # Split into two groups
        perm_group1 = combined[:n1]
        perm_group2 = combined[n1:]

        # Calculate statistic
        perm_stat = statistic(perm_group1, perm_group2)
        permuted_stats.append(perm_stat)

    permuted_stats = np.array(permuted_stats)

    # Calculate p-value
    if alternative == 'two-sided':
        p_value = np.mean(np.abs(permuted_stats) >= np.abs(observed_stat))
    elif alternative == 'greater':
        p_value = np.mean(permuted_stats >= observed_stat)
    elif alternative == 'less':
        p_value = np.mean(permuted_stats <= observed_stat)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return {
        'observed_statistic': observed_stat,
        'p_value': p_value,
        'permuted_distribution': permuted_stats,
        'n_permutations': n_bootstrap,
        'alternative': alternative
    }


# ==================== DATA TRANSFORMATION ====================

def standardize(
    data: Union[np.ndarray, pd.Series, List],
    method: str = 'zscore'
) -> np.ndarray:
    """
    Standardize data using various methods.

    Parameters:
    -----------
    data : array-like
        Data to standardize.
    method : str
        'zscore' (default), 'minmax', 'robust', or 'rank'.

    Returns:
    --------
    np.ndarray
        Standardized data.

    Examples:
    ---------
    >>> data = [10, 20, 30, 40, 50]
    >>> z_scores = standardize(data, method='zscore')
    >>> minmax_scaled = standardize(data, method='minmax')
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    if method == 'zscore':
        # Z-score standardization (mean=0, std=1)
        return (data - np.mean(data)) / np.std(data, ddof=1)

    elif method == 'minmax':
        # Min-max scaling to [0, 1]
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)

    elif method == 'robust':
        # Robust scaling using median and IQR
        median = np.median(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        return (data - median) / iqr

    elif method == 'rank':
        # Rank transformation
        return stats.rankdata(data)

    else:
        raise ValueError(f"Unknown method: {method}")


def winsorize(
    data: Union[np.ndarray, pd.Series, List],
    limits: Tuple[float, float] = (0.05, 0.05)
) -> np.ndarray:
    """
    Winsorize data by capping extreme values.

    Replaces extreme values with percentile limits.
    Useful for reducing impact of outliers without removing data points.

    Parameters:
    -----------
    data : array-like
        Data to winsorize.
    limits : tuple
        (lower_percentile, upper_percentile) as proportions.
        E.g., (0.05, 0.05) caps at 5th and 95th percentiles.

    Returns:
    --------
    np.ndarray
        Winsorized data.

    Examples:
    ---------
    >>> data = [1, 2, 3, 4, 5, 100]  # 100 is an outlier
    >>> winsorized = winsorize(data, limits=(0.1, 0.1))
    >>> print(winsorized)  # 100 will be capped
    """
    from scipy.stats.mstats import winsorize as scipy_winsorize

    data = np.asarray(data)
    return scipy_winsorize(data, limits=limits)


# ==================== UTILITY FUNCTIONS ====================

def relative_lift(
    control_value: float,
    treatment_value: float,
    as_percentage: bool = True
) -> float:
    """
    Calculate relative lift (percentage change).

    Parameters:
    -----------
    control_value : float
        Baseline value.
    treatment_value : float
        New value.
    as_percentage : bool
        Return as percentage (True) or proportion (False).

    Returns:
    --------
    float
        Relative lift.

    Examples:
    ---------
    >>> control_rate = 0.10  # 10% conversion
    >>> treatment_rate = 0.12  # 12% conversion
    >>> lift = relative_lift(control_rate, treatment_rate)
    >>> print(f"Lift: {lift:.1f}%")  # +20.0%
    """
    if control_value == 0:
        return np.inf if treatment_value > 0 else 0

    lift = (treatment_value - control_value) / control_value

    if as_percentage:
        return lift * 100
    return lift


def absolute_lift(
    control_value: float,
    treatment_value: float
) -> float:
    """
    Calculate absolute lift (absolute difference).

    Parameters:
    -----------
    control_value : float
        Baseline value.
    treatment_value : float
        New value.

    Returns:
    --------
    float
        Absolute lift.

    Examples:
    ---------
    >>> control_rate = 0.10
    >>> treatment_rate = 0.12
    >>> lift = absolute_lift(control_rate, treatment_rate)
    >>> print(f"Absolute lift: {lift:.3f}")  # +0.020 (2 percentage points)
    """
    return treatment_value - control_value


def pooled_std(
    group1: Union[np.ndarray, pd.Series, List],
    group2: Union[np.ndarray, pd.Series, List]
) -> float:
    """
    Calculate pooled standard deviation for two groups.

    Used in Cohen's d calculation and t-tests.

    Parameters:
    -----------
    group1 : array-like
        First group.
    group2 : array-like
        Second group.

    Returns:
    --------
    float
        Pooled standard deviation.
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    return np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
