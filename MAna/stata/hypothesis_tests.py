"""
Statistical Hypothesis Testing Module

Provides comprehensive statistical tests for data analysis and A/B testing.
All tests return detailed results with effect sizes, confidence intervals,
and interpretations.

Tests included:
- T-tests (independent, paired, one-sample)
- Z-tests
- Proportion tests
- Chi-square tests
- Non-parametric tests (Mann-Whitney, Wilcoxon, Kruskal-Wallis)
- Variance tests (Levene, Bartlett)
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field
from scipy import stats
import sys


def _terminal_safe(text: str) -> str:
    """Return text that can be printed by the active terminal encoding."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


@dataclass
class TestResult:
    """
    Standardized result object for all statistical tests.

    Attributes:
    -----------
    test_name : str
        Name of the statistical test performed.
    statistic : float
        Test statistic value.
    p_value : float
        P-value of the test.
    significant : bool
        Whether result is statistically significant at alpha level.
    alpha : float
        Significance level used.
    effect_size : float, optional
        Measure of effect size (Cohen's d, Cramér's V, etc.).
    confidence_interval : tuple, optional
        Confidence interval for the difference/effect.
    sample_sizes : dict
        Sample sizes for each group.
    descriptive_stats : dict
        Descriptive statistics for each group.
    interpretation : str
        Human-readable interpretation of results.
    recommendation : str
        Actionable recommendation based on results.
    """

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    effect_size: Optional[float] = None
    effect_size_name: Optional[str] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    sample_sizes: Dict[str, int] = field(default_factory=dict)
    descriptive_stats: Dict[str, Any] = field(default_factory=dict)
    interpretation: str = ""
    recommendation: str = ""
    power: Optional[float] = None

    def __str__(self) -> str:
        """Pretty print results."""
        output = f"""
╔══════════════════════════════════════════════════════════════╗
║  {self.test_name.upper():^60}  ║
╚══════════════════════════════════════════════════════════════╝

📊 TEST STATISTICS
  Statistic:        {self.statistic:.4f}
  P-value:          {self.p_value:.6f} {'***' if self.p_value < 0.001 else '**' if self.p_value < 0.01 else '*' if self.p_value < 0.05 else ''}
  Significance:     {'✓ Yes' if self.significant else '✗ No'} (α = {self.alpha})
"""

        if self.effect_size is not None:
            output += f"  Effect Size:      {self.effect_size:.4f} ({self.effect_size_name})\\n"

        if self.confidence_interval:
            output += f"  95% CI:           [{self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f}]\\n"

        if self.power:
            output += f"  Statistical Power: {self.power:.2%}\\n"

        if self.sample_sizes:
            output += "\\n📈 SAMPLE SIZES\\n"
            for group, size in self.sample_sizes.items():
                output += f"  {group}: {size:,}\\n"

        if self.descriptive_stats:
            output += "\\n📋 DESCRIPTIVE STATISTICS\\n"
            for group, stats_dict in self.descriptive_stats.items():
                output += f"  {group}:\\n"
                for stat_name, value in stats_dict.items():
                    if isinstance(value, (int, float)):
                        output += f"    {stat_name}: {value:.4f}\\n"
                    else:
                        output += f"    {stat_name}: {value}\\n"

        if self.interpretation:
            output += f"\\n💡 INTERPRETATION\\n  {self.interpretation}\\n"

        if self.recommendation:
            output += f"\\n🎯 RECOMMENDATION\\n  {self.recommendation}\\n"

        return _terminal_safe(output)

    def to_dict(self) -> Dict:
        """Convert result to dictionary."""
        return {
            'test_name': self.test_name,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'significant': self.significant,
            'alpha': self.alpha,
            'effect_size': self.effect_size,
            'effect_size_name': self.effect_size_name,
            'confidence_interval': self.confidence_interval,
            'sample_sizes': self.sample_sizes,
            'descriptive_stats': self.descriptive_stats,
            'interpretation': self.interpretation,
            'recommendation': self.recommendation,
            'power': self.power
        }


# ==================== T-TESTS ====================

def t_test(
    group1: Union[np.ndarray, pd.Series, List],
    group2: Optional[Union[np.ndarray, pd.Series, List]] = None,
    test_type: str = 'independent',
    alternative: str = 'two-sided',
    mu: float = 0,
    alpha: float = 0.05,
    equal_var: bool = True,
    paired: bool = False
) -> TestResult:
    """
    Comprehensive t-test for comparing means.

    Supports:
    - Independent samples t-test (default)
    - Paired samples t-test
    - One-sample t-test
    - Welch's t-test (unequal variances)

    Parameters:
    -----------
    group1 : array-like
        First group of observations (or only group for one-sample test).
    group2 : array-like, optional
        Second group of observations. Not needed for one-sample test.
    test_type : str
        Type of test: 'independent', 'paired', or 'one-sample'.
    alternative : str
        Alternative hypothesis: 'two-sided', 'greater', or 'less'.
    mu : float
        Hypothesized mean difference (for one-sample or paired tests).
    alpha : float
        Significance level (default: 0.05).
    equal_var : bool
        Assume equal variances (True) or use Welch's t-test (False).
    paired : bool
        If True, performs paired t-test (overrides test_type).

    Returns:
    --------
    TestResult
        Comprehensive test results with interpretation.

    Examples:
    ---------
    >>> # Independent samples t-test
    >>> control = [23, 25, 27, 22, 24, 26, 28]
    >>> treatment = [28, 30, 32, 29, 31, 33, 30]
    >>> result = t_test(control, treatment)
    >>> print(result)

    >>> # Paired t-test (before/after)
    >>> before = [120, 135, 128, 140, 132]
    >>> after = [115, 128, 122, 135, 125]
    >>> result = t_test(before, after, test_type='paired')

    >>> # One-sample t-test
    >>> sample = [98.6, 98.4, 98.8, 98.5, 98.7]
    >>> result = t_test(sample, test_type='one-sample', mu=98.6)

    >>> # Welch's t-test (unequal variances)
    >>> result = t_test(control, treatment, equal_var=False)
    """
    # Convert to numpy arrays
    group1 = np.asarray(group1)
    if group2 is not None:
        group2 = np.asarray(group2)

    # Remove NaN values
    group1 = group1[~np.isnan(group1)]
    if group2 is not None:
        group2 = group2[~np.isnan(group2)]

    # Determine test type
    if paired or test_type == 'paired':
        if group2 is None:
            raise ValueError("Paired t-test requires two groups")
        test_name = "Paired Samples T-Test"
        statistic, p_value = stats.ttest_rel(group1, group2, alternative=alternative)

        # Descriptive statistics
        diff = group1 - group2
        desc_stats = {
            'Group 1': {
                'mean': np.mean(group1),
                'std': np.std(group1, ddof=1),
                'n': len(group1)
            },
            'Group 2': {
                'mean': np.mean(group2),
                'std': np.std(group2, ddof=1),
                'n': len(group2)
            },
            'Difference': {
                'mean': np.mean(diff),
                'std': np.std(diff, ddof=1)
            }
        }

        # Effect size (Cohen's d for paired samples)
        effect_size = np.mean(diff) / np.std(diff, ddof=1)

        # Confidence interval for mean difference
        se = stats.sem(diff)
        ci = stats.t.interval(1 - alpha, len(diff) - 1, loc=np.mean(diff), scale=se)

        sample_sizes = {'paired_samples': len(group1)}

    elif test_type == 'one-sample' or group2 is None:
        test_name = "One-Sample T-Test"
        statistic, p_value = stats.ttest_1samp(group1, mu, alternative=alternative)

        desc_stats = {
            'Sample': {
                'mean': np.mean(group1),
                'std': np.std(group1, ddof=1),
                'n': len(group1)
            },
            'Hypothesized Mean': mu
        }

        # Effect size (Cohen's d)
        effect_size = (np.mean(group1) - mu) / np.std(group1, ddof=1)

        # Confidence interval
        se = stats.sem(group1)
        ci = stats.t.interval(1 - alpha, len(group1) - 1, loc=np.mean(group1), scale=se)

        sample_sizes = {'sample': len(group1)}

    else:  # Independent samples
        if equal_var:
            test_name = "Independent Samples T-Test (Equal Variances)"
        else:
            test_name = "Welch's T-Test (Unequal Variances)"

        statistic, p_value = stats.ttest_ind(
            group1, group2,
            equal_var=equal_var,
            alternative=alternative
        )

        desc_stats = {
            'Group 1': {
                'mean': np.mean(group1),
                'std': np.std(group1, ddof=1),
                'n': len(group1)
            },
            'Group 2': {
                'mean': np.mean(group2),
                'std': np.std(group2, ddof=1),
                'n': len(group2)
            },
            'Difference': {
                'mean': np.mean(group1) - np.mean(group2)
            }
        }

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(group1) - 1) * np.var(group1, ddof=1) +
             (len(group2) - 1) * np.var(group2, ddof=1)) /
            (len(group1) + len(group2) - 2)
        )
        effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std

        # Confidence interval for difference
        se = np.sqrt(np.var(group1, ddof=1) / len(group1) +
                    np.var(group2, ddof=1) / len(group2))
        df = len(group1) + len(group2) - 2
        mean_diff = np.mean(group1) - np.mean(group2)
        ci = stats.t.interval(1 - alpha, df, loc=mean_diff, scale=se)

        sample_sizes = {'group_1': len(group1), 'group_2': len(group2)}

    # Determine significance
    significant = p_value < alpha

    # Interpretation
    if test_type == 'one-sample' or group2 is None:
        interpretation = (
            f"The sample mean ({desc_stats['Sample']['mean']:.4f}) is "
            f"{'significantly' if significant else 'not significantly'} different from "
            f"the hypothesized mean ({mu}) at the {alpha} significance level."
        )
    elif paired or test_type == 'paired':
        interpretation = (
            f"The mean difference between paired observations ({desc_stats['Difference']['mean']:.4f}) is "
            f"{'significantly' if significant else 'not significantly'} different from zero "
            f"at the {alpha} significance level."
        )
    else:
        interpretation = (
            f"Group 1 (mean = {desc_stats['Group 1']['mean']:.4f}) and "
            f"Group 2 (mean = {desc_stats['Group 2']['mean']:.4f}) have "
            f"{'significantly' if significant else 'not significantly'} different means "
            f"at the {alpha} significance level."
        )

    # Effect size interpretation
    abs_effect = abs(effect_size)
    if abs_effect < 0.2:
        effect_desc = "negligible"
    elif abs_effect < 0.5:
        effect_desc = "small"
    elif abs_effect < 0.8:
        effect_desc = "medium"
    else:
        effect_desc = "large"

    interpretation += f" The effect size (Cohen's d = {effect_size:.4f}) is {effect_desc}."

    # Recommendation
    if significant:
        recommendation = (
            f"Reject the null hypothesis. There is sufficient evidence of a difference. "
            f"The effect size suggests a {effect_desc} practical significance."
        )
    else:
        recommendation = (
            "Fail to reject the null hypothesis. There is insufficient evidence of a difference. "
            "Consider increasing sample size if a difference is expected."
        )

    return TestResult(
        test_name=test_name,
        statistic=statistic,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        effect_size=effect_size,
        effect_size_name="Cohen's d",
        confidence_interval=ci,
        sample_sizes=sample_sizes,
        descriptive_stats=desc_stats,
        interpretation=interpretation,
        recommendation=recommendation
    )


# ==================== PROPORTION TESTS ====================

def proportion_test(
    successes1: int,
    n1: int,
    successes2: Optional[int] = None,
    n2: Optional[int] = None,
    p0: float = 0.5,
    alternative: str = 'two-sided',
    alpha: float = 0.05,
    method: str = 'normal'
) -> TestResult:
    """
    Test for equality of proportions.

    Supports:
    - One-sample proportion test (binomial test)
    - Two-sample proportion test (z-test for proportions)

    Parameters:
    -----------
    successes1 : int
        Number of successes in group 1.
    n1 : int
        Total number of trials in group 1.
    successes2 : int, optional
        Number of successes in group 2 (for two-sample test).
    n2 : int, optional
        Total number of trials in group 2 (for two-sample test).
    p0 : float
        Hypothesized proportion (for one-sample test).
    alternative : str
        Alternative hypothesis: 'two-sided', 'greater', or 'less'.
    alpha : float
        Significance level.
    method : str
        Method: 'normal' (z-test) or 'exact' (binomial test for one-sample).

    Returns:
    --------
    TestResult
        Comprehensive test results.

    Examples:
    ---------
    >>> # One-sample proportion test
    >>> # Test if conversion rate is different from 50%
    >>> result = proportion_test(successes1=520, n1=1000, p0=0.5)

    >>> # Two-sample proportion test
    >>> # Compare conversion rates between control and treatment
    >>> result = proportion_test(
    ...     successes1=48, n1=500,  # Control: 9.6%
    ...     successes2=65, n2=500   # Treatment: 13%
    ... )

    >>> # One-sided test (is treatment better?)
    >>> result = proportion_test(
    ...     successes1=48, n1=500,
    ...     successes2=65, n2=500,
    ...     alternative='less'  # Group 1 < Group 2
    ... )
    """

    if successes2 is None or n2 is None:
        # One-sample proportion test
        p1 = successes1 / n1

        if method == 'exact':
            # Exact binomial test
            test_name = "Exact Binomial Test"
            p_value = stats.binomtest(successes1, n1, p0, alternative=alternative).pvalue

            # Approximation for statistic
            statistic = (p1 - p0) / np.sqrt(p0 * (1 - p0) / n1)
        else:
            # Normal approximation (z-test)
            test_name = "One-Sample Proportion Z-Test"
            statistic = (p1 - p0) / np.sqrt(p0 * (1 - p0) / n1)

            if alternative == 'two-sided':
                p_value = 2 * (1 - stats.norm.cdf(abs(statistic)))
            elif alternative == 'greater':
                p_value = 1 - stats.norm.cdf(statistic)
            else:  # less
                p_value = stats.norm.cdf(statistic)

        # Confidence interval for proportion
        se = np.sqrt(p1 * (1 - p1) / n1)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci = (p1 - z_crit * se, p1 + z_crit * se)

        # Effect size (h - Cohen's h for proportions)
        effect_size = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p0)))

        desc_stats = {
            'Sample': {
                'successes': successes1,
                'total': n1,
                'proportion': p1,
                'percentage': p1 * 100
            },
            'Hypothesized': {
                'proportion': p0,
                'percentage': p0 * 100
            }
        }

        sample_sizes = {'sample': n1}

        interpretation = (
            f"The observed proportion ({p1:.4f} or {p1*100:.2f}%) is "
            f"{'significantly' if p_value < alpha else 'not significantly'} different from "
            f"the hypothesized proportion ({p0:.4f} or {p0*100:.2f}%) at the {alpha} significance level."
        )

    else:
        # Two-sample proportion test
        test_name = "Two-Sample Proportion Z-Test"

        p1 = successes1 / n1
        p2 = successes2 / n2
        p_pooled = (successes1 + successes2) / (n1 + n2)

        # Z-test statistic
        se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        statistic = (p1 - p2) / se_pooled

        # P-value
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.norm.cdf(abs(statistic)))
        elif alternative == 'greater':
            p_value = 1 - stats.norm.cdf(statistic)
        else:  # less
            p_value = stats.norm.cdf(statistic)

        # Confidence interval for difference
        se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        diff = p1 - p2
        ci = (diff - z_crit * se_diff, diff + z_crit * se_diff)

        # Effect size (Cohen's h)
        effect_size = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

        # Relative lift
        relative_lift = ((p2 - p1) / p1) * 100 if p1 > 0 else np.inf

        desc_stats = {
            'Group 1': {
                'successes': successes1,
                'total': n1,
                'proportion': p1,
                'percentage': p1 * 100
            },
            'Group 2': {
                'successes': successes2,
                'total': n2,
                'proportion': p2,
                'percentage': p2 * 100
            },
            'Difference': {
                'absolute': p2 - p1,
                'relative_lift_%': relative_lift
            }
        }

        sample_sizes = {'group_1': n1, 'group_2': n2}

        interpretation = (
            f"Group 1 ({p1:.4f} or {p1*100:.2f}%) and Group 2 ({p2:.4f} or {p2*100:.2f}%) have "
            f"{'significantly' if p_value < alpha else 'not significantly'} different proportions "
            f"at the {alpha} significance level. "
            f"The relative lift is {relative_lift:+.2f}%."
        )

    significant = p_value < alpha

    # Effect size interpretation
    abs_effect = abs(effect_size)
    if abs_effect < 0.2:
        effect_desc = "negligible"
    elif abs_effect < 0.5:
        effect_desc = "small"
    elif abs_effect < 0.8:
        effect_desc = "medium"
    else:
        effect_desc = "large"

    interpretation += f" The effect size (Cohen's h = {effect_size:.4f}) is {effect_desc}."

    # Recommendation
    if significant:
        if successes2 is not None:
            winner = "Group 2" if p2 > p1 else "Group 1"
            recommendation = (
                f"Reject the null hypothesis. {winner} has a significantly higher proportion. "
                f"The effect size suggests {effect_desc} practical significance."
            )
        else:
            recommendation = (
                f"Reject the null hypothesis. The proportion is significantly different from {p0}. "
                f"The effect size suggests {effect_desc} practical significance."
            )
    else:
        recommendation = (
            "Fail to reject the null hypothesis. There is insufficient evidence of a difference. "
            "Consider increasing sample size if a difference is expected."
        )

    return TestResult(
        test_name=test_name,
        statistic=statistic,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        effect_size=effect_size,
        effect_size_name="Cohen's h",
        confidence_interval=ci,
        sample_sizes=sample_sizes,
        descriptive_stats=desc_stats,
        interpretation=interpretation,
        recommendation=recommendation
    )


# ==================== CHI-SQUARE TESTS ====================

def chi_square_test(
    observed: Union[np.ndarray, pd.DataFrame, pd.Series],
    expected: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
    test_type: str = 'independence',
    alpha: float = 0.05
) -> TestResult:
    """
    Chi-square test for independence or goodness of fit.

    Parameters:
    -----------
    observed : array-like or DataFrame
        Observed frequencies. For independence test, provide contingency table.
    expected : array-like, optional
        Expected frequencies (for goodness of fit test).
    test_type : str
        Type: 'independence' (default) or 'goodness_of_fit'.
    alpha : float
        Significance level.

    Returns:
    --------
    TestResult
        Comprehensive test results.

    Examples:
    ---------
    >>> # Test of independence (2x2 contingency table)
    >>> # Rows: Gender (M/F), Columns: Product Preference (A/B)
    >>> table = np.array([[30, 20], [25, 35]])
    >>> result = chi_square_test(table)

    >>> # Goodness of fit test
    >>> # Test if dice is fair
    >>> observed = [12, 15, 10, 18, 14, 11]  # Observed rolls
    >>> expected = [13.33] * 6  # Expected for fair dice
    >>> result = chi_square_test(observed, expected, test_type='goodness_of_fit')

    >>> # Using pandas DataFrame
    >>> df = pd.DataFrame({
    ...     'Product_A': [30, 25],
    ...     'Product_B': [20, 35]
    ... }, index=['Male', 'Female'])
    >>> result = chi_square_test(df)
    """

    # Convert to numpy array
    if isinstance(observed, (pd.DataFrame, pd.Series)):
        observed_array = observed.values
    else:
        observed_array = np.asarray(observed)

    if test_type == 'independence':
        test_name = "Chi-Square Test of Independence"

        # Perform chi-square test
        chi2, p_value, dof, expected_freq = stats.chi2_contingency(observed_array)
        statistic = chi2

        # Calculate Cramér's V (effect size)
        n = np.sum(observed_array)
        min_dim = min(observed_array.shape[0] - 1, observed_array.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        effect_size = cramers_v
        effect_size_name = "Cramér's V"

        # Descriptive statistics
        row_totals = observed_array.sum(axis=1)
        col_totals = observed_array.sum(axis=0)

        desc_stats = {
            'Observed Table': observed_array.tolist(),
            'Expected Table': expected_freq.tolist(),
            'Row Totals': row_totals.tolist(),
            'Column Totals': col_totals.tolist(),
            'Total N': int(n)
        }

        sample_sizes = {'total': int(n)}

        # Interpretation
        significant = p_value < alpha
        interpretation = (
            f"The chi-square test {'shows' if significant else 'does not show'} "
            f"a significant association between the row and column variables "
            f"at the {alpha} significance level (χ² = {chi2:.4f}, df = {dof}, p = {p_value:.4f})."
        )

    else:  # goodness_of_fit
        test_name = "Chi-Square Goodness of Fit Test"

        if expected is None:
            # Uniform distribution
            expected = np.ones_like(observed_array) * np.mean(observed_array)
        else:
            if isinstance(expected, (pd.DataFrame, pd.Series)):
                expected = expected.values
            expected = np.asarray(expected)

        # Perform chi-square test
        chi2, p_value = stats.chisquare(observed_array, expected)
        statistic = chi2
        dof = len(observed_array) - 1

        # Effect size (w - Cohen's w for goodness of fit)
        n = np.sum(observed_array)
        effect_size = np.sqrt(chi2 / n)
        effect_size_name = "Cohen's w"

        desc_stats = {
            'Observed': observed_array.tolist(),
            'Expected': expected.tolist(),
            'Residuals': (observed_array - expected).tolist(),
            'Total N': int(n)
        }

        sample_sizes = {'total': int(n)}

        significant = p_value < alpha
        interpretation = (
            f"The observed distribution {'significantly differs' if significant else 'does not significantly differ'} "
            f"from the expected distribution at the {alpha} significance level "
            f"(χ² = {chi2:.4f}, df = {dof}, p = {p_value:.4f})."
        )

    # Effect size interpretation
    if test_type == 'independence':
        if cramers_v < 0.1:
            effect_desc = "negligible"
        elif cramers_v < 0.3:
            effect_desc = "small"
        elif cramers_v < 0.5:
            effect_desc = "medium"
        else:
            effect_desc = "large"
    else:
        if effect_size < 0.1:
            effect_desc = "negligible"
        elif effect_size < 0.3:
            effect_desc = "small"
        elif effect_size < 0.5:
            effect_desc = "medium"
        else:
            effect_desc = "large"

    interpretation += f" The effect size ({effect_size_name} = {effect_size:.4f}) is {effect_desc}."

    # Recommendation
    if significant:
        recommendation = (
            f"Reject the null hypothesis. There is a significant {'association' if test_type == 'independence' else 'difference'}. "
            f"The effect size suggests {effect_desc} practical significance."
        )
    else:
        recommendation = (
            f"Fail to reject the null hypothesis. There is insufficient evidence of {'association' if test_type == 'independence' else 'difference'}."
        )

    # Confidence interval (approximation for chi-square)
    ci_low = chi2 - 1.96 * np.sqrt(2 * dof)
    ci_high = chi2 + 1.96 * np.sqrt(2 * dof)
    ci = (max(0, ci_low), ci_high)

    return TestResult(
        test_name=test_name,
        statistic=statistic,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        effect_size=effect_size,
        effect_size_name=effect_size_name,
        confidence_interval=ci,
        sample_sizes=sample_sizes,
        descriptive_stats=desc_stats,
        interpretation=interpretation,
        recommendation=recommendation
    )


# ==================== NON-PARAMETRIC TESTS ====================

def mann_whitney_test(
    group1: Union[np.ndarray, pd.Series, List],
    group2: Union[np.ndarray, pd.Series, List],
    alternative: str = 'two-sided',
    alpha: float = 0.05
) -> TestResult:
    """
    Mann-Whitney U test (non-parametric alternative to independent t-test).

    Use when:
    - Data is not normally distributed
    - Ordinal data
    - Small sample sizes
    - Outliers present

    Parameters:
    -----------
    group1 : array-like
        First group of observations.
    group2 : array-like
        Second group of observations.
    alternative : str
        Alternative hypothesis: 'two-sided', 'greater', or 'less'.
    alpha : float
        Significance level.

    Returns:
    --------
    TestResult
        Comprehensive test results.

    Examples:
    ---------
    >>> # Compare satisfaction scores (ordinal data, 1-5 scale)
    >>> control = [3, 4, 3, 5, 4, 3, 4, 5, 3, 4]
    >>> treatment = [4, 5, 4, 5, 5, 4, 5, 5, 4, 5]
    >>> result = mann_whitney_test(control, treatment)

    >>> # One-sided test
    >>> result = mann_whitney_test(control, treatment, alternative='less')
    """
    # Convert to numpy arrays and remove NaN
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    # Perform Mann-Whitney U test
    statistic, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)

    # Effect size (rank-biserial correlation)
    n1, n2 = len(group1), len(group2)
    r = 1 - (2 * statistic) / (n1 * n2)  # Rank-biserial correlation
    effect_size = r

    # Descriptive statistics (medians and IQR)
    desc_stats = {
        'Group 1': {
            'median': np.median(group1),
            'mean': np.mean(group1),
            'q1': np.percentile(group1, 25),
            'q3': np.percentile(group1, 75),
            'n': n1
        },
        'Group 2': {
            'median': np.median(group2),
            'mean': np.mean(group2),
            'q1': np.percentile(group2, 25),
            'q3': np.percentile(group2, 75),
            'n': n2
        }
    }

    sample_sizes = {'group_1': n1, 'group_2': n2}

    significant = p_value < alpha

    interpretation = (
        f"Group 1 (median = {desc_stats['Group 1']['median']:.4f}) and "
        f"Group 2 (median = {desc_stats['Group 2']['median']:.4f}) have "
        f"{'significantly' if significant else 'not significantly'} different distributions "
        f"at the {alpha} significance level (U = {statistic:.4f}, p = {p_value:.4f})."
    )

    # Effect size interpretation
    abs_r = abs(r)
    if abs_r < 0.1:
        effect_desc = "negligible"
    elif abs_r < 0.3:
        effect_desc = "small"
    elif abs_r < 0.5:
        effect_desc = "medium"
    else:
        effect_desc = "large"

    interpretation += f" The effect size (r = {r:.4f}) is {effect_desc}."

    # Recommendation
    if significant:
        winner = "Group 2" if np.median(group2) > np.median(group1) else "Group 1"
        recommendation = (
            f"Reject the null hypothesis. {winner} has a significantly higher median. "
            f"The effect size suggests {effect_desc} practical significance."
        )
    else:
        recommendation = (
            "Fail to reject the null hypothesis. There is insufficient evidence of a difference in distributions."
        )

    return TestResult(
        test_name="Mann-Whitney U Test",
        statistic=statistic,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        effect_size=effect_size,
        effect_size_name="Rank-biserial correlation (r)",
        confidence_interval=None,  # Not typically computed for Mann-Whitney
        sample_sizes=sample_sizes,
        descriptive_stats=desc_stats,
        interpretation=interpretation,
        recommendation=recommendation
    )


def wilcoxon_test(
    group1: Union[np.ndarray, pd.Series, List],
    group2: Union[np.ndarray, pd.Series, List],
    alternative: str = 'two-sided',
    alpha: float = 0.05
) -> TestResult:
    """
    Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Use for paired/matched samples when data is not normally distributed.

    Parameters:
    -----------
    group1 : array-like
        First set of paired observations.
    group2 : array-like
        Second set of paired observations.
    alternative : str
        Alternative hypothesis: 'two-sided', 'greater', or 'less'.
    alpha : float
        Significance level.

    Returns:
    --------
    TestResult
        Comprehensive test results.

    Examples:
    ---------
    >>> # Before/after treatment comparison
    >>> before = [8, 7, 9, 6, 8, 7, 9, 8]
    >>> after = [6, 5, 7, 5, 6, 4, 6, 5]
    >>> result = wilcoxon_test(before, after)
    """
    # Convert to numpy arrays and remove NaN
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    # Check for equal length
    if len(group1) != len(group2):
        raise ValueError("Wilcoxon test requires paired samples of equal length")

    # Remove pairs with NaN
    mask = ~(np.isnan(group1) | np.isnan(group2))
    group1 = group1[mask]
    group2 = group2[mask]

    # Perform Wilcoxon signed-rank test
    statistic, p_value = stats.wilcoxon(group1, group2, alternative=alternative)

    # Calculate differences
    diff = group1 - group2

    # Effect size (r = Z / sqrt(N))
    n = len(diff)
    z_score = (statistic - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    r = z_score / np.sqrt(n)
    effect_size = r

    # Descriptive statistics
    desc_stats = {
        'Group 1': {
            'median': np.median(group1),
            'mean': np.mean(group1),
            'n': len(group1)
        },
        'Group 2': {
            'median': np.median(group2),
            'mean': np.mean(group2),
            'n': len(group2)
        },
        'Difference': {
            'median': np.median(diff),
            'mean': np.mean(diff)
        }
    }

    sample_sizes = {'paired_samples': n}

    significant = p_value < alpha

    interpretation = (
        f"The median difference between paired observations ({desc_stats['Difference']['median']:.4f}) is "
        f"{'significantly' if significant else 'not significantly'} different from zero "
        f"at the {alpha} significance level (W = {statistic:.4f}, p = {p_value:.4f})."
    )

    # Effect size interpretation
    abs_r = abs(r)
    if abs_r < 0.1:
        effect_desc = "negligible"
    elif abs_r < 0.3:
        effect_desc = "small"
    elif abs_r < 0.5:
        effect_desc = "medium"
    else:
        effect_desc = "large"

    interpretation += f" The effect size (r = {r:.4f}) is {effect_desc}."

    # Recommendation
    if significant:
        recommendation = (
            f"Reject the null hypothesis. There is a significant difference between paired observations. "
            f"The effect size suggests {effect_desc} practical significance."
        )
    else:
        recommendation = (
            "Fail to reject the null hypothesis. There is insufficient evidence of a difference."
        )

    return TestResult(
        test_name="Wilcoxon Signed-Rank Test",
        statistic=statistic,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        effect_size=effect_size,
        effect_size_name="Effect size (r)",
        confidence_interval=None,
        sample_sizes=sample_sizes,
        descriptive_stats=desc_stats,
        interpretation=interpretation,
        recommendation=recommendation
    )


def kruskal_wallis_test(
    *groups,
    alpha: float = 0.05
) -> TestResult:
    """
    Kruskal-Wallis H-test (non-parametric alternative to one-way ANOVA).

    Use for comparing 3+ groups when data is not normally distributed.

    Parameters:
    -----------
    *groups : array-like
        Two or more groups to compare.
    alpha : float
        Significance level.

    Returns:
    --------
    TestResult
        Comprehensive test results.

    Examples:
    ---------
    >>> # Compare satisfaction across 3 regions
    >>> region_a = [3, 4, 3, 5, 4, 3]
    >>> region_b = [4, 5, 4, 5, 5, 4]
    >>> region_c = [2, 3, 2, 4, 3, 2]
    >>> result = kruskal_wallis_test(region_a, region_b, region_c)
    """
    if len(groups) < 2:
        raise ValueError("Kruskal-Wallis test requires at least 2 groups")

    # Convert to numpy arrays and remove NaN
    clean_groups = []
    for group in groups:
        g = np.asarray(group)
        g = g[~np.isnan(g)]
        clean_groups.append(g)

    # Perform Kruskal-Wallis test
    statistic, p_value = stats.kruskal(*clean_groups)

    # Effect size (epsilon squared)
    n = sum(len(g) for g in clean_groups)
    k = len(clean_groups)
    epsilon_squared = (statistic - k + 1) / (n - k)
    effect_size = epsilon_squared

    # Descriptive statistics
    desc_stats = {}
    sample_sizes = {}
    for i, group in enumerate(clean_groups, 1):
        desc_stats[f'Group {i}'] = {
            'median': np.median(group),
            'mean': np.mean(group),
            'q1': np.percentile(group, 25),
            'q3': np.percentile(group, 75),
            'n': len(group)
        }
        sample_sizes[f'group_{i}'] = len(group)

    significant = p_value < alpha

    interpretation = (
        f"The Kruskal-Wallis test {'shows' if significant else 'does not show'} "
        f"a significant difference among the {k} groups at the {alpha} significance level "
        f"(H = {statistic:.4f}, df = {k-1}, p = {p_value:.4f})."
    )

    # Effect size interpretation
    if epsilon_squared < 0.01:
        effect_desc = "negligible"
    elif epsilon_squared < 0.06:
        effect_desc = "small"
    elif epsilon_squared < 0.14:
        effect_desc = "medium"
    else:
        effect_desc = "large"

    interpretation += f" The effect size (ε² = {epsilon_squared:.4f}) is {effect_desc}."

    # Recommendation
    if significant:
        recommendation = (
            f"Reject the null hypothesis. At least one group differs significantly from the others. "
            f"Consider post-hoc pairwise comparisons to identify which groups differ. "
            f"The effect size suggests {effect_desc} practical significance."
        )
    else:
        recommendation = (
            "Fail to reject the null hypothesis. There is insufficient evidence that the groups differ."
        )

    return TestResult(
        test_name="Kruskal-Wallis H-Test",
        statistic=statistic,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        effect_size=effect_size,
        effect_size_name="Epsilon squared (ε²)",
        confidence_interval=None,
        sample_sizes=sample_sizes,
        descriptive_stats=desc_stats,
        interpretation=interpretation,
        recommendation=recommendation
    )


# ==================== VARIANCE TESTS ====================

def levene_test(
    *groups,
    center: str = 'median',
    alpha: float = 0.05
) -> TestResult:
    """
    Levene's test for equality of variances.

    Tests whether groups have equal variances (homoscedasticity).
    Use before performing parametric tests that assume equal variances.

    Parameters:
    -----------
    *groups : array-like
        Two or more groups to compare.
    center : str
        Method to compute center: 'median' (default, more robust) or 'mean'.
    alpha : float
        Significance level.

    Returns:
    --------
    TestResult
        Comprehensive test results.

    Examples:
    ---------
    >>> # Check if variances are equal before t-test
    >>> control = [23, 25, 27, 22, 24, 26, 28, 25, 23, 26]
    >>> treatment = [28, 30, 32, 29, 31, 33, 30, 29, 31, 30]
    >>> result = levene_test(control, treatment)
    >>>
    >>> if not result.significant:
    ...     # Variances are equal, use standard t-test
    ...     print("Use equal_var=True in t-test")
    ... else:
    ...     # Variances are unequal, use Welch's t-test
    ...     print("Use equal_var=False in t-test")
    """
    if len(groups) < 2:
        raise ValueError("Levene's test requires at least 2 groups")

    # Convert to numpy arrays and remove NaN
    clean_groups = []
    for group in groups:
        g = np.asarray(group)
        g = g[~np.isnan(g)]
        clean_groups.append(g)

    # Perform Levene's test
    statistic, p_value = stats.levene(*clean_groups, center=center)

    # Descriptive statistics (variances and std)
    desc_stats = {}
    sample_sizes = {}
    variances = []
    for i, group in enumerate(clean_groups, 1):
        var = np.var(group, ddof=1)
        variances.append(var)
        desc_stats[f'Group {i}'] = {
            'variance': var,
            'std': np.std(group, ddof=1),
            'n': len(group)
        }
        sample_sizes[f'group_{i}'] = len(group)

    # Effect size (ratio of largest to smallest variance)
    max_var = max(variances)
    min_var = min(variances)
    variance_ratio = max_var / min_var if min_var > 0 else np.inf
    effect_size = variance_ratio

    significant = p_value < alpha

    interpretation = (
        f"Levene's test {'indicates' if significant else 'does not indicate'} "
        f"significantly different variances among groups at the {alpha} significance level "
        f"(W = {statistic:.4f}, p = {p_value:.4f}). "
    )

    if significant:
        interpretation += (
            "The assumption of homogeneity of variance is violated. "
            "Consider using tests that don't assume equal variances (e.g., Welch's t-test)."
        )
    else:
        interpretation += (
            "The assumption of homogeneity of variance is met. "
            "Standard parametric tests assuming equal variances can be used."
        )

    # Recommendation
    if significant:
        recommendation = (
            f"Reject the null hypothesis of equal variances. "
            f"The variance ratio ({variance_ratio:.4f}) suggests unequal spread. "
            f"Use Welch's t-test or non-parametric alternatives."
        )
    else:
        recommendation = (
            "Fail to reject the null hypothesis of equal variances. "
            "Standard t-test or ANOVA with equal variance assumption can be used."
        )

    return TestResult(
        test_name="Levene's Test for Equality of Variances",
        statistic=statistic,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        effect_size=effect_size,
        effect_size_name="Variance Ratio (max/min)",
        confidence_interval=None,
        sample_sizes=sample_sizes,
        descriptive_stats=desc_stats,
        interpretation=interpretation,
        recommendation=recommendation
    )


# ==================== NORMALITY TESTS ====================

def normality_test(
    data: Union[np.ndarray, pd.Series, List],
    method: str = 'shapiro',
    alpha: float = 0.05
) -> TestResult:
    """
    Test for normality of data distribution.

    Important for determining whether to use parametric or non-parametric tests.

    Parameters:
    -----------
    data : array-like
        Data to test for normality.
    method : str
        Test method: 'shapiro' (Shapiro-Wilk, default),
        'anderson' (Anderson-Darling), or 'kstest' (Kolmogorov-Smirnov).
    alpha : float
        Significance level.

    Returns:
    --------
    TestResult
        Comprehensive test results.

    Examples:
    ---------
    >>> # Test if data is normally distributed
    >>> data = [23, 25, 27, 22, 24, 26, 28, 25, 23, 26, 24, 27]
    >>> result = normality_test(data)
    >>>
    >>> if result.significant:
    ...     print("Data is NOT normally distributed - use non-parametric tests")
    ... else:
    ...     print("Data is normally distributed - parametric tests OK")

    >>> # Try different test
    >>> result = normality_test(data, method='anderson')
    """
    # Convert to numpy array and remove NaN
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    if len(data) < 3:
        raise ValueError("Normality tests require at least 3 observations")

    if method == 'shapiro':
        test_name = "Shapiro-Wilk Normality Test"
        statistic, p_value = stats.shapiro(data)

        desc_stats = {
            'Sample': {
                'mean': np.mean(data),
                'std': np.std(data, ddof=1),
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data),
                'n': len(data)
            }
        }

    elif method == 'kstest':
        test_name = "Kolmogorov-Smirnov Normality Test"
        # Standardize data
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        statistic, p_value = stats.kstest(data, 'norm', args=(mean, std))

        desc_stats = {
            'Sample': {
                'mean': mean,
                'std': std,
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data),
                'n': len(data)
            }
        }

    elif method == 'anderson':
        test_name = "Anderson-Darling Normality Test"
        result_anderson = stats.anderson(data, dist='norm')
        statistic = result_anderson.statistic

        # Find critical value for given alpha
        critical_values = result_anderson.critical_values
        significance_levels = result_anderson.significance_level / 100

        # Interpolate p-value (Anderson-Darling doesn't directly provide it)
        idx = np.searchsorted(critical_values, statistic)
        if idx == 0:
            p_value = significance_levels[0] / 2
        elif idx >= len(critical_values):
            p_value = 1 - significance_levels[-1]
        else:
            p_value = significance_levels[idx]

        desc_stats = {
            'Sample': {
                'mean': np.mean(data),
                'std': np.std(data, ddof=1),
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data),
                'n': len(data)
            }
        }
    else:
        raise ValueError(f"Unknown method: {method}. Use 'shapiro', 'kstest', or 'anderson'")

    significant = p_value < alpha

    # Interpretation
    interpretation = (
        f"The {method.capitalize()} test {'indicates' if significant else 'does not indicate'} "
        f"a significant departure from normality at the {alpha} significance level "
        f"(statistic = {statistic:.4f}, p = {p_value:.4f}). "
    )

    skewness = desc_stats['Sample']['skewness']
    kurtosis = desc_stats['Sample']['kurtosis']

    if abs(skewness) > 1:
        interpretation += f"The data shows {'positive' if skewness > 0 else 'negative'} skewness ({skewness:.4f}). "

    if abs(kurtosis) > 1:
        interpretation += f"The kurtosis ({kurtosis:.4f}) suggests {'heavy' if kurtosis > 0 else 'light'} tails. "

    # Recommendation
    if significant:
        recommendation = (
            "Reject the null hypothesis of normality. "
            "The data is NOT normally distributed. "
            "Consider using non-parametric tests (Mann-Whitney, Wilcoxon, Kruskal-Wallis) "
            "or transforming the data (log, sqrt, Box-Cox)."
        )
    else:
        recommendation = (
            "Fail to reject the null hypothesis of normality. "
            "The data appears to be normally distributed. "
            "Parametric tests (t-test, ANOVA) are appropriate."
        )

    sample_sizes = {'sample': len(data)}

    return TestResult(
        test_name=test_name,
        statistic=statistic,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        effect_size=None,
        effect_size_name=None,
        confidence_interval=None,
        sample_sizes=sample_sizes,
        descriptive_stats=desc_stats,
        interpretation=interpretation,
        recommendation=recommendation
    )
