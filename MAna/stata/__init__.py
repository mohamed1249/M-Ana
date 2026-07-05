"""Statistical testing and experimentation tools.

Use :mod:`MAna.stata` as the single public namespace for statistical tests,
effect sizes, A/B testing, and experiment visualizations.
"""

from .ab_testing import ABTest, ABTestResult, MultiVariantTest
from .associations import (
    correlation_matrix,
    correlation_test,
    linear_regression_test,
    partial_correlation,
)
from .bayesian import bayesian_proportion_ab_test
from .descriptive import (
    descriptive_statistics,
    frequency_table,
    grouped_descriptive_statistics,
)
from .hypothesis_tests import (
    TestResult,
    chi_square_test,
    kruskal_wallis_test,
    levene_test,
    mann_whitney_test,
    normality_test,
    proportion_test,
    t_test,
    wilcoxon_test,
)
from .utils import (
    absolute_lift,
    bonferroni_correction,
    bootstrap,
    bootstrap_test,
    cohens_d,
    cohens_h,
    confidence_interval_difference,
    confidence_interval_mean,
    confidence_interval_proportion,
    cramers_v,
    eta_squared,
    fdr_correction,
    glass_delta,
    holm_bonferroni_correction,
    interpret_effect_size,
    omega_squared,
    pooled_std,
    relative_lift,
    sidak_correction,
    standardize,
    winsorize,
)
from .power import required_sample_size_two_sample, two_sample_ttest_power
from importlib import import_module

_VISUALIZATION_EXPORTS = {
    "plot_ab_test_results",
    "plot_confidence_intervals",
    "plot_lift_analysis",
    "plot_sequential_results",
    "plot_conversion_funnel",
    "plot_sample_size_curve",
    "plot_power_curve",
    "plot_ab_dashboard",
}


def __getattr__(name):
    if name in _VISUALIZATION_EXPORTS:
        value = getattr(import_module(f"{__name__}.visualizations"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ABTest",
    "ABTestResult",
    "MultiVariantTest",
    "TestResult",
    "descriptive_statistics",
    "grouped_descriptive_statistics",
    "frequency_table",
    "correlation_test",
    "partial_correlation",
    "correlation_matrix",
    "linear_regression_test",
    "bayesian_proportion_ab_test",
    "two_sample_ttest_power",
    "required_sample_size_two_sample",
    "t_test",
    "proportion_test",
    "chi_square_test",
    "mann_whitney_test",
    "wilcoxon_test",
    "kruskal_wallis_test",
    "levene_test",
    "normality_test",
    "cohens_d",
    "cohens_h",
    "cramers_v",
    "glass_delta",
    "eta_squared",
    "omega_squared",
    "interpret_effect_size",
    "bonferroni_correction",
    "holm_bonferroni_correction",
    "fdr_correction",
    "sidak_correction",
    "confidence_interval_mean",
    "confidence_interval_proportion",
    "confidence_interval_difference",
    "bootstrap",
    "bootstrap_test",
    "standardize",
    "winsorize",
    "relative_lift",
    "absolute_lift",
    "pooled_std",
    "plot_ab_test_results",
    "plot_confidence_intervals",
    "plot_lift_analysis",
    "plot_sequential_results",
    "plot_conversion_funnel",
    "plot_sample_size_curve",
    "plot_power_curve",
    "plot_ab_dashboard",
]
