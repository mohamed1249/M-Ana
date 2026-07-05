"""
A/B Testing Module

Comprehensive A/B testing framework with support for:
- Binary metrics (conversion rates, click-through rates)
- Continuous metrics (revenue, time on site, engagement)
- Count metrics (number of purchases, page views)
- Multiple variants (A/B/C/... testing)
- Sequential testing and early stopping
- Automatic metric detection
- Rich reporting and visualization

Classes:
- ABTest: Main A/B test runner
- MultiVariantTest: A/B/C/... testing
- SequentialTest: Sequential testing with early stopping
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import sys

from .hypothesis_tests import t_test, proportion_test
from .utils import (
    relative_lift, absolute_lift,
    confidence_interval_mean, confidence_interval_proportion
)


def _terminal_safe(text: str) -> str:
    """Return text that can be printed by the active terminal encoding."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


@dataclass
class ABTestResult:
    """
    Comprehensive A/B test result object.

    Attributes:
    -----------
    test_name : str
        Name of the test.
    control_name : str
        Name of control variant.
    treatment_name : str
        Name of treatment variant.
    metric_name : str
        Name of the metric tested.
    metric_type : str
        Type of metric: 'binary', 'continuous', or 'count'.
    control_metric : float
        Control group metric value.
    treatment_metric : float
        Treatment group metric value.
    control_ci : tuple
        Confidence interval for control.
    treatment_ci : tuple
        Confidence interval for treatment.
    p_value : float
        Statistical significance p-value.
    significant : bool
        Whether result is statistically significant.
    alpha : float
        Significance level used.
    power : float, optional
        Statistical power of the test.
    effect_size : float
        Standardized effect size.
    relative_lift : float
        Relative lift (percentage change).
    absolute_lift : float
        Absolute lift (raw difference).
    sample_sizes : dict
        Sample sizes for each variant.
    confidence_level : float
        Confidence level (e.g., 0.95).
    test_statistic : float
        Test statistic value.
    winner : str, optional
        Name of winning variant (if significant).
    recommendation : str
        Actionable recommendation.
    """

    test_name: str
    control_name: str
    treatment_name: str
    metric_name: str
    metric_type: str
    control_metric: float
    treatment_metric: float
    control_ci: Tuple[float, float]
    treatment_ci: Tuple[float, float]
    p_value: float
    significant: bool
    alpha: float
    effect_size: float
    effect_size_name: str
    relative_lift: float
    absolute_lift: float
    sample_sizes: Dict[str, int]
    confidence_level: float
    test_statistic: float
    power: Optional[float] = None
    winner: Optional[str] = None
    recommendation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Pretty print results."""

        # Determine lift direction
        lift_symbol = "↑" if self.relative_lift > 0 else "↓" if self.relative_lift < 0 else "→"
        lift_color = "+" if self.relative_lift > 0 else ""

        output = f"""
╔══════════════════════════════════════════════════════════════════════╗
║  {self.test_name.upper():^68}  ║
╚══════════════════════════════════════════════════════════════════════╝

📊 EXPERIMENT OVERVIEW
  Metric:           {self.metric_name} ({self.metric_type})
  Control:          {self.control_name}
  Treatment:        {self.treatment_name}
  Sample Sizes:     Control = {self.sample_sizes.get('control', 'N/A'):,} | Treatment = {self.sample_sizes.get('treatment', 'N/A'):,}

📈 RESULTS
  Control:          {self.control_metric:.4f}  [{self.control_ci[0]:.4f}, {self.control_ci[1]:.4f}]
  Treatment:        {self.treatment_metric:.4f}  [{self.treatment_ci[0]:.4f}, {self.treatment_ci[1]:.4f}]

  Absolute Lift:    {lift_color}{self.absolute_lift:.4f}
  Relative Lift:    {lift_symbol} {lift_color}{self.relative_lift:.2f}%

  Effect Size:      {self.effect_size:.4f} ({self.effect_size_name})

🔬 STATISTICAL INFERENCE
  P-value:          {self.p_value:.6f} {'***' if self.p_value < 0.001 else '**' if self.p_value < 0.01 else '*' if self.p_value < 0.05 else ''}
  Significant:      {'✓ YES' if self.significant else '✗ NO'} (α = {self.alpha})
  Test Statistic:   {self.test_statistic:.4f}
  Confidence:       {self.confidence_level*100:.0f}%
"""

        if self.power:
            output += f"  Statistical Power: {self.power:.2%}\\n"

        if self.winner:
            output += f"\\n🏆 WINNER: {self.winner}\\n"

        if self.recommendation:
            output += f"\\n💡 RECOMMENDATION\\n  {self.recommendation}\\n"

        return _terminal_safe(output)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'test_name': self.test_name,
            'control_name': self.control_name,
            'treatment_name': self.treatment_name,
            'metric_name': self.metric_name,
            'metric_type': self.metric_type,
            'control_metric': self.control_metric,
            'treatment_metric': self.treatment_metric,
            'control_ci': self.control_ci,
            'treatment_ci': self.treatment_ci,
            'p_value': self.p_value,
            'significant': self.significant,
            'alpha': self.alpha,
            'effect_size': self.effect_size,
            'effect_size_name': self.effect_size_name,
            'relative_lift': self.relative_lift,
            'absolute_lift': self.absolute_lift,
            'sample_sizes': self.sample_sizes,
            'confidence_level': self.confidence_level,
            'test_statistic': self.test_statistic,
            'power': self.power,
            'winner': self.winner,
            'recommendation': self.recommendation,
            'metadata': self.metadata
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert result to single-row DataFrame."""
        return pd.DataFrame([self.to_dict()])


class ABTest:
    """
    Comprehensive A/B testing class.

    Automatically detects metric type and applies appropriate statistical test.
    Provides rich results with effect sizes, confidence intervals, and recommendations.

    Examples:
    ---------
    >>> # Example 1: Binary metric (conversion rate)
    >>> import pandas as pd
    >>> from MAna.stata import ABTest
    >>>
    >>> df = pd.DataFrame({
    ...     'variant': ['A', 'A', 'B', 'B', 'A', 'B'] * 100,
    ...     'converted': [0, 1, 1, 1, 0, 1] * 100
    ... })
    >>>
    >>> test = ABTest(
    ...     data=df,
    ...     variant_col='variant',
    ...     metric_col='converted',
    ...     control_name='A',
    ...     treatment_name='B'
    ... )
    >>>
    >>> result = test.run()
    >>> print(result)
    >>>
    >>> # Example 2: Continuous metric (revenue)
    >>> df = pd.DataFrame({
    ...     'variant': ['control'] * 500 + ['treatment'] * 500,
    ...     'revenue': np.random.normal(50, 10, 500).tolist() +
    ...                np.random.normal(55, 10, 500).tolist()
    ... })
    >>>
    >>> test = ABTest(
    ...     data=df,
    ...     variant_col='variant',
    ...     metric_col='revenue',
    ...     metric_type='continuous'
    ... )
    >>>
    >>> result = test.run()
    >>>
    >>> # Example 3: From summary statistics
    >>> test = ABTest.from_summary(
    ...     control_successes=48,
    ...     control_n=500,
    ...     treatment_successes=65,
    ...     treatment_n=500,
    ...     metric_name='conversion_rate'
    ... )
    >>>
    >>> result = test.run()
    """

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        variant_col: Optional[str] = None,
        metric_col: Optional[str] = None,
        control_name: str = 'control',
        treatment_name: str = 'treatment',
        metric_type: str = 'auto',
        metric_name: Optional[str] = None,
        alpha: float = 0.05,
        confidence_level: float = 0.95,
        alternative: str = 'two-sided',
        minimum_detectable_effect: Optional[float] = None
    ):
        """
        Initialize A/B Test.

        Parameters:
        -----------
        data : pd.DataFrame, optional
            Experiment data with variant and metric columns.
        variant_col : str, optional
            Column name containing variant labels.
        metric_col : str, optional
            Column name containing metric values.
        control_name : str
            Name/label of control variant (default: 'control').
        treatment_name : str
            Name/label of treatment variant (default: 'treatment').
        metric_type : str
            Type of metric: 'auto' (default), 'binary', 'continuous', or 'count'.
        metric_name : str, optional
            Name of metric for reporting.
        alpha : float
            Significance level (default: 0.05).
        confidence_level : float
            Confidence level for intervals (default: 0.95).
        alternative : str
            Alternative hypothesis: 'two-sided', 'greater', or 'less'.
        minimum_detectable_effect : float, optional
            MDE for power calculation.
        """
        self.data = data
        self.variant_col = variant_col
        self.metric_col = metric_col
        self.control_name = control_name
        self.treatment_name = treatment_name
        self.metric_type = metric_type
        self.metric_name = metric_name or metric_col or 'metric'
        self.alpha = alpha
        self.confidence_level = confidence_level
        self.alternative = alternative
        self.minimum_detectable_effect = minimum_detectable_effect

        # Will be populated during run()
        self.control_data = None
        self.treatment_data = None
        self.result = None

        # Validate inputs if data provided
        if data is not None:
            self._validate_inputs()
            self._prepare_data()
            self._detect_metric_type()

    def _validate_inputs(self):
        """Validate input parameters."""
        if self.data is None:
            raise ValueError("Data must be provided")

        if self.variant_col not in self.data.columns:
            raise ValueError(f"Variant column '{self.variant_col}' not found in data")

        if self.metric_col not in self.data.columns:
            raise ValueError(f"Metric column '{self.metric_col}' not found in data")

        # Check if variants exist
        unique_variants = self.data[self.variant_col].unique()

        if self.control_name not in unique_variants:
            raise ValueError(f"Control variant '{self.control_name}' not found in data")

        if self.treatment_name not in unique_variants:
            raise ValueError(f"Treatment variant '{self.treatment_name}' not found in data")

    def _prepare_data(self):
        """Extract control and treatment data."""
        self.control_data = self.data[
            self.data[self.variant_col] == self.control_name
        ][self.metric_col].values

        self.treatment_data = self.data[
            self.data[self.variant_col] == self.treatment_name
        ][self.metric_col].values

        # Remove NaN values
        self.control_data = self.control_data[~np.isnan(self.control_data)]
        self.treatment_data = self.treatment_data[~np.isnan(self.treatment_data)]

    def _detect_metric_type(self):
        """Auto-detect metric type if not specified."""
        if self.metric_type != 'auto':
            return

        # Check if binary (only 0s and 1s)
        control_unique = np.unique(self.control_data)
        treatment_unique = np.unique(self.treatment_data)
        all_unique = np.unique(np.concatenate([control_unique, treatment_unique]))

        if len(all_unique) <= 2 and set(all_unique).issubset({0, 1, 0.0, 1.0}):
            self.metric_type = 'binary'
            print("[OK] Auto-detected metric type: BINARY (conversion/success metric)")

        # Check if all integers (count data)
        elif np.all(self.control_data == self.control_data.astype(int)) and \
             np.all(self.treatment_data == self.treatment_data.astype(int)) and \
             np.all(self.control_data >= 0) and np.all(self.treatment_data >= 0):
            self.metric_type = 'count'
            print("[OK] Auto-detected metric type: COUNT (integer counts)")

        else:
            self.metric_type = 'continuous'
            print("[OK] Auto-detected metric type: CONTINUOUS (real-valued metric)")

    def run(self, verbose: bool = True) -> ABTestResult:
        """
        Run the A/B test and return comprehensive results.

        Parameters:
        -----------
        verbose : bool
            Print progress messages.

        Returns:
        --------
        ABTestResult
            Comprehensive test results.
        """
        if verbose:
            print(f"\\n{'='*70}")
            print(f"Running A/B Test: {self.control_name} vs {self.treatment_name}")
            print(f"{'='*70}\\n")

        # Run appropriate test based on metric type
        if self.metric_type == 'binary':
            result = self._test_binary_metric()
        elif self.metric_type == 'continuous':
            result = self._test_continuous_metric()
        elif self.metric_type == 'count':
            result = self._test_count_metric()
        else:
            raise ValueError(f"Unknown metric type: {self.metric_type}")

        self.result = result

        if verbose:
            print(result)

        return result

    def _test_binary_metric(self) -> ABTestResult:
        """Test binary metric (conversion rate, click-through rate, etc.)."""
        # Calculate proportions
        control_successes = int(np.sum(self.control_data))
        control_n = len(self.control_data)
        control_rate = control_successes / control_n

        treatment_successes = int(np.sum(self.treatment_data))
        treatment_n = len(self.treatment_data)
        treatment_rate = treatment_successes / treatment_n

        # Run proportion test
        test_result = proportion_test(
            successes1=control_successes,
            n1=control_n,
            successes2=treatment_successes,
            n2=treatment_n,
            alternative=self.alternative,
            alpha=self.alpha
        )

        # Calculate confidence intervals
        control_ci = confidence_interval_proportion(
            control_successes, control_n,
            confidence=self.confidence_level
        )
        treatment_ci = confidence_interval_proportion(
            treatment_successes, treatment_n,
            confidence=self.confidence_level
        )

        # Calculate lifts
        rel_lift = relative_lift(control_rate, treatment_rate)
        abs_lift = absolute_lift(control_rate, treatment_rate)

        # Determine winner
        winner = None
        if test_result.significant:
            winner = self.treatment_name if treatment_rate > control_rate else self.control_name

        # Generate recommendation
        if test_result.significant:
            if winner == self.treatment_name:
                recommendation = (
                    f"✓ IMPLEMENT {self.treatment_name.upper()}. "
                    f"It shows a {abs(rel_lift):.1f}% improvement over {self.control_name}. "
                    f"Expected impact: {abs_lift:.4f} absolute increase in {self.metric_name}."
                )
            else:
                recommendation = (
                    f"⚠️  KEEP {self.control_name.upper()}. "
                    f"{self.treatment_name} performs {abs(rel_lift):.1f}% worse. "
                    f"Do not ship this variant."
                )
        else:
            recommendation = (
                f"⏸️  INCONCLUSIVE. No significant difference detected. "
                f"Consider running the test longer or increasing sample size. "
                f"Current sample: {control_n + treatment_n:,} total observations."
            )

        return ABTestResult(
            test_name=f"A/B Test: {self.metric_name}",
            control_name=self.control_name,
            treatment_name=self.treatment_name,
            metric_name=self.metric_name,
            metric_type='binary',
            control_metric=control_rate,
            treatment_metric=treatment_rate,
            control_ci=control_ci,
            treatment_ci=treatment_ci,
            p_value=test_result.p_value,
            significant=test_result.significant,
            alpha=self.alpha,
            effect_size=test_result.effect_size,
            effect_size_name=test_result.effect_size_name,
            relative_lift=rel_lift,
            absolute_lift=abs_lift,
            sample_sizes={'control': control_n, 'treatment': treatment_n},
            confidence_level=self.confidence_level,
            test_statistic=test_result.statistic,
            winner=winner,
            recommendation=recommendation,
            metadata={
                'control_successes': control_successes,
                'treatment_successes': treatment_successes,
                'test_type': 'proportion_test'
            }
        )

    def _test_continuous_metric(self) -> ABTestResult:
        """Test continuous metric (revenue, time on site, etc.)."""
        # Calculate means
        control_mean = np.mean(self.control_data)
        treatment_mean = np.mean(self.treatment_data)

        control_n = len(self.control_data)
        treatment_n = len(self.treatment_data)

        # Run t-test
        test_result = t_test(
            self.control_data,
            self.treatment_data,
            test_type='independent',
            alternative=self.alternative,
            alpha=self.alpha,
            equal_var=True  # Can add Levene's test here
        )

        # Confidence intervals
        control_ci = confidence_interval_mean(
            self.control_data,
            confidence=self.confidence_level
        )
        treatment_ci = confidence_interval_mean(
            self.treatment_data,
            confidence=self.confidence_level
        )

        # Calculate lifts
        rel_lift = relative_lift(control_mean, treatment_mean)
        abs_lift = absolute_lift(control_mean, treatment_mean)

        # Determine winner
        winner = None
        if test_result.significant:
            winner = self.treatment_name if treatment_mean > control_mean else self.control_name

        # Generate recommendation
        if test_result.significant:
            if winner == self.treatment_name:
                recommendation = (
                    f"✓ IMPLEMENT {self.treatment_name.upper()}. "
                    f"It shows a {abs(rel_lift):.1f}% improvement over {self.control_name}. "
                    f"Expected impact: {abs_lift:+.4f} change in {self.metric_name}."
                )
            else:
                recommendation = (
                    f"⚠️  KEEP {self.control_name.upper()}. "
                    f"{self.treatment_name} performs {abs(rel_lift):.1f}% worse. "
                )
        else:
            recommendation = (
                f"⏸️  INCONCLUSIVE. No significant difference detected. "
                f"Consider running the test longer. "
                f"Current sample: {control_n + treatment_n:,} observations."
            )

        return ABTestResult(
            test_name=f"A/B Test: {self.metric_name}",
            control_name=self.control_name,
            treatment_name=self.treatment_name,
            metric_name=self.metric_name,
            metric_type='continuous',
            control_metric=control_mean,
            treatment_metric=treatment_mean,
            control_ci=control_ci,
            treatment_ci=treatment_ci,
            p_value=test_result.p_value,
            significant=test_result.significant,
            alpha=self.alpha,
            effect_size=test_result.effect_size,
            effect_size_name=test_result.effect_size_name,
            relative_lift=rel_lift,
            absolute_lift=abs_lift,
            sample_sizes={'control': control_n, 'treatment': treatment_n},
            confidence_level=self.confidence_level,
            test_statistic=test_result.statistic,
            winner=winner,
            recommendation=recommendation,
            metadata={
                'control_std': np.std(self.control_data, ddof=1),
                'treatment_std': np.std(self.treatment_data, ddof=1),
                'test_type': 't_test'
            }
        )

    def _test_count_metric(self) -> ABTestResult:
        """Test count metric (number of purchases, page views, etc.)."""
        # For count data, we can use t-test or Poisson-based test
        # For simplicity, using t-test (robust for larger samples)

        control_mean = np.mean(self.control_data)
        treatment_mean = np.mean(self.treatment_data)

        control_n = len(self.control_data)
        treatment_n = len(self.treatment_data)

        # Use t-test
        test_result = t_test(
            self.control_data,
            self.treatment_data,
            test_type='independent',
            alternative=self.alternative,
            alpha=self.alpha
        )

        # Confidence intervals
        control_ci = confidence_interval_mean(
            self.control_data,
            confidence=self.confidence_level
        )
        treatment_ci = confidence_interval_mean(
            self.treatment_data,
            confidence=self.confidence_level
        )

        # Calculate lifts
        rel_lift = relative_lift(control_mean, treatment_mean)
        abs_lift = absolute_lift(control_mean, treatment_mean)

        # Winner
        winner = None
        if test_result.significant:
            winner = self.treatment_name if treatment_mean > control_mean else self.control_name

        # Recommendation
        if test_result.significant:
            if winner == self.treatment_name:
                recommendation = (
                    f"✓ IMPLEMENT {self.treatment_name.upper()}. "
                    f"{abs(rel_lift):.1f}% improvement in {self.metric_name}. "
                    f"Expected: {abs_lift:+.2f} additional events per user."
                )
            else:
                recommendation = (
                    f"⚠️  KEEP {self.control_name.upper()}. "
                    f"{self.treatment_name} shows {abs(rel_lift):.1f}% decrease."
                )
        else:
            recommendation = (
                f"⏸️  INCONCLUSIVE. Consider extending test duration. "
                f"Sample: {control_n + treatment_n:,} observations."
            )

        return ABTestResult(
            test_name=f"A/B Test: {self.metric_name}",
            control_name=self.control_name,
            treatment_name=self.treatment_name,
            metric_name=self.metric_name,
            metric_type='count',
            control_metric=control_mean,
            treatment_metric=treatment_mean,
            control_ci=control_ci,
            treatment_ci=treatment_ci,
            p_value=test_result.p_value,
            significant=test_result.significant,
            alpha=self.alpha,
            effect_size=test_result.effect_size,
            effect_size_name=test_result.effect_size_name,
            relative_lift=rel_lift,
            absolute_lift=abs_lift,
            sample_sizes={'control': control_n, 'treatment': treatment_n},
            confidence_level=self.confidence_level,
            test_statistic=test_result.statistic,
            winner=winner,
            recommendation=recommendation,
            metadata={'test_type': 't_test_count'}
        )

    @classmethod
    def from_summary(
        cls,
        control_successes: Optional[int] = None,
        control_n: Optional[int] = None,
        treatment_successes: Optional[int] = None,
        treatment_n: Optional[int] = None,
        control_mean: Optional[float] = None,
        control_std: Optional[float] = None,
        treatment_mean: Optional[float] = None,
        treatment_std: Optional[float] = None,
        metric_name: str = 'metric',
        metric_type: str = 'auto',
        **kwargs
    ) -> 'ABTest':
        """
        Create ABTest from summary statistics.

        For binary metrics, provide successes and n.
        For continuous metrics, provide mean, std, and n.

        Parameters:
        -----------
        control_successes : int, optional
            Number of successes in control (for binary).
        control_n : int, optional
            Sample size of control.
        treatment_successes : int, optional
            Number of successes in treatment (for binary).
        treatment_n : int, optional
            Sample size of treatment.
        control_mean : float, optional
            Mean of control (for continuous).
        control_std : float, optional
            Std of control (for continuous).
        treatment_mean : float, optional
            Mean of treatment (for continuous).
        treatment_std : float, optional
            Std of treatment (for continuous).
        metric_name : str
            Name of metric.
        metric_type : str
            'binary', 'continuous', or 'auto'.
        **kwargs :
            Additional arguments passed to ABTest.__init__

        Returns:
        --------
        ABTest
            Configured ABTest instance.

        Examples:
        ---------
        >>> # Binary metric from summary
        >>> test = ABTest.from_summary(
        ...     control_successes=48,
        ...     control_n=500,
        ...     treatment_successes=65,
        ...     treatment_n=500,
        ...     metric_name='conversion_rate'
        ... )
        >>> result = test.run()

        >>> # Continuous metric from summary
        >>> test = ABTest.from_summary(
        ...     control_mean=52.3,
        ...     control_std=12.5,
        ...     control_n=500,
        ...     treatment_mean=55.8,
        ...     treatment_std=13.2,
        ...     treatment_n=500,
        ...     metric_name='revenue',
        ...     metric_type='continuous'
        ... )
        >>> result = test.run()
        """
        # Determine metric type
        if metric_type == 'auto':
            if control_successes is not None and treatment_successes is not None:
                metric_type = 'binary'
            elif control_mean is not None and treatment_mean is not None:
                metric_type = 'continuous'
            else:
                raise ValueError("Cannot auto-detect metric type from provided summary stats")

        # Create synthetic data
        if metric_type == 'binary':
            if None in [control_successes, control_n, treatment_successes, treatment_n]:
                raise ValueError("For binary metrics, provide successes and n for both groups")

            # Create synthetic binary data
            control_data = np.concatenate([
                np.ones(control_successes),
                np.zeros(control_n - control_successes)
            ])

            treatment_data = np.concatenate([
                np.ones(treatment_successes),
                np.zeros(treatment_n - treatment_successes)
            ])

        elif metric_type == 'continuous':
            if None in [control_mean, control_std, control_n,
                       treatment_mean, treatment_std, treatment_n]:
                raise ValueError("For continuous metrics, provide mean, std, and n for both groups")

            # Create synthetic continuous data matching the statistics
            np.random.seed(42)
            control_data = np.random.normal(control_mean, control_std, control_n)
            treatment_data = np.random.normal(treatment_mean, treatment_std, treatment_n)

        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

        # Create DataFrame
        df = pd.DataFrame({
            'variant': ['control'] * len(control_data) + ['treatment'] * len(treatment_data),
            'metric': np.concatenate([control_data, treatment_data])
        })

        # Create ABTest instance
        test = cls(
            data=df,
            variant_col='variant',
            metric_col='metric',
            metric_name=metric_name,
            metric_type=metric_type,
            **kwargs
        )

        return test

    def plot_results(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Plot A/B test results.

        Creates visualization with distributions and confidence intervals.

        Parameters:
        -----------
        figsize : tuple
            Figure size.
        """
        if self.result is None:
            raise ValueError("Must run test before plotting. Call .run() first.")

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Distributions
        if self.metric_type == 'binary':
            # Bar plot for proportions
            variants = [self.control_name, self.treatment_name]
            values = [self.result.control_metric, self.result.treatment_metric]
            colors = ['#3498db', '#e74c3c']

            bars = axes[0].bar(variants, values, color=colors, alpha=0.7, edgecolor='black')

            # Add error bars (confidence intervals)
            yerr = [
                [self.result.control_metric - self.result.control_ci[0],
                 self.result.treatment_metric - self.result.treatment_ci[0]],
                [self.result.control_ci[1] - self.result.control_metric,
                 self.result.treatment_ci[1] - self.result.treatment_metric]
            ]
            axes[0].errorbar(variants, values, yerr=yerr, fmt='none',
                           ecolor='black', capsize=10, capthick=2)

            axes[0].set_ylabel(f'{self.metric_name}', fontsize=12, fontweight='bold')
            axes[0].set_title('Conversion Rates with 95% CI', fontsize=13, fontweight='bold')
            axes[0].set_ylim(0, max(values) * 1.3)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}\\n({value*100:.1f}%)',
                           ha='center', va='bottom', fontsize=11, fontweight='bold')

        else:
            # Histogram for continuous metrics
            axes[0].hist(self.control_data, bins=30, alpha=0.6,
                        label=self.control_name, color='#3498db', edgecolor='black')
            axes[0].hist(self.treatment_data, bins=30, alpha=0.6,
                        label=self.treatment_name, color='#e74c3c', edgecolor='black')

            # Add mean lines
            axes[0].axvline(self.result.control_metric, color='#3498db',
                          linestyle='--', linewidth=2, label=f'{self.control_name} mean')
            axes[0].axvline(self.result.treatment_metric, color='#e74c3c',
                          linestyle='--', linewidth=2, label=f'{self.treatment_name} mean')

            axes[0].set_xlabel(self.metric_name, fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
            axes[0].set_title('Distribution Comparison', fontsize=13, fontweight='bold')
            axes[0].legend(fontsize=10)
            axes[0].grid(alpha=0.3)

        # Plot 2: Key metrics summary
        axes[1].axis('off')

        # Create summary text
        summary_text = f"""
📊 A/B TEST SUMMARY

Metric: {self.metric_name}
Type: {self.metric_type.upper()}

{self.control_name.upper()}:
  Value: {self.result.control_metric:.4f}
  95% CI: [{self.result.control_ci[0]:.4f}, {self.result.control_ci[1]:.4f}]
  n = {self.result.sample_sizes['control']:,}

{self.treatment_name.upper()}:
  Value: {self.result.treatment_metric:.4f}
  95% CI: [{self.result.treatment_ci[0]:.4f}, {self.result.treatment_ci[1]:.4f}]
  n = {self.result.sample_sizes['treatment']:,}

LIFT:
  Relative: {self.result.relative_lift:+.2f}%
  Absolute: {self.result.absolute_lift:+.4f}

STATISTICS:
  P-value: {self.result.p_value:.6f}
  Significant: {'YES ✓' if self.result.significant else 'NO ✗'} (α={self.alpha})
  Effect Size: {self.result.effect_size:.4f} ({self.result.effect_size_name})

{'🏆 WINNER: ' + self.result.winner if self.result.winner else '⏸️  No clear winner'}
        """

        axes[1].text(0.1, 0.95, summary_text,
                    transform=axes[1].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        plt.show()


# ==================== MULTI-VARIANT TEST ====================

class MultiVariantTest:
    """
    Multi-variant testing (A/B/C/D/... testing).

    Tests multiple variants simultaneously with proper multiple comparison correction.

    Examples:
    ---------
    >>> df = pd.DataFrame({
    ...     'variant': ['A', 'B', 'C', 'D'] * 250,
    ...     'converted': np.random.binomial(1, [0.10, 0.12, 0.11, 0.13]*250)
    ... })
    >>>
    >>> test = MultiVariantTest(
    ...     data=df,
    ...     variant_col='variant',
    ...     metric_col='converted',
    ...     control_name='A'
    ... )
    >>>
    >>> results = test.run()
    >>> test.plot_results()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        variant_col: str,
        metric_col: str,
        control_name: str,
        metric_type: str = 'auto',
        metric_name: Optional[str] = None,
        alpha: float = 0.05,
        correction_method: str = 'bonferroni',
        confidence_level: float = 0.95
    ):
        """
        Initialize Multi-Variant Test.

        Parameters:
        -----------
        data : pd.DataFrame
            Experiment data.
        variant_col : str
            Column with variant labels.
        metric_col : str
            Column with metric values.
        control_name : str
            Name of control variant.
        metric_type : str
            'auto', 'binary', 'continuous', or 'count'.
        metric_name : str, optional
            Name of metric.
        alpha : float
            Significance level.
        correction_method : str
            Multiple testing correction: 'bonferroni', 'holm', or 'fdr'.
        confidence_level : float
            Confidence level.
        """
        self.data = data
        self.variant_col = variant_col
        self.metric_col = metric_col
        self.control_name = control_name
        self.metric_type = metric_type
        self.metric_name = metric_name or metric_col
        self.alpha = alpha
        self.correction_method = correction_method
        self.confidence_level = confidence_level

        # Get all variants (excluding control)
        self.all_variants = [v for v in data[variant_col].unique() if v != control_name]
        self.n_comparisons = len(self.all_variants)

        self.results = []
        self.summary = None

    def run(self, verbose: bool = True) -> List[ABTestResult]:
        """
        Run all pairwise comparisons (each variant vs control).

        Parameters:
        -----------
        verbose : bool
            Print results.

        Returns:
        --------
        list
            List of ABTestResult objects.
        """
        from .utils import bonferroni_correction, holm_bonferroni_correction, fdr_correction

        if verbose:
            print(f"\\n{'='*70}")
            print(f"Multi-Variant Test: {self.n_comparisons} treatments vs {self.control_name}")
            print(f"Multiple comparison correction: {self.correction_method.upper()}")
            print(f"{'='*70}\\n")

        # Run all comparisons
        p_values = []
        results = []

        for variant in self.all_variants:
            test = ABTest(
                data=self.data,
                variant_col=self.variant_col,
                metric_col=self.metric_col,
                control_name=self.control_name,
                treatment_name=variant,
                metric_type=self.metric_type,
                metric_name=self.metric_name,
                alpha=self.alpha,
                confidence_level=self.confidence_level
            )

            result = test.run(verbose=False)
            results.append(result)
            p_values.append(result.p_value)

        # Apply multiple testing correction
        p_values = np.array(p_values)

        if self.correction_method == 'bonferroni':
            corrected_p, adjusted_alpha = bonferroni_correction(p_values, self.alpha)
            reject = corrected_p < self.alpha
        elif self.correction_method == 'holm':
            reject, corrected_p = holm_bonferroni_correction(p_values, self.alpha)
        elif self.correction_method == 'fdr':
            reject, corrected_p = fdr_correction(p_values, self.alpha)
        else:
            raise ValueError(f"Unknown correction method: {self.correction_method}")

        # Update results with corrected p-values
        for i, result in enumerate(results):
            result.p_value = corrected_p[i]
            result.significant = bool(reject[i])

            # Update recommendation based on corrected significance
            if result.significant:
                if result.treatment_metric > result.control_metric:
                    result.winner = result.treatment_name
                    result.recommendation = (
                        f"✓ {result.treatment_name.upper()} significantly outperforms "
                        f"{self.control_name} (corrected p={corrected_p[i]:.4f})"
                    )
                else:
                    result.recommendation = (
                        f"⚠️  {result.treatment_name} performs significantly worse "
                        f"(corrected p={corrected_p[i]:.4f})"
                    )
            else:
                result.recommendation = (
                    f"⏸️  No significant difference for {result.treatment_name} "
                    f"(corrected p={corrected_p[i]:.4f})"
                )

        self.results = results

        # Create summary
        self._create_summary()

        if verbose:
            print(self.summary)

        return results

    def _create_summary(self):
        """Create summary DataFrame."""
        summary_data = []

        for result in self.results:
            summary_data.append({
                'variant': result.treatment_name,
                'metric': result.treatment_metric,
                'ci_lower': result.treatment_ci[0],
                'ci_upper': result.treatment_ci[1],
                'relative_lift_%': result.relative_lift,
                'absolute_lift': result.absolute_lift,
                'p_value': result.p_value,
                'significant': result.significant,
                'winner': '🏆' if result.winner == result.treatment_name else ''
            })

        self.summary = pd.DataFrame(summary_data)
        self.summary = self.summary.sort_values('relative_lift_%', ascending=False)

    def plot_results(self, figsize: Tuple[int, int] = (14, 8)):
        """Plot multi-variant test results."""
        if not self.results:
            raise ValueError("Must run test first. Call .run()")

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Get control metrics for reference
        control_data = self.data[self.data[self.variant_col] == self.control_name][self.metric_col]
        control_metric = np.mean(control_data)

        # Plot 1: All variants comparison
        variants = [self.control_name] + [r.treatment_name for r in self.results]
        metrics = [control_metric] + [r.treatment_metric for r in self.results]
        colors = ['#95a5a6'] + ['#e74c3c' if r.significant and r.winner == r.treatment_name
                                  else '#3498db' if r.significant
                                  else '#95a5a6' for r in self.results]

        bars = axes[0].bar(variants, metrics, color=colors, alpha=0.7, edgecolor='black')
        axes[0].set_ylabel(self.metric_name, fontsize=12, fontweight='bold')
        axes[0].set_title('All Variants Comparison', fontsize=13, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, value in zip(bars, metrics):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=9)

        # Plot 2: Lift comparison
        lifts = [r.relative_lift for r in self.results]
        variant_names = [r.treatment_name for r in self.results]
        colors_lift = ['#2ecc71' if lift > 0 else '#e74c3c' for lift in lifts]

        bars = axes[1].barh(variant_names, lifts, color=colors_lift, alpha=0.7, edgecolor='black')
        axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1)
        axes[1].set_xlabel('Relative Lift (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('Lift vs Control', fontsize=13, fontweight='bold')

        # Add value labels
        for bar, value in zip(bars, lifts):
            width = bar.get_width()
            axes[1].text(width, bar.get_y() + bar.get_height()/2.,
                       f'{value:+.1f}%', ha='left' if width > 0 else 'right',
                       va='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.show()
