"""
Statistical Visualizations Module

Publication-ready visualizations for A/B testing and statistical analysis.

Functions included:
- A/B test result plots
- Confidence interval visualizations
- Sequential testing plots
- Funnel analysis
- Power curves
- Sample size curves
- Statistical dashboards
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple
import warnings

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# Set default style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


# ==================== A/B TEST VISUALIZATIONS ====================

def plot_ab_test_results(
    control_metric: float,
    treatment_metric: float,
    control_ci: Tuple[float, float],
    treatment_ci: Tuple[float, float],
    control_name: str = 'Control',
    treatment_name: str = 'Treatment',
    metric_name: str = 'Metric',
    p_value: Optional[float] = None,
    significant: Optional[bool] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    interactive: bool = False
):
    """
    Create comprehensive A/B test results visualization.

    Shows both variants with confidence intervals, highlighting the winner.

    Parameters:
    -----------
    control_metric : float
        Control group metric value.
    treatment_metric : float
        Treatment group metric value.
    control_ci : tuple
        Confidence interval for control (lower, upper).
    treatment_ci : tuple
        Confidence interval for treatment (lower, upper).
    control_name : str
        Name of control variant.
    treatment_name : str
        Name of treatment variant.
    metric_name : str
        Name of the metric.
    p_value : float, optional
        P-value from statistical test.
    significant : bool, optional
        Whether result is statistically significant.
    figsize : tuple
        Figure size.
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save figure.
    interactive : bool
        Create interactive Plotly version.

    Returns:
    --------
    matplotlib.figure.Figure or plotly.graph_objs.Figure

    Examples:
    ---------
    >>> from MAna.stata.visualizations import plot_ab_test_results
    >>>
    >>> plot_ab_test_results(
    ...     control_metric=0.095,
    ...     treatment_metric=0.128,
    ...     control_ci=(0.085, 0.105),
    ...     treatment_ci=(0.118, 0.138),
    ...     control_name='Version A',
    ...     treatment_name='Version B',
    ...     metric_name='Conversion Rate',
    ...     p_value=0.0023,
    ...     significant=True
    ... )
    """
    if interactive and not PLOTLY_AVAILABLE:
        warnings.warn("Plotly not installed. Falling back to matplotlib.")
        interactive = False

    variants = [control_name, treatment_name]
    metrics = [control_metric, treatment_metric]

    if interactive:
        # Plotly version
        fig = go.Figure()

        # Determine colors
        if significant is not None:
            winner_idx = 1 if treatment_metric > control_metric else 0
            colors = ['#95a5a6', '#95a5a6']
            if significant:
                colors[winner_idx] = '#2ecc71'  # Green for winner
                colors[1 - winner_idx] = '#e74c3c'  # Red for loser
        else:
            colors = ['#3498db', '#e74c3c']

        # Add bars with error bars
        fig.add_trace(go.Bar(
            x=variants,
            y=metrics,
            marker_color=colors,
            error_y=dict(
                type='data',
                symmetric=False,
                array=[treatment_ci[1] - treatment_metric, control_ci[1] - control_metric],
                arrayminus=[treatment_metric - treatment_ci[0], control_metric - control_ci[0]],
                color='black',
                thickness=2,
                width=8
            ),
            text=[f'{m:.4f}' for m in metrics],
            textposition='outside',
            textfont=dict(size=14, color='black'),
            showlegend=False
        ))

        # Add title with p-value if provided
        if title:
            plot_title = title
        else:
            plot_title = f'{metric_name} Comparison'
            if p_value is not None:
                sig_text = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
                plot_title += f'<br><sub>p-value: {p_value:.6f} {sig_text}</sub>'

        fig.update_layout(
            title=dict(text=plot_title, x=0.5, xanchor='center'),
            xaxis_title='Variant',
            yaxis_title=metric_name,
            width=700,
            height=500,
            template='plotly_white',
            font=dict(size=12)
        )

        if save_path:
            fig.write_html(save_path)

        fig.show()
        return fig

    else:
        # Matplotlib version
        fig, ax = plt.subplots(figsize=figsize)

        # Determine colors
        if significant is not None:
            winner_idx = 1 if treatment_metric > control_metric else 0
            colors = ['#95a5a6', '#95a5a6']
            if significant:
                colors[winner_idx] = '#2ecc71'  # Green for winner
                colors[1 - winner_idx] = '#e74c3c'  # Red for loser
        else:
            colors = ['#3498db', '#e74c3c']

        # Create bars
        bars = ax.bar(variants, metrics, color=colors, alpha=0.7,
                      edgecolor='black', linewidth=2, width=0.6)

        # Add error bars (confidence intervals)
        yerr = [
            [control_metric - control_ci[0], treatment_metric - treatment_ci[0]],
            [control_ci[1] - control_metric, treatment_ci[1] - treatment_metric]
        ]
        ax.errorbar(variants, metrics, yerr=yerr, fmt='none',
                   ecolor='black', capsize=12, capthick=2, elinewidth=2)

        # Add value labels on bars
        for i, (bar, value, ci) in enumerate(zip(bars, metrics, [control_ci, treatment_ci])):
            height = bar.get_height()
            # Main value
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}',
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
            # CI below
            ax.text(bar.get_x() + bar.get_width()/2., ci[0],
                   f'[{ci[0]:.4f}, {ci[1]:.4f}]',
                   ha='center', va='top', fontsize=9, style='italic', color='gray')

        # Styling
        ax.set_ylabel(metric_name, fontsize=13, fontweight='bold')
        ax.set_xlabel('Variant', fontsize=13, fontweight='bold')

        # Title with p-value
        if title:
            plot_title = title
        else:
            plot_title = f'{metric_name} Comparison'
            if p_value is not None:
                sig_text = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
                plot_title += f'\n(p-value: {p_value:.6f} {sig_text})'

        ax.set_title(plot_title, fontsize=15, fontweight='bold', pad=20)
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylim(0, max(metrics + [control_ci[1], treatment_ci[1]]) * 1.2)

        # Add significance annotation if provided
        if significant is not None and significant:
            # Draw significance bar
            y_max = max(control_ci[1], treatment_ci[1])
            h = y_max * 1.05
            ax.plot([0, 0, 1, 1], [h, h*1.02, h*1.02, h], lw=1.5, c='black')
            ax.text(0.5, h*1.03, '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*',
                   ha='center', va='bottom', fontsize=16, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
        return fig


def plot_confidence_intervals(
    metrics: List[float],
    ci_lower: List[float],
    ci_upper: List[float],
    labels: List[str],
    metric_name: str = 'Metric',
    reference_line: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = 'Confidence Intervals Comparison',
    save_path: Optional[str] = None,
    interactive: bool = False
):
    """
    Create forest plot showing confidence intervals for multiple groups.

    Useful for comparing multiple variants or showing results over time.

    Parameters:
    -----------
    metrics : list
        Point estimates for each group.
    ci_lower : list
        Lower bounds of confidence intervals.
    ci_upper : list
        Upper bounds of confidence intervals.
    labels : list
        Labels for each group.
    metric_name : str
        Name of the metric.
    reference_line : float, optional
        Value for reference line (e.g., control mean).
    figsize : tuple
        Figure size.
    title : str
        Plot title.
    save_path : str, optional
        Path to save figure.
    interactive : bool
        Create interactive Plotly version.

    Returns:
    --------
    matplotlib.figure.Figure or plotly.graph_objs.Figure

    Examples:
    ---------
    >>> # Compare 5 different variants
    >>> metrics = [0.095, 0.102, 0.110, 0.098, 0.115]
    >>> ci_lower = [0.085, 0.092, 0.100, 0.088, 0.105]
    >>> ci_upper = [0.105, 0.112, 0.120, 0.108, 0.125]
    >>> labels = ['Control', 'Variant A', 'Variant B', 'Variant C', 'Variant D']
    >>>
    >>> plot_confidence_intervals(
    ...     metrics, ci_lower, ci_upper, labels,
    ...     metric_name='Conversion Rate',
    ...     reference_line=0.095  # Control value
    ... )
    """
    if interactive and not PLOTLY_AVAILABLE:
        warnings.warn("Plotly not installed. Falling back to matplotlib.")
        interactive = False

    n = len(metrics)

    if interactive:
        # Plotly version
        fig = go.Figure()

        # Add confidence intervals as error bars
        for i, (metric, lower, upper, label) in enumerate(zip(metrics, ci_lower, ci_upper, labels)):
            # Determine color
            if reference_line is not None:
                if lower > reference_line:
                    color = '#2ecc71'  # Green - significantly higher
                elif upper < reference_line:
                    color = '#e74c3c'  # Red - significantly lower
                else:
                    color = '#95a5a6'  # Gray - overlaps reference
            else:
                color = '#3498db'

            fig.add_trace(go.Scatter(
                x=[metric],
                y=[label],
                mode='markers',
                marker=dict(size=12, color=color),
                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=[upper - metric],
                    arrayminus=[metric - lower],
                    color=color,
                    thickness=3,
                    width=8
                ),
                showlegend=False,
                hovertemplate=f'{label}<br>{metric_name}: %{{x:.4f}}<br>CI: [{lower:.4f}, {upper:.4f}]<extra></extra>'
            ))

        # Add reference line if provided
        if reference_line is not None:
            fig.add_vline(x=reference_line, line_dash="dash", line_color="black",
                         annotation_text="Reference", annotation_position="top")

        fig.update_layout(
            title=title,
            xaxis_title=metric_name,
            yaxis_title='',
            width=800,
            height=max(400, n * 60),
            template='plotly_white',
            font=dict(size=12)
        )

        if save_path:
            fig.write_html(save_path)

        fig.show()
        return fig

    else:
        # Matplotlib version (forest plot)
        fig, ax = plt.subplots(figsize=figsize)

        y_positions = np.arange(n)

        # Plot confidence intervals
        for i, (metric, lower, upper) in enumerate(zip(metrics, ci_lower, ci_upper)):
            # Determine color
            if reference_line is not None:
                if lower > reference_line:
                    color = '#2ecc71'  # Green
                elif upper < reference_line:
                    color = '#e74c3c'  # Red
                else:
                    color = '#95a5a6'  # Gray
            else:
                color = '#3498db'

            # Plot CI line
            ax.plot([lower, upper], [i, i], color=color, linewidth=3, alpha=0.6)

            # Plot point estimate
            ax.plot(metric, i, 'o', color=color, markersize=10,
                   markeredgecolor='black', markeredgewidth=1.5)

            # Add value label
            ax.text(upper + 0.01 * (max(ci_upper) - min(ci_lower)), i,
                   f'{metric:.4f}', va='center', fontsize=10, fontweight='bold')

        # Add reference line
        if reference_line is not None:
            ax.axvline(reference_line, color='black', linestyle='--',
                      linewidth=2, alpha=0.7, label='Reference')
            ax.legend(fontsize=11)

        # Styling
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel(metric_name, fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
        ax.grid(alpha=0.3, axis='x')
        ax.invert_yaxis()  # Highest at top

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
        return fig


def plot_lift_analysis(
    control_metric: float,
    treatment_metric: float,
    control_ci: Tuple[float, float],
    treatment_ci: Tuple[float, float],
    metric_name: str = 'Conversion Rate',
    control_name: str = 'Control',
    treatment_name: str = 'Treatment',
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
):
    """
    Create comprehensive lift analysis visualization.

    Shows absolute lift, relative lift, and statistical significance.

    Parameters:
    -----------
    control_metric : float
        Control metric value.
    treatment_metric : float
        Treatment metric value.
    control_ci : tuple
        Control confidence interval.
    treatment_ci : tuple
        Treatment confidence interval.
    metric_name : str
        Name of metric.
    control_name : str
        Control variant name.
    treatment_name : str
        Treatment variant name.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.

    Returns:
    --------
    matplotlib.figure.Figure

    Examples:
    ---------
    >>> plot_lift_analysis(
    ...     control_metric=0.095,
    ...     treatment_metric=0.128,
    ...     control_ci=(0.085, 0.105),
    ...     treatment_ci=(0.118, 0.138),
    ...     metric_name='Conversion Rate'
    ... )
    """
    from .utils import relative_lift, absolute_lift

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Calculate lifts
    abs_lift = absolute_lift(control_metric, treatment_metric)
    rel_lift = relative_lift(control_metric, treatment_metric, as_percentage=True)

    # Plot 1: Absolute values comparison
    variants = [control_name, treatment_name]
    metrics = [control_metric, treatment_metric]
    colors = ['#3498db', '#2ecc71' if treatment_metric > control_metric else '#e74c3c']

    bars = axes[0].bar(variants, metrics, color=colors, alpha=0.7,
                      edgecolor='black', linewidth=2)

    # Error bars
    yerr = [
        [control_metric - control_ci[0], treatment_metric - treatment_ci[0]],
        [control_ci[1] - control_metric, treatment_ci[1] - treatment_metric]
    ]
    axes[0].errorbar(variants, metrics, yerr=yerr, fmt='none',
                    ecolor='black', capsize=10, capthick=2)

    # Value labels
    for bar, value in zip(bars, metrics):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}', ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

    axes[0].set_ylabel(metric_name, fontsize=12, fontweight='bold')
    axes[0].set_title('Absolute Values', fontsize=13, fontweight='bold')
    axes[0].grid(alpha=0.3, axis='y')

    # Plot 2: Absolute lift
    lift_color = '#2ecc71' if abs_lift > 0 else '#e74c3c'
    bar = axes[1].bar(['Absolute\nLift'], [abs_lift], color=lift_color,
                     alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1].text(0, abs_lift, f'{abs_lift:+.4f}', ha='center',
                va='bottom' if abs_lift > 0 else 'top',
                fontsize=14, fontweight='bold')
    axes[1].set_ylabel(f'Change in {metric_name}', fontsize=12, fontweight='bold')
    axes[1].set_title('Absolute Lift', fontsize=13, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')

    # Plot 3: Relative lift
    bar = axes[2].bar(['Relative\nLift'], [rel_lift], color=lift_color,
                     alpha=0.7, edgecolor='black', linewidth=2)
    axes[2].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[2].text(0, rel_lift, f'{rel_lift:+.1f}%', ha='center',
                va='bottom' if rel_lift > 0 else 'top',
                fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Percentage Change', fontsize=12, fontweight='bold')
    axes[2].set_title('Relative Lift', fontsize=13, fontweight='bold')
    axes[2].grid(alpha=0.3, axis='y')

    fig.suptitle(f'Lift Analysis: {treatment_name} vs {control_name}',
                fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def plot_sequential_results(
    dates: List[str],
    control_metrics: List[float],
    treatment_metrics: List[float],
    control_cis: List[Tuple[float, float]],
    treatment_cis: List[Tuple[float, float]],
    p_values: List[float],
    alpha: float = 0.05,
    metric_name: str = 'Conversion Rate',
    control_name: str = 'Control',
    treatment_name: str = 'Treatment',
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
):
    """
    Visualize sequential A/B test results over time.

    Shows how metrics and significance evolve as more data is collected.

    Parameters:
    -----------
    dates : list
        List of dates/time points.
    control_metrics : list
        Control metric values over time.
    treatment_metrics : list
        Treatment metric values over time.
    control_cis : list
        Control confidence intervals over time.
    treatment_cis : list
        Treatment confidence intervals over time.
    p_values : list
        P-values over time.
    alpha : float
        Significance level.
    metric_name : str
        Name of metric.
    control_name : str
        Control variant name.
    treatment_name : str
        Treatment variant name.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.

    Returns:
    --------
    matplotlib.figure.Figure

    Examples:
    ---------
    >>> dates = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5']
    >>> control_metrics = [0.090, 0.092, 0.094, 0.095, 0.095]
    >>> treatment_metrics = [0.100, 0.105, 0.110, 0.118, 0.125]
    >>> control_cis = [(0.08, 0.10), (0.084, 0.10), (0.088, 0.10), (0.089, 0.101), (0.090, 0.100)]
    >>> treatment_cis = [(0.09, 0.11), (0.095, 0.115), (0.10, 0.12), (0.108, 0.128), (0.115, 0.135)]
    >>> p_values = [0.15, 0.08, 0.04, 0.012, 0.003]
    >>>
    >>> plot_sequential_results(
    ...     dates, control_metrics, treatment_metrics,
    ...     control_cis, treatment_cis, p_values
    ... )
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    x = np.arange(len(dates))

    # Plot 1: Metric values over time
    axes[0].plot(x, control_metrics, 'o-', label=control_name,
                color='#3498db', linewidth=2, markersize=8)
    axes[0].plot(x, treatment_metrics, 's-', label=treatment_name,
                color='#e74c3c', linewidth=2, markersize=8)

    # Add confidence bands
    control_lower = [ci[0] for ci in control_cis]
    control_upper = [ci[1] for ci in control_cis]
    treatment_lower = [ci[0] for ci in treatment_cis]
    treatment_upper = [ci[1] for ci in treatment_cis]

    axes[0].fill_between(x, control_lower, control_upper,
                         alpha=0.2, color='#3498db')
    axes[0].fill_between(x, treatment_lower, treatment_upper,
                         alpha=0.2, color='#e74c3c')

    axes[0].set_ylabel(metric_name, fontsize=12, fontweight='bold')
    axes[0].set_title('Metric Evolution Over Time', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11, loc='best')
    axes[0].grid(alpha=0.3)

    # Plot 2: Relative lift over time
    from .utils import relative_lift
    lifts = [relative_lift(c, t) for c, t in zip(control_metrics, treatment_metrics)]

    colors = ['#2ecc71' if lift > 0 else '#e74c3c' for lift in lifts]
    axes[1].bar(x, lifts, color=colors, alpha=0.7, edgecolor='black')
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_ylabel('Relative Lift (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Lift Over Time', fontsize=13, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')

    # Add value labels
    for i, lift in enumerate(lifts):
        axes[1].text(i, lift, f'{lift:+.1f}%', ha='center',
                    va='bottom' if lift > 0 else 'top', fontsize=9)

    # Plot 3: P-value over time
    axes[2].plot(x, p_values, 'o-', color='#9b59b6', linewidth=2, markersize=8)
    axes[2].axhline(alpha, color='red', linestyle='--', linewidth=2,
                   label=f'α = {alpha}')

    # Shade significant region
    axes[2].fill_between(x, 0, alpha, alpha=0.2, color='green',
                        label='Significant region')

    axes[2].set_ylabel('P-value', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time Period', fontsize=12, fontweight='bold')
    axes[2].set_title('Statistical Significance Over Time', fontsize=13, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(dates, rotation=45, ha='right')
    axes[2].legend(fontsize=11)
    axes[2].grid(alpha=0.3)
    axes[2].set_yscale('log')

    # Add annotation for when it became significant
    significant_idx = next((i for i, p in enumerate(p_values) if p < alpha), None)
    if significant_idx is not None:
        axes[2].annotate(
            f'Significant from {dates[significant_idx]}',
            xy=(significant_idx, p_values[significant_idx]),
            xytext=(significant_idx, p_values[significant_idx] * 10),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, fontweight='bold', color='red'
        )

    fig.suptitle('Sequential A/B Test Analysis', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def plot_conversion_funnel(
    control_funnel: Dict[str, int],
    treatment_funnel: Dict[str, int],
    control_name: str = 'Control',
    treatment_name: str = 'Treatment',
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
):
    """
    Compare conversion funnels between two variants.

    Shows step-by-step comparison of how users progress through the funnel.

    Parameters:
    -----------
    control_funnel : dict
        Control funnel data {step_name: count}.
    treatment_funnel : dict
        Treatment funnel data {step_name: count}.
    control_name : str
        Control variant name.
    treatment_name : str
        Treatment variant name.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.

    Returns:
    --------
    matplotlib.figure.Figure

    Examples:
    ---------
    >>> control_funnel = {
    ...     'Visitors': 10000,
    ...     'Product View': 3000,
    ...     'Add to Cart': 1200,
    ...     'Checkout': 600,
    ...     'Purchase': 480
    ... }
    >>>
    >>> treatment_funnel = {
    ...     'Visitors': 10000,
    ...     'Product View': 3200,
    ...     'Add to Cart': 1400,
    ...     'Checkout': 750,
    ...     'Purchase': 650
    ... }
    >>>
    >>> plot_conversion_funnel(control_funnel, treatment_funnel)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    steps = list(control_funnel.keys())
    control_counts = list(control_funnel.values())
    treatment_counts = list(treatment_funnel.values())

    # Calculate conversion rates for each step
    control_rates = [control_counts[0]] + [
        (control_counts[i] / control_counts[i-1] * 100) if control_counts[i-1] > 0 else 0
        for i in range(1, len(control_counts))
    ]
    treatment_rates = [treatment_counts[0]] + [
        (treatment_counts[i] / treatment_counts[i-1] * 100) if treatment_counts[i-1] > 0 else 0
        for i in range(1, len(treatment_counts))
    ]

    # Plot 1: Absolute numbers (funnel chart)
    x = np.arange(len(steps))
    width = 0.35

    bars1 = axes[0].barh(x - width/2, control_counts, width,
                        label=control_name, color='#3498db', alpha=0.7,
                        edgecolor='black')
    bars2 = axes[0].barh(x + width/2, treatment_counts, width,
                        label=treatment_name, color='#e74c3c', alpha=0.7,
                        edgecolor='black')

    axes[0].set_yticks(x)
    axes[0].set_yticklabels(steps, fontsize=11)
    axes[0].set_xlabel('Number of Users', fontsize=12, fontweight='bold')
    axes[0].set_title('Funnel - Absolute Numbers', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3, axis='x')
    axes[0].invert_yaxis()

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            width_val = bar.get_width()
            axes[0].text(width_val, bar.get_y() + bar.get_height()/2.,
                        f'{int(width_val):,}', ha='left', va='center',
                        fontsize=9, fontweight='bold')

    # Plot 2: Conversion rates
    axes[1].plot(control_rates, x, 'o-', label=control_name,
                color='#3498db', linewidth=2.5, markersize=10)
    axes[1].plot(treatment_rates, x, 's-', label=treatment_name,
                color='#e74c3c', linewidth=2.5, markersize=10)

    axes[1].set_yticks(x)
    axes[1].set_yticklabels(steps, fontsize=11)
    axes[1].set_xlabel('Step Conversion Rate (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Funnel - Step Conversion Rates', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)
    axes[1].invert_yaxis()

    # Add value labels for rates
    for i, (c_rate, t_rate) in enumerate(zip(control_rates, treatment_rates)):
        if i == 0:  # Skip first step (it's total visitors)
            continue
        axes[1].text(c_rate, i, f'{c_rate:.1f}%', ha='right', va='bottom',
                    fontsize=9, color='#3498db', fontweight='bold')
        axes[1].text(t_rate, i, f'{t_rate:.1f}%', ha='left', va='top',
                    fontsize=9, color='#e74c3c', fontweight='bold')

    fig.suptitle('Conversion Funnel Comparison', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def plot_sample_size_curve(
    baseline_rate: float,
    mde_range: np.ndarray,
    alpha: float = 0.05,
    power: float = 0.80,
    figsize: Tuple[int, int] = (10, 6),
    title: str = 'Sample Size Requirements',
    save_path: Optional[str] = None
):
    """
    Plot required sample size as a function of minimum detectable effect.

    Helps determine how long to run an A/B test.

    Parameters:
    -----------
    baseline_rate : float
        Baseline conversion rate (e.g., 0.10 for 10%).
    mde_range : np.ndarray
        Range of minimum detectable effects to plot (as proportions).
    alpha : float
        Significance level.
    power : float
        Desired statistical power.
    figsize : tuple
        Figure size.
    title : str
        Plot title.
    save_path : str, optional
        Path to save figure.

    Returns:
    --------
    matplotlib.figure.Figure

    Examples:
    ---------
    >>> import numpy as np
    >>>
    >>> plot_sample_size_curve(
    ...     baseline_rate=0.10,
    ...     mde_range=np.arange(0.05, 0.30, 0.01),
    ...     alpha=0.05,
    ...     power=0.80
    ... )
    """
    from scipy import stats

    # Calculate required sample size for each MDE
    sample_sizes = []

    for mde in mde_range:
        p1 = baseline_rate
        p2 = baseline_rate * (1 + mde)

        # Effect size (Cohen's h)
        effect_size = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))

        # Z-scores
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        # Sample size calculation (per group)
        n = ((z_alpha + z_beta) ** 2) / (effect_size ** 2)
        sample_sizes.append(int(np.ceil(n)))

    sample_sizes = np.array(sample_sizes)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot curve
    ax.plot(mde_range * 100, sample_sizes, linewidth=3, color='#3498db')
    ax.fill_between(mde_range * 100, 0, sample_sizes, alpha=0.3, color='#3498db')

    # Add reference lines for common MDEs
    common_mdes = [0.05, 0.10, 0.15, 0.20]
    for mde in common_mdes:
        if mde in mde_range:
            idx = np.argmin(np.abs(mde_range - mde))
            n_required = sample_sizes[idx]
            ax.axvline(mde * 100, color='red', linestyle='--', alpha=0.5)
            ax.text(mde * 100, ax.get_ylim()[1] * 0.9,
                   f'{mde*100:.0f}%\n{n_required:,}',
                   ha='center', fontsize=9, bbox=dict(boxstyle='round',
                   facecolor='yellow', alpha=0.5))

    ax.set_xlabel('Minimum Detectable Effect (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Required Sample Size (per variant)', fontsize=12, fontweight='bold')
    ax.set_title(f'{title}\nBaseline: {baseline_rate*100:.1f}%, α={alpha}, Power={power}',
                fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

    # Format y-axis with commas
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def plot_power_curve(
    baseline_rate: float,
    sample_size: int,
    effect_size_range: np.ndarray,
    alpha: float = 0.05,
    figsize: Tuple[int, int] = (10, 6),
    title: str = 'Statistical Power Curve',
    save_path: Optional[str] = None
):
    """
    Plot statistical power as a function of effect size.

    Shows probability of detecting an effect for different effect sizes.

    Parameters:
    -----------
    baseline_rate : float
        Baseline conversion rate.
    sample_size : int
        Sample size per variant.
    effect_size_range : np.ndarray
        Range of effect sizes (as Cohen's h).
    alpha : float
        Significance level.
    figsize : tuple
        Figure size.
    title : str
        Plot title.
    save_path : str, optional
        Path to save figure.

    Returns:
    --------
    matplotlib.figure.Figure

    Examples:
    ---------
    >>> import numpy as np
    >>>
    >>> plot_power_curve(
    ...     baseline_rate=0.10,
    ...     sample_size=5000,
    ...     effect_size_range=np.arange(0, 0.5, 0.01),
    ...     alpha=0.05
    ... )
    """
    from scipy import stats

    # Calculate power for each effect size
    powers = []

    z_alpha = stats.norm.ppf(1 - alpha / 2)

    for h in effect_size_range:
        # Non-centrality parameter
        ncp = h * np.sqrt(sample_size / 2)

        # Power calculation
        power = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)
        powers.append(power)

    powers = np.array(powers)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot power curve
    ax.plot(effect_size_range, powers, linewidth=3, color='#e74c3c')
    ax.fill_between(effect_size_range, 0, powers, alpha=0.3, color='#e74c3c')

    # Add reference lines
    ax.axhline(0.80, color='green', linestyle='--', linewidth=2,
              label='80% Power (Typical Target)', alpha=0.7)
    ax.axhline(0.90, color='blue', linestyle='--', linewidth=2,
              label='90% Power (High)', alpha=0.7)

    # Find effect size for 80% power
    idx_80 = np.argmin(np.abs(powers - 0.80))
    effect_80 = effect_size_range[idx_80]
    ax.axvline(effect_80, color='green', linestyle=':', alpha=0.5)
    ax.text(effect_80, 0.4, f'h = {effect_80:.3f}\n(80% power)',
           ha='center', fontsize=10, bbox=dict(boxstyle='round',
           facecolor='lightgreen', alpha=0.5))

    ax.set_xlabel("Effect Size (Cohen's h)", fontsize=12, fontweight='bold')
    ax.set_ylabel('Statistical Power', fontsize=12, fontweight='bold')
    ax.set_title(f'{title}\nSample Size: {sample_size:,} per variant, α={alpha}',
                fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def plot_ab_dashboard(
    result,
    include_distribution: bool = True,
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None
):
    """
    Create comprehensive one-page A/B test dashboard.

    Shows all key metrics, visualizations, and recommendations in one view.

    Parameters:
    -----------
    result : ABTestResult
        Result object from ABTest.run().
    include_distribution : bool
        Include distribution plots (requires raw data).
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.

    Returns:
    --------
    matplotlib.figure.Figure

    Examples:
    ---------
    >>> from MAna.stata import ABTest
    >>>
    >>> test = ABTest(data=df, variant_col='variant', metric_col='converted')
    >>> result = test.run()
    >>> plot_ab_dashboard(result)
    """
    # Create figure with GridSpec for flexible layout
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Colors
    winner_color = '#2ecc71'
    loser_color = '#e74c3c'
    neutral_color = '#95a5a6'

    control_color = winner_color if result.winner == result.control_name else loser_color if result.significant else neutral_color
    treatment_color = winner_color if result.winner == result.treatment_name else loser_color if result.significant else neutral_color

    # ===== Plot 1: Main comparison (top left) =====
    ax1 = fig.add_subplot(gs[0, :2])

    variants = [result.control_name, result.treatment_name]
    metrics = [result.control_metric, result.treatment_metric]
    colors = [control_color, treatment_color]

    bars = ax1.bar(variants, metrics, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=2, width=0.5)

    # Error bars
    yerr = [
        [result.control_metric - result.control_ci[0],
         result.treatment_metric - result.treatment_ci[0]],
        [result.control_ci[1] - result.control_metric,
         result.treatment_ci[1] - result.treatment_metric]
    ]
    ax1.errorbar(variants, metrics, yerr=yerr, fmt='none',
                ecolor='black', capsize=15, capthick=3, elinewidth=2.5)

    # Value labels
    for bar, value, ci in zip(bars, metrics, [result.control_ci, result.treatment_ci]):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom',
                fontsize=13, fontweight='bold')
        ax1.text(bar.get_x() + bar.get_width()/2., ci[0] * 0.95,
                f'[{ci[0]:.4f}, {ci[1]:.4f}]', ha='center', va='top',
                fontsize=9, style='italic')

    ax1.set_ylabel(result.metric_name, fontsize=12, fontweight='bold')
    ax1.set_title('Main Comparison', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')

    # ===== Plot 2: Lift bars (top right) =====
    ax2 = fig.add_subplot(gs[0, 2])

    lifts = [result.absolute_lift, result.relative_lift]
    lift_labels = ['Absolute', 'Relative\n(%)']
    lift_colors = [winner_color if result.relative_lift > 0 else loser_color] * 2

    bars = ax2.barh(lift_labels, [result.absolute_lift, result.relative_lift],
                   color=lift_colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.axvline(0, color='black', linestyle='--', linewidth=1.5)

    for i, (bar, value) in enumerate(zip(bars, lifts)):
        width = bar.get_width()
        label = f'{value:+.4f}' if i == 0 else f'{value:+.1f}%'
        ax2.text(width, bar.get_y() + bar.get_height()/2., label,
                ha='left' if width > 0 else 'right', va='center',
                fontsize=11, fontweight='bold')

    ax2.set_xlabel('Lift', fontsize=11, fontweight='bold')
    ax2.set_title('Lift Analysis', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, axis='x')

    # ===== Plot 3: Key metrics summary (middle left) =====
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')

    # Create summary table
    summary_text = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    A/B TEST SUMMARY                                              ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

📊 EXPERIMENT DETAILS
   Metric: {result.metric_name} ({result.metric_type})
   Control: {result.control_name} (n = {result.sample_sizes.get('control', 'N/A'):,})
   Treatment: {result.treatment_name} (n = {result.sample_sizes.get('treatment', 'N/A'):,})
   Total Sample: {sum(result.sample_sizes.values()):,}

📈 RESULTS
   {result.control_name:.<30} {result.control_metric:.6f}  [{result.control_ci[0]:.6f}, {result.control_ci[1]:.6f}]
   {result.treatment_name:.<30} {result.treatment_metric:.6f}  [{result.treatment_ci[0]:.6f}, {result.treatment_ci[1]:.6f}]

   Absolute Lift:                    {result.absolute_lift:+.6f}
   Relative Lift:                    {result.relative_lift:+.2f}%

🔬 STATISTICAL INFERENCE
   P-value:                          {result.p_value:.8f} {'***' if result.p_value < 0.001 else '**' if result.p_value < 0.01 else '*' if result.p_value < 0.05 else 'ns'}
   Significant:                      {'✓ YES' if result.significant else '✗ NO'} (α = {result.alpha})
   Effect Size:                      {result.effect_size:.4f} ({result.effect_size_name})
   Confidence Level:                 {result.confidence_level*100:.0f}%
   {'Winner:                           🏆 ' + result.winner if result.winner else 'Result:                          ⏸️  Inconclusive'}

💡 RECOMMENDATION
   {result.recommendation}
    """

    ax3.text(0.5, 0.5, summary_text, transform=ax3.transAxes,
            fontsize=9, verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # ===== Plot 4: Effect size visualization (bottom left) =====
    ax4 = fig.add_subplot(gs[2, 0])

    from .utils import interpret_effect_size
    interpretation = interpret_effect_size(result.effect_size, result.effect_size_name.lower().replace("'", "").replace(" ", "_"))

    effect_categories = ['Negligible', 'Small', 'Medium', 'Large']
    # Determine which category
    category_idx = effect_categories.index(interpretation.capitalize())
    colors_effect = ['#95a5a6'] * 4
    colors_effect[category_idx] = '#2ecc71'

    bars = ax4.barh(effect_categories, [0.2, 0.5, 0.8, 1.5],
                   color=colors_effect, alpha=0.7, edgecolor='black')
    ax4.axvline(abs(result.effect_size), color='red', linestyle='--',
               linewidth=3, label=f'Your effect: {abs(result.effect_size):.3f}')

    ax4.set_xlabel('Effect Size', fontsize=11, fontweight='bold')
    ax4.set_title('Effect Size Interpretation', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3, axis='x')

    # ===== Plot 5: Sample size info (bottom middle) =====
    ax5 = fig.add_subplot(gs[2, 1])

    total_n = sum(result.sample_sizes.values())
    control_n = result.sample_sizes.get('control', 0)
    treatment_n = result.sample_sizes.get('treatment', 0)

    sizes = [control_n, treatment_n]
    labels_pie = [f'{result.control_name}\n{control_n:,}',
                  f'{result.treatment_name}\n{treatment_n:,}']
    colors_pie = ['#3498db', '#e74c3c']

    wedges, texts, autotexts = ax5.pie(sizes, labels=labels_pie, colors=colors_pie,
                                       autopct='%1.1f%%', startangle=90,
                                       textprops={'fontsize': 10, 'fontweight': 'bold'})

    ax5.set_title(f'Sample Distribution\nTotal: {total_n:,}',
                 fontsize=12, fontweight='bold')

    # ===== Plot 6: Significance indicator (bottom right) =====
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')

    # Create big visual indicator
    if result.significant:
        indicator_color = winner_color
        indicator_text = "✓ SIGNIFICANT"
        detail_text = f"p = {result.p_value:.6f}\n{result.winner} WINS!"
    else:
        indicator_color = neutral_color
        indicator_text = "⏸ INCONCLUSIVE"
        detail_text = f"p = {result.p_value:.6f}\nKeep testing"

    # Draw circle
    circle = plt.Circle((0.5, 0.6), 0.3, color=indicator_color, alpha=0.3)
    ax6.add_patch(circle)

    ax6.text(0.5, 0.6, indicator_text, ha='center', va='center',
            fontsize=16, fontweight='bold', color=indicator_color,
            transform=ax6.transAxes)

    ax6.text(0.5, 0.2, detail_text, ha='center', va='center',
            fontsize=11, transform=ax6.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Overall title
    fig.suptitle(f'A/B Test Dashboard: {result.test_name}',
                fontsize=18, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig
