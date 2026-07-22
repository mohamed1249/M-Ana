# def evaluate(model, X, y):
#     """
#     Evaluate the performance of a classification model using various metrics.

#     Args:
#     model: A trained classification model object with `predict()` and `predict_proba()` methods.
#     X: Input features for prediction.
#     y: Target variable for prediction.

#     Returns:
#     A dictionary with evaluation metrics including accuracy, precision, recall, f1-score, AUC-ROC, AUC-PRC, and confusion matrix.
#     """
#     from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix


#     y_pred = model.predict(X)
#     y_prob = model.predict_proba(X)[:, 1]

#     accuracy = accuracy_score(y, y_pred)
#     precision = precision_score(y, y_pred)
#     recall = recall_score(y, y_pred)
#     f1 = f1_score(y, y_pred)
#     roc_auc = roc_auc_score(y, y_prob)
#     prc_auc = average_precision_score(y, y_prob)
#     conf_matrix = confusion_matrix(y, y_pred)

#     metrics = {
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1-score': f1,
#         'AUC-ROC': roc_auc,
#         'AUC-PRC': prc_auc,
#         'confusion_matrix': conf_matrix
#     }

#     return metrics


import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings
from dataclasses import dataclass, field
import json


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report with metrics and visualizations."""

    model_type: str  # 'classification', 'regression', 'clustering'
    task_type: str  # 'binary', 'multiclass', 'regression', etc.
    metrics: Dict[str, Any] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        return {
            'model_type': self.model_type,
            'task_type': self.task_type,
            'metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                       for k, v in self.metrics.items()},
            'feature_importance': self.feature_importance
        }

    def to_json(self, filepath: Optional[str] = None) -> str:
        """Export report as JSON."""
        json_str = json.dumps(self.to_dict(), indent=2, default=str)
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
        return json_str

    def summary(self) -> str:
        """Generate human-readable summary."""
        summary = f"""
==============================================================
MODEL EVALUATION REPORT
==============================================================

MODEL TYPE: {self.model_type.upper()}
TASK: {self.task_type}

KEY METRICS
"""
        for metric, value in self.metrics.items():
            if isinstance(value, float):
                summary += f"  {metric:.<30} {value:.4f}\n"
            elif isinstance(value, (int, np.integer)):
                summary += f"  {metric:.<30} {value:,}\n"

        return summary


# ==================== CLASSIFICATION EVALUATION ====================

def evaluate_classification(
    model,
    X,
    y,
    class_names: Optional[List[str]] = None,
    average: str = 'binary',
    sample_weight: Optional[np.ndarray] = None,
    plot: bool = True,
    plot_size: Tuple[int, int] = (12, 4),
    return_report: bool = True
) -> Union[Dict, Tuple[Dict, EvaluationReport]]:
    """
    Comprehensive classification model evaluation.

    Supports binary and multi-class classification with extensive metrics,
    visualizations, and detailed reporting.

    Parameters:
    -----------
    model : sklearn-compatible model
        Trained model with predict() and predict_proba() methods.
    X : array-like
        Input features for prediction.
    y : array-like
        True target values.
    class_names : list, optional
        Names of classes for display.
    average : str, optional
        Averaging method for multi-class: 'binary', 'micro', 'macro', 'weighted'.
    sample_weight : array-like, optional
        Sample weights for weighted metrics.
    plot : bool, optional
        Generate visualization plots.
    plot_size : tuple, optional
        Size of plots (width, height).
    return_report : bool, optional
        Return detailed EvaluationReport object.

    Returns:
    --------
    dict or (dict, EvaluationReport)
        Metrics dictionary and optionally detailed report.

    Examples:
    ---------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>>
    >>> X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
    >>> model = RandomForestClassifier().fit(X[:800], y[:800])
    >>>
    >>> # Basic evaluation
    >>> metrics = evaluate_classification(model, X[800:], y[800:])
    >>> print(metrics['accuracy'])
    >>>
    >>> # With full report and visualizations
    >>> metrics, report = evaluate_classification(
    ...     model, X[800:], y[800:],
    ...     class_names=['Class 0', 'Class 1'],
    ...     plot=True,
    ...     return_report=True
    ... )
    >>> print(report.summary())
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix,
        classification_report, matthews_corrcoef, cohen_kappa_score,
        log_loss, balanced_accuracy_score, roc_curve, precision_recall_curve
    )

    # Get predictions
    y_pred = model.predict(X)

    # Determine if binary or multiclass
    n_classes = len(np.unique(y))
    is_binary = n_classes == 2

    # Get probabilities if available
    try:
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)
            y_prob_positive = y_prob[:, 1] if is_binary else None
        elif hasattr(model, 'decision_function'):
            y_prob_positive = model.decision_function(X)
            y_prob = None
        else:
            y_prob = None
            y_prob_positive = None
    except Exception:
        y_prob = None
        y_prob_positive = None

    # Calculate metrics
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y, y_pred, sample_weight=sample_weight)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y, y_pred, sample_weight=sample_weight)

    # Precision, Recall, F1
    if is_binary:
        avg = 'binary'
    else:
        avg = average

    metrics['precision'] = precision_score(y, y_pred, average=avg,
                                          sample_weight=sample_weight, zero_division=0)
    metrics['recall'] = recall_score(y, y_pred, average=avg,
                                    sample_weight=sample_weight, zero_division=0)
    metrics['f1_score'] = f1_score(y, y_pred, average=avg,
                                  sample_weight=sample_weight, zero_division=0)

    # Additional metrics
    metrics['matthews_corrcoef'] = matthews_corrcoef(y, y_pred, sample_weight=sample_weight)
    metrics['cohen_kappa'] = cohen_kappa_score(y, y_pred, sample_weight=sample_weight)

    # ROC AUC and PR AUC
    if y_prob_positive is not None:
        try:
            if is_binary:
                metrics['roc_auc'] = roc_auc_score(y, y_prob_positive, sample_weight=sample_weight)
                metrics['pr_auc'] = average_precision_score(y, y_prob_positive, sample_weight=sample_weight)
            else:
                # Multi-class ROC AUC
                metrics['roc_auc_ovr'] = roc_auc_score(y, y_prob, average='macro',
                                                       multi_class='ovr', sample_weight=sample_weight)
                metrics['roc_auc_ovo'] = roc_auc_score(y, y_prob, average='macro',
                                                       multi_class='ovo', sample_weight=sample_weight)
        except (TypeError, ValueError):
            pass

    # Log loss
    if y_prob is not None:
        try:
            metrics['log_loss'] = log_loss(y, y_prob, sample_weight=sample_weight)
        except (TypeError, ValueError):
            pass

    # Confusion matrix
    conf_matrix = confusion_matrix(y, y_pred, sample_weight=sample_weight)
    metrics['confusion_matrix'] = conf_matrix

    # Per-class metrics
    class_report = classification_report(y, y_pred, target_names=class_names,
                                        sample_weight=sample_weight, output_dict=True)
    metrics['per_class_metrics'] = class_report

    # Specificity (for binary)
    if is_binary:
        tn, fp, fn, tp = conf_matrix.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = metrics['recall']  # Same as recall
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        metrics['false_discovery_rate'] = fp / (fp + tp) if (fp + tp) > 0 else 0

    # Create report object
    if return_report:
        report = EvaluationReport(
            model_type='classification',
            task_type='binary' if is_binary else 'multiclass',
            metrics=metrics,
            confusion_matrix=conf_matrix,
            predictions=y_pred,
            probabilities=y_prob
        )

    # Visualizations
    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns

        if is_binary and y_prob_positive is not None:
            # Binary classification plots
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            # 1. Confusion Matrix
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names or ['Class 0', 'Class 1'],
                       yticklabels=class_names or ['Class 0', 'Class 1'],
                       ax=axes[0])
            axes[0].set_title('Confusion Matrix')
            axes[0].set_ylabel('True Label')
            axes[0].set_xlabel('Predicted Label')

            # 2. ROC Curve
            fpr, tpr, _ = roc_curve(y, y_prob_positive, sample_weight=sample_weight)
            axes[1].plot(fpr, tpr, linewidth=2, label=f'AUC = {metrics["roc_auc"]:.3f}')
            axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            axes[1].set_xlabel('False Positive Rate')
            axes[1].set_ylabel('True Positive Rate')
            axes[1].set_title('ROC Curve')
            axes[1].legend()
            axes[1].grid(alpha=0.3)

            # 3. Precision-Recall Curve
            precision_curve, recall_curve, _ = precision_recall_curve(y, y_prob_positive,
                                                                      sample_weight=sample_weight)
            axes[2].plot(recall_curve, precision_curve, linewidth=2,
                        label=f'AP = {metrics["pr_auc"]:.3f}')
            axes[2].set_xlabel('Recall')
            axes[2].set_ylabel('Precision')
            axes[2].set_title('Precision-Recall Curve')
            axes[2].legend()
            axes[2].grid(alpha=0.3)

            plt.tight_layout()
            plt.show()

        else:
            # Multi-class confusion matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names or [f'Class {i}' for i in range(n_classes)],
                       yticklabels=class_names or [f'Class {i}' for i in range(n_classes)],
                       ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            plt.tight_layout()
            plt.show()

    if return_report:
        return metrics, report
    return metrics


# ==================== REGRESSION EVALUATION ====================

def evaluate_regression(
    model,
    X,
    y,
    sample_weight: Optional[np.ndarray] = None,
    plot: bool = True,
    plot_size: Tuple[int, int] = (15, 5),
    return_report: bool = True
) -> Union[Dict, Tuple[Dict, EvaluationReport]]:
    """
    Comprehensive regression model evaluation.

    Parameters:
    -----------
    model : sklearn-compatible model
        Trained regression model with predict() method.
    X : array-like
        Input features for prediction.
    y : array-like
        True target values.
    sample_weight : array-like, optional
        Sample weights for weighted metrics.
    plot : bool, optional
        Generate visualization plots.
    plot_size : tuple, optional
        Size of plots (width, height).
    return_report : bool, optional
        Return detailed EvaluationReport object.

    Returns:
    --------
    dict or (dict, EvaluationReport)
        Metrics dictionary and optionally detailed report.

    Examples:
    ---------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.datasets import make_regression
    >>>
    >>> X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
    >>> model = RandomForestRegressor().fit(X[:800], y[:800])
    >>>
    >>> # Evaluate
    >>> metrics, report = evaluate_regression(
    ...     model, X[800:], y[800:],
    ...     plot=True,
    ...     return_report=True
    ... )
    >>> print(report.summary())
    """
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
        mean_absolute_percentage_error, median_absolute_error,
        max_error, explained_variance_score
    )

    # Get predictions
    y_pred = model.predict(X)

    # Calculate residuals
    residuals = y - y_pred

    # Calculate metrics
    metrics = {}

    # Error metrics
    metrics['mse'] = mean_squared_error(y, y_pred, sample_weight=sample_weight)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y, y_pred, sample_weight=sample_weight)
    metrics['median_ae'] = median_absolute_error(y, y_pred)
    metrics['max_error'] = max_error(y, y_pred)

    # MAPE (handle zero division)
    try:
        metrics['mape'] = mean_absolute_percentage_error(y, y_pred, sample_weight=sample_weight)
    except (TypeError, ValueError, ZeroDivisionError):
        metrics['mape'] = np.nan

    # R-squared and adjusted R-squared
    metrics['r2'] = r2_score(y, y_pred, sample_weight=sample_weight)
    n = len(y)
    p = X.shape[1] if hasattr(X, 'shape') else 1
    metrics['adjusted_r2'] = 1 - (1 - metrics['r2']) * (n - 1) / (n - p - 1)

    # Explained variance
    metrics['explained_variance'] = explained_variance_score(y, y_pred, sample_weight=sample_weight)

    # Additional metrics
    metrics['mean_residual'] = np.mean(residuals)
    metrics['std_residual'] = np.std(residuals)

    # Mean squared logarithmic error (for positive values)
    if np.all(y > 0) and np.all(y_pred > 0):
        from sklearn.metrics import mean_squared_log_error
        metrics['msle'] = mean_squared_log_error(y, y_pred, sample_weight=sample_weight)
        metrics['rmsle'] = np.sqrt(metrics['msle'])

    # Relative metrics
    metrics['relative_rmse'] = metrics['rmse'] / (np.max(y) - np.min(y))
    metrics['relative_mae'] = metrics['mae'] / (np.max(y) - np.min(y))

    # Create report object
    if return_report:
        report = EvaluationReport(
            model_type='regression',
            task_type='regression',
            metrics=metrics,
            predictions=y_pred,
            residuals=residuals
        )

    # Visualizations
    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=plot_size)

        # 1. Predicted vs Actual
        axes[0].scatter(y, y_pred, alpha=0.5, s=20)
        axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title(f'Predicted vs Actual\nR² = {metrics["r2"]:.4f}')
        axes[0].grid(alpha=0.3)

        # 2. Residual Plot
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        axes[1].grid(alpha=0.3)

        # 3. Residual Distribution
        axes[2].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[2].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[2].set_xlabel('Residuals')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title(f'Residual Distribution\nMean = {metrics["mean_residual"]:.4f}')
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    if return_report:
        return metrics, report
    return metrics


# ==================== CLUSTERING EVALUATION ====================

def evaluate_clustering(
    model,
    X,
    labels_true: Optional[np.ndarray] = None,
    plot: bool = True,
    plot_size: Tuple[int, int] = (12, 4),
    return_report: bool = True
) -> Union[Dict, Tuple[Dict, EvaluationReport]]:
    """
    Comprehensive clustering model evaluation.

    Parameters:
    -----------
    model : sklearn-compatible clustering model
        Fitted clustering model with labels_ attribute or predict() method.
    X : array-like
        Input features.
    labels_true : array-like, optional
        True labels (if available for supervised metrics).
    plot : bool, optional
        Generate visualization plots.
    plot_size : tuple, optional
        Size of plots (width, height).
    return_report : bool, optional
        Return detailed EvaluationReport object.

    Returns:
    --------
    dict or (dict, EvaluationReport)
        Metrics dictionary and optionally detailed report.

    Examples:
    ---------
    >>> from sklearn.cluster import KMeans
    >>> from sklearn.datasets import make_blobs
    >>>
    >>> X, y_true = make_blobs(n_samples=500, centers=3, random_state=42)
    >>> model = KMeans(n_clusters=3, random_state=42).fit(X)
    >>>
    >>> # Evaluate with true labels
    >>> metrics, report = evaluate_clustering(
    ...     model, X, labels_true=y_true,
    ...     plot=True,
    ...     return_report=True
    ... )
    >>> print(report.summary())
    """
    from sklearn.metrics import (
        silhouette_score, calinski_harabasz_score, davies_bouldin_score,
        adjusted_rand_score, normalized_mutual_info_score,
        fowlkes_mallows_score, homogeneity_score, completeness_score, v_measure_score
    )

    # Get cluster labels
    if hasattr(model, 'labels_'):
        labels_pred = model.labels_
    elif hasattr(model, 'predict'):
        labels_pred = model.predict(X)
    else:
        raise ValueError("Model must have either 'labels_' attribute or 'predict()' method")

    # Calculate metrics
    metrics = {}

    # Unsupervised metrics (don't need true labels)
    metrics['silhouette_score'] = silhouette_score(X, labels_pred)
    metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels_pred)
    metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels_pred)

    # Number of clusters
    n_clusters = len(np.unique(labels_pred))
    metrics['n_clusters'] = n_clusters

    # Cluster sizes
    unique, counts = np.unique(labels_pred, return_counts=True)
    metrics['cluster_sizes'] = dict(zip(unique.tolist(), counts.tolist()))
    metrics['min_cluster_size'] = int(np.min(counts))
    metrics['max_cluster_size'] = int(np.max(counts))
    metrics['mean_cluster_size'] = float(np.mean(counts))
    metrics['std_cluster_size'] = float(np.std(counts))

    # Inertia (for K-Means)
    if hasattr(model, 'inertia_'):
        metrics['inertia'] = float(model.inertia_)

    # Supervised metrics (if true labels available)
    if labels_true is not None:
        metrics['adjusted_rand_score'] = adjusted_rand_score(labels_true, labels_pred)
        metrics['normalized_mutual_info'] = normalized_mutual_info_score(labels_true, labels_pred)
        metrics['fowlkes_mallows_score'] = fowlkes_mallows_score(labels_true, labels_pred)
        metrics['homogeneity'] = homogeneity_score(labels_true, labels_pred)
        metrics['completeness'] = completeness_score(labels_true, labels_pred)
        metrics['v_measure'] = v_measure_score(labels_true, labels_pred)

    # Create report object
    if return_report:
        report = EvaluationReport(
            model_type='clustering',
            task_type='unsupervised',
            metrics=metrics,
            predictions=labels_pred
        )

    # Visualizations
    if plot and X.shape[1] >= 2:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        # If more than 2D, use PCA
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_plot = pca.fit_transform(X)
        else:
            X_plot = X

        if labels_true is not None:
            fig, axes = plt.subplots(1, 2, figsize=plot_size)

            # True labels
            scatter1 = axes[0].scatter(X_plot[:, 0], X_plot[:, 1],
                                      c=labels_true, cmap='viridis',
                                      alpha=0.6, s=30)
            axes[0].set_title('True Labels')
            axes[0].set_xlabel('Component 1')
            axes[0].set_ylabel('Component 2')
            plt.colorbar(scatter1, ax=axes[0])

            # Predicted labels
            scatter2 = axes[1].scatter(X_plot[:, 0], X_plot[:, 1],
                                      c=labels_pred, cmap='viridis',
                                      alpha=0.6, s=30)
            axes[1].set_title(f'Predicted Clusters (n={n_clusters})')
            axes[1].set_xlabel('Component 1')
            axes[1].set_ylabel('Component 2')
            plt.colorbar(scatter2, ax=axes[1])
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1],
                               c=labels_pred, cmap='viridis',
                               alpha=0.6, s=30)
            ax.set_title(f'Clusters (n={n_clusters})')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            plt.colorbar(scatter, ax=ax)

        plt.tight_layout()
        plt.show()

    if return_report:
        return metrics, report
    return metrics


# ==================== CROSS-VALIDATION ====================

def cross_validate_model(
    model,
    X,
    y,
    cv: int = 5,
    scoring: Optional[Union[str, List[str]]] = None,
    return_train_score: bool = True,
    plot: bool = True
) -> Dict:
    """
    Perform cross-validation with comprehensive metrics and visualization.

    Parameters:
    -----------
    model : sklearn-compatible model
        Model to evaluate.
    X : array-like
        Input features.
    y : array-like
        Target values.
    cv : int, optional
        Number of cross-validation folds.
    scoring : str or list, optional
        Scoring metric(s). If None, uses default for task type.
    return_train_score : bool, optional
        Calculate training scores.
    plot : bool, optional
        Generate visualization.

    Returns:
    --------
    dict
        Cross-validation results with mean and std for each metric.

    Examples:
    ---------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>>
    >>> X, y = make_classification(n_samples=1000, random_state=42)
    >>> model = RandomForestClassifier(random_state=42)
    >>>
    >>> # Cross-validate
    >>> cv_results = cross_validate_model(
    ...     model, X, y, cv=5,
    ...     scoring=['accuracy', 'f1', 'roc_auc'],
    ...     plot=True
    ... )
    >>> print(f"Mean Accuracy: {cv_results['test_accuracy_mean']:.4f}")
    """
    from sklearn.model_selection import cross_validate as sk_cross_validate

    # Default scoring based on task
    if scoring is None:
        # Try to detect task type
        if hasattr(model, '_estimator_type'):
            if model._estimator_type == 'classifier':
                scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            elif model._estimator_type == 'regressor':
                scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
        else:
            scoring = 'accuracy'  # Default

    # Perform cross-validation
    cv_results = sk_cross_validate(
        model, X, y, cv=cv,
        scoring=scoring,
        return_train_score=return_train_score,
        n_jobs=-1
    )

    # Calculate statistics
    results = {}
    for key, values in cv_results.items():
        if key.startswith('test_') or key.startswith('train_'):
            metric_name = key
            results[f'{metric_name}_mean'] = np.mean(values)
            results[f'{metric_name}_std'] = np.std(values)
            results[f'{metric_name}_min'] = np.min(values)
            results[f'{metric_name}_max'] = np.max(values)
            results[f'{metric_name}_scores'] = values

    # Visualization
    if plot:
        import matplotlib.pyplot as plt

        # Extract test metrics
        test_metrics = {k.replace('test_', ''): v
                       for k, v in results.items()
                       if k.startswith('test_') and k.endswith('_scores')}

        if test_metrics:
            n_metrics = len(test_metrics)
            fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))

            if n_metrics == 1:
                axes = [axes]

            for idx, (metric_name, scores) in enumerate(test_metrics.items()):
                metric_clean = metric_name.replace('_scores', '')
                axes[idx].boxplot([scores], labels=[metric_clean])
                axes[idx].scatter([1]*len(scores), scores, alpha=0.5, s=50)
                axes[idx].set_title(f'{metric_clean.upper()}\nMean: {np.mean(scores):.4f} +/- {np.std(scores):.4f}')
                axes[idx].set_ylabel('Score')
                axes[idx].grid(alpha=0.3, axis='y')

            plt.tight_layout()
            plt.show()

    return results


# ==================== FEATURE IMPORTANCE ====================

def plot_feature_importance(
    model,
    feature_names: Optional[List[str]] = None,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8)
) -> pd.DataFrame:
    """
    Plot and return feature importance from a trained model.

    Parameters:
    -----------
    model : sklearn-compatible model
        Trained model with feature_importances_ or coef_ attribute.
    feature_names : list, optional
        Names of features.
    top_n : int, optional
        Number of top features to display.
    figsize : tuple, optional
        Figure size.

    Returns:
    --------
    pd.DataFrame
        Feature importance dataframe sorted by importance.

    Examples:
    ---------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_iris
    >>>
    >>> iris = load_iris()
    >>> model = RandomForestClassifier().fit(iris.data, iris.target)
    >>>
    >>> importance_df = plot_feature_importance(
    ...     model,
    ...     feature_names=iris.feature_names,
    ...     top_n=4
    ... )
    """
    import matplotlib.pyplot as plt

    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
    else:
        raise ValueError("Model must have 'feature_importances_' or 'coef_' attribute")

    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importances))]

    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Plot
    top_features = importance_df.head(top_n)

    plt.figure(figsize=figsize)
    plt.barh(range(len(top_features)), top_features['importance'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    backend = plt.get_backend().lower()
    if "agg" not in backend or "inline" in backend:
        plt.show()

    return importance_df


# ==================== LEGACY FUNCTION (backward compatibility) ====================

def evaluate(model, X, y):
    """
    Evaluate classification model (legacy function for backward compatibility).

    Use evaluate_classification() for more comprehensive evaluation.
    """
    warnings.warn(
        "This function is deprecated. Use evaluate_classification() for more features.",
        DeprecationWarning
    )

    metrics = evaluate_classification(model, X, y, plot=False, return_report=False)

    # Return in old format
    return {
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1-score': metrics['f1_score'],
        'AUC-ROC': metrics.get('roc_auc', None),
        'AUC-PRC': metrics.get('pr_auc', None),
        'confusion_matrix': metrics['confusion_matrix']
    }
