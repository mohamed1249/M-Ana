# def confusion_matrix(y_true, y_pred, size=20, cmap='Greens'):
#     """
#     Plots a confusion matrix using seaborn.

#     Parameters:
#     y_true: array-like of shape (n_samples,)
#         Ground truth (correct) target values.
#     y_pred: array-like of shape (n_samples,)
#         Estimated targets as returned by a classifier.

#     Returns:
#     None
#     """
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     from sklearn.metrics import confusion_matrix

#     plt.figure(figsize = (20,20))
#     cm = confusion_matrix(y_true, y_pred)
#     ax = sns.heatmap(cm, annot=True, fmt='g', cbar=False, cmap=cmap)
#     ax.set_xlabel('Predicted Values')
#     ax.set_ylabel('Actual Values')
#     plt.show()


# def roc_curve(y_true, y_pred):
#     """
#     Plots the ROC curve for binary classification given the true labels and predicted probabilities.

#     Parameters:
#     y_true: array-like, shape (n_samples,)
#         True binary labels in {0, 1}

#     y_pred: array-like, shape (n_samples,)
#         Target scores, can either be probability estimates of the positive class or confidence values.
#     """
#     import matplotlib.pyplot as plt
#     from sklearn.metrics import roc_curve, auc

#     # Calculate the false positive rate (fpr), true positive rate (tpr), and thresholds
#     fpr, tpr, thresholds = roc_curve(y_true, y_pred)

#     # Calculate the area under the curve (AUC)
#     roc_auc = auc(fpr, tpr)

#     # Plot the ROC curve
#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic (ROC) curve')
#     plt.legend(loc="lower right")
#     plt.show()


# def precision_recall_curve(y_true, y_pred):
#     """
#     Plots a Precision-Recall curve based on the true and predicted labels.

#     Parameters:
#     y_true: array-like of shape (n_samples,), true binary labels.
#     y_pred: array-like of shape (n_samples,), the predicted probabilities or binary labels.

#     Returns:
#     None
#     """
#     import matplotlib.pyplot as plt
#     from sklearn.metrics import precision_recall_curve


#     precision, recall, _ = precision_recall_curve(y_true, y_pred)
#     plt.plot(recall, precision, label='Precision-Recall curve')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.legend()
#     plt.show()


# def lagged_scatter_plot(data, variable, lag=1):
#     """
#     Plots a lagged scatter plot of a variable with its lagged values.

#     Parameters:
#     data (pandas.DataFrame): The input data.
#     variable (str): The name of the variable to plot.
#     lag (int): The lag to use for plotting. Default is 1.

#     Returns:
#     None
#     """
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     ax.scatter(data[variable], data[variable].shift(lag))
#     ax.set_xlabel(variable)
#     ax.set_ylabel(variable + f" (lag={lag})")
#     ax.set_title(f"Lagged Scatter Plot of {variable} (lag={lag})")
#     plt.show()


# def lag_correlation(series):
#     """
#     Description: This function generates an autocorrelation and partial autocorrelation plot for a given time series.

#     Parameters:
#     series: pandas Series representing the time series to be plotted.

#     Returns:
#     None
#     """
#     import matplotlib.pyplot as plt
#     import statsmodels.api as sm


#     fig, ax = plt.subplots(nrows=2, figsize=(10, 8))

#     # Autocorrelation plot
#     sm.graphics.tsa.plot_acf(series, lags=50, ax=ax[0])
#     ax[0].set_xlabel('Lag')
#     ax[0].set_ylabel('Autocorrelation')
#     ax[0].set_title('Autocorrelation Plot')

#     # Partial autocorrelation plot
#     sm.graphics.tsa.plot_pacf(series, lags=50, ax=ax[1])
#     ax[1].set_xlabel('Lag')
#     ax


# def seasonal_decompose_plot(data, freq):
#     """
#     Description: This function takes in a time series data and the frequency of the seasonality and decomposes it into seasonal, trend, and residual components.

#     Parameters:
#     data: pandas Series or DataFrame with datetime index containing the time series data
#     freq: integer representing the frequency of the seasonality

#     Returns:
#     A matplotlib figure object containing the seasonal decomposition plot
#     """
#     import matplotlib.pyplot as plt
#     import statsmodels.api as sm


#     # Decompose the time series
#     decomposition = sm.tsa.seasonal_decompose(data, model='additive', period=freq)

#     # Plot the decomposition
#     fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(12,8))
#     plt.subplots_adjust(hspace=0.3)

#     axes[0].set_title('Observed')
#     decomposition.observed.plot(ax=axes[0], legend=False)

#     axes[1].set_title('Trend')
#     decomposition.trend.plot(ax=axes[1], legend=False)

#     axes[2].set_title('Seasonal')
#     decomposition.seasonal.plot(ax=axes[2], legend=False)

#     axes[3].set_title('Residual')
#     decomposition.resid.plot(ax=axes[3], legend=False)

#     plt.tight_layout()

#     return fig


# def plot_forecast(actual, forecast, xlabel='', ylabel='', title=''):
#     """
#     Plots a time series forecasting plot showing the actual and forecasted values.

#     Parameters:
#     actual (pandas.Series): Actual time series data.
#     forecast (pandas.Series): Forecasted time series data.
#     xlabel (str): Label for the x-axis.
#     ylabel (str): Label for the y-axis.
#     title (str): Title for the plot.
#     """
#     import matplotlib.pyplot as plt


#     plt.plot(actual.index, actual.values, label='Actual')
#     plt.plot(forecast.index, forecast.values, label='Forecast')
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.legend()
#     plt.show()


# def plot_residuals(y_true, y_pred):
#     """
#     Plots a residual plot to visualize the difference between predicted and actual values.

#     Parameters:
#     y_true: array-like, true values of the target variable
#     y_pred: array-like, predicted values of the target variable

#     Returns:
#     None
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np


#     residuals = y_true - y_pred
#     plt.figure(figsize=(10,6))
#     plt.scatter(np.arange(len(residuals)), residuals)
#     plt.axhline(y=0, color='r', linestyle='-')
#     plt.title("Residual Plot")
#     plt.xlabel("Observations")
#     plt.ylabel("Residuals")
#     plt.show()


# def plot_spectral_density(data, sampling_rate):
#     """
#     Plots a spectral density plot of the given data.

#     Parameters:
#     data (pandas.Series): Time series data to plot.
#     sampling_rate (int): Sampling rate of the data.

#     Returns:
#     None (displays plot in console)
#     """
#     import matplotlib.pyplot as plt
#     from scipy import signal

#     freq, psd = signal.welch(data, fs=sampling_rate)
#     plt.figure(figsize=(8, 4))
#     plt.plot(freq, psd)
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('PSD')
#     plt.title('Spectral Density Plot')
#     plt.show()


# def plot_cluster(X, y, title=''):
#     """
#     Description: This function creates a cluster plot to visualize the clusters formed by unsupervised learning algorithms.

#     Parameters:

#     X: numpy array or pandas DataFrame containing the data to be plotted.
#     y: numpy array or pandas Series containing the cluster labels for each data point.
#     title: string representing the title of the plot (optional).

#     Returns:

#     A seaborn cluster plot object.
#     """
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     import pandas as pd

#     # Convert pandas objects to numpy arrays
#     if isinstance(X, pd.DataFrame):
#         X = X.values
#     if isinstance(y, pd.Series):
#         y = y.values

#     # Create the plot
#     sns.clustermap(X, row_cluster=False, col_cluster=False, cmap='viridis', robust=True, yticklabels=y, figsize=(10,10))
#     plt.title(title)
#     plt.show()


# def plot_pca(df, target_col=None):
#     """
#     Plots a scatter plot using Principal Component Analysis (PCA) for dimensionality reduction.

#     Parameters:
#     df (pandas.DataFrame): DataFrame containing the data to be plotted.
#     target_col (str, optional): Name of the target column. If provided, the plot will show different colors for different target values.

#     Returns:
#     None
#     """

#     import matplotlib.pyplot as plt
#     from sklearn.decomposition import PCA
#     import pandas as pd

#     # Apply PCA to the data
#     pca = PCA(n_components=2)
#     pca_df = pd.DataFrame(pca.fit_transform(df.drop(target_col, axis=1) if target_col else df))
#     pca_df.columns = ['PC1', 'PC2']

#     # Add target column to the PCA DataFrame if provided
#     if target_col:
#         pca_df['target'] = df[target_col]

#     # Plot the PCA scatter plot
#     fig, ax = plt.subplots()
#     if target_col:
#         for target_val in df[target_col].unique():
#             ix = pca_df['target'] == target_val
#             ax.scatter(pca_df.loc[ix, 'PC1'], pca_df.loc[ix, 'PC2'], label=target_val)
#     else:
#         ax.scatter(pca_df['PC1'], pca_df['PC2'])

#     ax.set_xlabel('PC1')
#     ax.set_ylabel('PC2')
#     ax.legend()
#     plt.show()


# def silhouette(X, y_pred):

#     import matplotlib.pyplot as plt
#     from sklearn.metrics import silhouette_samples, silhouette_score
#     import numpy as np

#     cluster_labels = np.unique(y_pred)
#     n_clusters = cluster_labels.shape[0]
#     silhouette_vals = silhouette_samples(X, y_pred)
#     y_ax_lower, y_ax_upper = 0, 0
#     yticks = []
#     for i, c in enumerate(cluster_labels):
#         c_silhouette_vals = silhouette_vals[y_pred == c]
#         c_silhouette_vals.sort()
#         y_ax_upper += len(c_silhouette_vals)
#         color = plt.cm.Spectral(float(i) / n_clusters)
#         plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=color)
#         yticks.append((y_ax_lower + y_ax_upper) / 2.)
#         y_ax_lower += len(c_silhouette_vals)
#     silhouette_avg = silhouette_score(X, y_pred)
#     plt.axvline(silhouette_avg, color="red", linestyle="--")
#     plt.yticks(yticks, cluster_labels + 1)
#     plt.ylabel('Cluster')
#     plt.xlabel('Silhouette Coefficient')
#     plt.show()


# def dendrogram(X):
#     '''
#     Plots a dendrogram based on hierarchical clustering of the data X.

#     Parameters:
#     X (numpy array): data to be clustered

#     Returns:
#     None
#     '''
#     import matplotlib.pyplot as plt
#     from scipy.cluster.hierarchy import dendrogram, linkage

#     # Calculate linkage matrix
#     Z = linkage(X, 'ward')

#     # Plot dendrogram
#     plt.figure(figsize=(10, 5))
#     plt.title('Hierarchical Clustering Dendrogram')
#     plt.xlabel('Data points')
#     plt.ylabel('Distance')
#     dendrogram(Z)
#     plt.show()


# def plot_elbow(X, max_k=10):
#     """
#     Plots the elbow method curve to help determine the optimal number of clusters for k-means clustering.

#     Parameters:
#         X (numpy.ndarray): Array of shape (n_samples, n_features) containing the data to be clustered.
#         max_k (int): Maximum number of clusters to test. Default is 10.

#     Returns:
#         None
#     """
#     import matplotlib.pyplot as plt
#     from sklearn.cluster import KMeans

#     sse = []
#     for k in range(1, max_k+1):
#         kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
#         sse.append(kmeans.inertia_)

#     # Plot the elbow curve
#     plt.plot(range(1, max_k+1), sse, marker='o')
#     plt.xlabel('Number of clusters (k)')
#     plt.ylabel('Sum of squared distances')
#     plt.title('Elbow Method Curve')
#     plt.show()


# def loss_plot(model, loss_path = ['loss', 'val_loss'], start=0):
#     import pandas as pd

#     history_df = pd.DataFrame(model.history)
#     print("Minimum loss: {}".format(history_df['loss'].min()))
#     print("Minimum validation loss: {}".format(history_df['val_loss'].min()))
#     history_df.loc[start:,loss_path].plot();


# # ==================== MODEL VISUALIZATION ENHANCEMENTS ====================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Union


# ==================== CLASSIFICATION VISUALIZATIONS ====================

def confusion_matrix(
    y_true,
    y_pred,
    labels: Optional[List[str]] = None,
    normalize: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'Blues',
    annot_kws: Optional[Dict] = None,
    title: str = 'Confusion Matrix',
    save_path: Optional[str] = None,
    interactive: bool = False
):
    """
    Enhanced confusion matrix visualization with normalization and interactive options.

    Parameters:
    -----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    labels : list, optional
        Class labels for display.
    normalize : str, optional
        Normalize: 'true', 'pred', 'all', or None.
    figsize : tuple
        Figure size.
    cmap : str
        Color map.
    annot_kws : dict, optional
        Annotation keyword arguments.
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
    >>> # Basic confusion matrix
    >>> confusion_matrix(y_true, y_pred)
    >>>
    >>> # Normalized by true labels
    >>> confusion_matrix(y_true, y_pred, normalize='true',
    ...                  labels=['Class A', 'Class B', 'Class C'])
    >>>
    >>> # Interactive version
    >>> confusion_matrix(y_true, y_pred, interactive=True)
    """
    from sklearn.metrics import confusion_matrix as sklearn_cm

    # Calculate confusion matrix
    cm = sklearn_cm(y_true, y_pred)

    # Normalize if requested
    if normalize == 'true':
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    elif normalize == 'pred':
        cm = cm.astype('float') / cm.sum(axis=0)
        fmt = '.2%'
    elif normalize == 'all':
        cm = cm.astype('float') / cm.sum()
        fmt = '.2%'
    else:
        fmt = 'd'

    if interactive:
        # Interactive Plotly version
        import plotly.graph_objects as go

        if labels is None:
            labels = [f'Class {i}' for i in range(len(cm))]

        # Format annotations
        if normalize:
            z_text = [[f'{val:.1%}' for val in row] for row in cm]
        else:
            z_text = [[f'{int(val)}' for val in row] for row in cm]

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=z_text,
            texttemplate='%{text}',
            colorscale=cmap,
            showscale=True
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            width=600,
            height=600
        )

        if save_path:
            fig.write_html(save_path)

        fig.show()
        return fig

    else:
        # Matplotlib version
        fig, ax = plt.subplots(figsize=figsize)

        annot_kws = annot_kws or {'fontsize': 10}

        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            cbar=True,
            square=True,
            xticklabels=labels if labels else 'auto',
            yticklabels=labels if labels else 'auto',
            annot_kws=annot_kws,
            ax=ax
        )

        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

        # Add accuracy info
        accuracy = np.trace(cm) / np.sum(cm)
        ax.text(
            0.5, -0.1,
            f'Overall Accuracy: {accuracy:.2%}',
            ha='center',
            va='top',
            transform=ax.transAxes,
            fontsize=11,
            fontweight='bold'
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
        return fig


def roc_curve(
    y_true,
    y_pred_proba,
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = 'ROC Curve',
    save_path: Optional[str] = None,
    interactive: bool = False
):
    """
    Enhanced ROC curve with support for multi-class classification.

    Parameters:
    -----------
    y_true : array-like
        True labels (can be multi-class).
    y_pred_proba : array-like
        Predicted probabilities. For binary: 1D array. For multi-class: 2D array.
    labels : list, optional
        Class labels.
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
    >>> # Binary classification
    >>> roc_curve(y_true, y_pred_proba)
    >>>
    >>> # Multi-class classification
    >>> roc_curve(y_true, y_pred_proba, labels=['Class A', 'Class B', 'Class C'])
    >>>
    >>> # Interactive version
    >>> roc_curve(y_true, y_pred_proba, interactive=True)
    """
    from sklearn.metrics import roc_curve as sklearn_roc, auc
    from sklearn.preprocessing import label_binarize

    # Determine if binary or multi-class
    unique_classes = np.unique(y_true)
    n_classes = len(unique_classes)

    if n_classes == 2:
        # Binary classification
        if len(y_pred_proba.shape) > 1:
            y_pred_proba = y_pred_proba[:, 1]

        fpr, tpr, _ = sklearn_roc(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        if interactive:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC (AUC = {roc_auc:.3f})',
                line=dict(color='darkorange', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='navy', width=2, dash='dash')
            ))

            fig.update_layout(
                title=title,
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                width=700,
                height=600,
                showlegend=True
            )

            if save_path:
                fig.write_html(save_path)

            fig.show()
            return fig

        else:
            fig, ax = plt.subplots(figsize=figsize)

            ax.plot(fpr, tpr, color='darkorange', lw=2.5,
                   label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                   label='Random Classifier')

            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(loc="lower right", fontsize=11)
            ax.grid(alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            plt.show()
            return fig

    else:
        # Multi-class classification
        y_true_bin = label_binarize(y_true, classes=unique_classes)

        if labels is None:
            labels = [f'Class {i}' for i in unique_classes]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = sklearn_roc(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve
        fpr["micro"], tpr["micro"], _ = sklearn_roc(
            y_true_bin.ravel(), y_pred_proba.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        if interactive:
            import plotly.graph_objects as go

            fig = go.Figure()

            # Plot each class
            for i in range(n_classes):
                fig.add_trace(go.Scatter(
                    x=fpr[i], y=tpr[i],
                    mode='lines',
                    name=f'{labels[i]} (AUC = {roc_auc[i]:.3f})',
                    line=dict(width=2)
                ))

            # Micro-average
            fig.add_trace(go.Scatter(
                x=fpr["micro"], y=tpr["micro"],
                mode='lines',
                name=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
                line=dict(color='black', width=3, dash='dot')
            ))

            # Random classifier
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='gray', width=2, dash='dash')
            ))

            fig.update_layout(
                title=title,
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                width=800,
                height=600
            )

            if save_path:
                fig.write_html(save_path)

            fig.show()
            return fig

        else:
            fig, ax = plt.subplots(figsize=figsize)

            # Plot each class
            colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
            for i, color in zip(range(n_classes), colors):
                ax.plot(fpr[i], tpr[i], color=color, lw=2,
                       label=f'{labels[i]} (AUC = {roc_auc[i]:.3f})')

            # Micro-average
            ax.plot(fpr["micro"], tpr["micro"], color='black', lw=3,
                   linestyle=':', label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})')

            # Random classifier
            ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')

            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            plt.show()
            return fig


def precision_recall_curve(
    y_true,
    y_pred_proba,
    figsize: Tuple[int, int] = (10, 8),
    title: str = 'Precision-Recall Curve',
    save_path: Optional[str] = None,
    interactive: bool = False
):
    """
    Enhanced Precision-Recall curve with Average Precision score.

    Parameters:
    -----------
    y_true : array-like
        True binary labels.
    y_pred_proba : array-like
        Predicted probabilities for positive class.
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
    >>> # Basic PR curve
    >>> precision_recall_curve(y_true, y_pred_proba)
    >>>
    >>> # Interactive version
    >>> precision_recall_curve(y_true, y_pred_proba, interactive=True)
    """
    from sklearn.metrics import precision_recall_curve as sklearn_pr, average_precision_score

    precision, recall, _ = sklearn_pr(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)

    if interactive:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'PR Curve (AP = {avg_precision:.3f})',
            fill='tozeroy',
            line=dict(color='blue', width=2)
        ))

        # Baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[baseline, baseline],
            mode='lines',
            name='Baseline (Random)',
            line=dict(color='red', width=2, dash='dash')
        ))

        fig.update_layout(
            title=f'{title}<br>Average Precision: {avg_precision:.3f}',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=700,
            height=600
        )

        if save_path:
            fig.write_html(save_path)

        fig.show()
        return fig

    else:
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(recall, precision, color='blue', lw=2.5,
               label=f'PR curve (AP = {avg_precision:.3f})')

        # Baseline
        baseline = np.sum(y_true) / len(y_true)
        ax.axhline(y=baseline, color='red', linestyle='--', lw=2,
                  label=f'Baseline (Random: {baseline:.3f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc="lower left", fontsize=11)
        ax.grid(alpha=0.3)

        # Fill area under curve
        ax.fill_between(recall, precision, alpha=0.2, color='blue')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
        return fig


def plot_classification_report(
    y_true,
    y_pred,
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    cmap: str = 'RdYlGn',
    title: str = 'Classification Report',
    save_path: Optional[str] = None
):
    """
    Visualize classification report as heatmap.

    Parameters:
    -----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    labels : list, optional
        Class labels.
    figsize : tuple
        Figure size.
    cmap : str
        Color map.
    title : str
        Plot title.
    save_path : str, optional
        Path to save figure.

    Returns:
    --------
    matplotlib.figure.Figure

    Examples:
    ---------
    >>> plot_classification_report(y_true, y_pred,
    ...                            labels=['Negative', 'Positive'])
    """
    from sklearn.metrics import classification_report

    # Get classification report as dict
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)

    # Convert to DataFrame
    df_report = pd.DataFrame(report).transpose()

    # Remove support column and accuracy/macro/weighted avg rows for cleaner viz
    df_viz = df_report.iloc[:-3, :-1]  # Remove last 3 rows and support column

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        df_viz,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        cbar=True,
        linewidths=1,
        linecolor='white',
        vmin=0,
        vmax=1,
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Class', fontsize=12, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')

    # Add overall metrics as text
    accuracy = report['accuracy']
    macro_avg = report['macro avg']
    weighted_avg = report['weighted avg']

    info_text = (
        f"Accuracy: {accuracy:.3f}\\n"
        f"Macro Avg - P: {macro_avg['precision']:.3f}, "
        f"R: {macro_avg['recall']:.3f}, "
        f"F1: {macro_avg['f1-score']:.3f}\\n"
        f"Weighted Avg - P: {weighted_avg['precision']:.3f}, "
        f"R: {weighted_avg['recall']:.3f}, "
        f"F1: {weighted_avg['f1-score']:.3f}"
    )

    ax.text(
        0.5, -0.15,
        info_text,
        ha='center',
        va='top',
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


# ==================== REGRESSION VISUALIZATIONS ====================

def plot_residuals(
    y_true,
    y_pred,
    figsize: Tuple[int, int] = (15, 5),
    title: str = 'Residual Analysis',
    save_path: Optional[str] = None
):
    """
    Enhanced residual plot with multiple diagnostic plots.

    Parameters:
    -----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.
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
    >>> plot_residuals(y_true, y_pred)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=30)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Residuals', fontsize=11, fontweight='bold')
    axes[0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)

    # Add lowess smoothing line
    try:
        from scipy.signal import savgol_filter
        if len(y_pred) > 50:
            idx_sorted = np.argsort(y_pred)
            smoothed = savgol_filter(residuals[idx_sorted],
                                    min(51, len(y_pred)//2*2-1), 3)
            axes[0].plot(y_pred[idx_sorted], smoothed, 'g-', lw=2, alpha=0.7)
    except (ImportError, TypeError, ValueError):
        pass

    # 2. Histogram of residuals
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residuals', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].set_title(f'Residual Distribution\\nMean: {np.mean(residuals):.4f}, Std: {np.std(residuals):.4f}',
                     fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')

    # 3. Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[2].grid(alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def plot_predictions_vs_actual(
    y_true,
    y_pred,
    figsize: Tuple[int, int] = (10, 8),
    title: str = 'Predictions vs Actual Values',
    save_path: Optional[str] = None,
    interactive: bool = False
):
    """
    Plot predicted vs actual values with perfect prediction line.

    Parameters:
    -----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.
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
    """
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    if interactive:
        import plotly.graph_objects as go

        fig = go.Figure()

        # Scatter plot
        fig.add_trace(go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(size=8, opacity=0.6),
            text=[f'True: {t:.2f}<br>Pred: {p:.2f}<br>Error: {abs(t-p):.2f}'
                  for t, p in zip(y_true, y_pred)],
            hovertemplate='%{text}<extra></extra>'
        ))

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', width=2, dash='dash')
        ))

        fig.update_layout(
            title=f'{title}<br>R²: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}',
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            width=700,
            height=600,
            showlegend=True
        )

        if save_path:
            fig.write_html(save_path)

        fig.show()
        return fig

    else:
        fig, ax = plt.subplots(figsize=figsize)

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=50, edgecolors='k', linewidth=0.5)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5,
               label='Perfect Prediction')

        ax.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        # Add metrics as text
        metrics_text = f'R² = {r2:.4f}\\nMAE = {mae:.4f}\\nRMSE = {rmse:.4f}'
        ax.text(
            0.05, 0.95,
            metrics_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
        return fig


# ==================== TIME SERIES VISUALIZATIONS ====================

def lagged_scatter_plot(
    data,
    variable: str,
    lags: Union[int, List[int]] = 1,
    figsize: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Enhanced lagged scatter plot with multiple lags.

    Parameters:
    -----------
    data : pandas.DataFrame
        Input data.
    variable : str
        Variable name to plot.
    lags : int or list
        Lag(s) to visualize. If list, creates subplot for each lag.
    figsize : tuple, optional
        Figure size.
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save figure.

    Returns:
    --------
    matplotlib.figure.Figure

    Examples:
    ---------
    >>> # Single lag
    >>> lagged_scatter_plot(df, 'sales', lags=1)
    >>>
    >>> # Multiple lags
    >>> lagged_scatter_plot(df, 'sales', lags=[1, 7, 30])
    """
    if isinstance(lags, int):
        lags = [lags]

    n_lags = len(lags)

    if figsize is None:
        figsize = (6 * n_lags, 5)

    fig, axes = plt.subplots(1, n_lags, figsize=figsize)

    if n_lags == 1:
        axes = [axes]

    for idx, lag in enumerate(lags):
        ax = axes[idx]

        x = data[variable].values[:-lag] if lag > 0 else data[variable].values
        y = data[variable].shift(lag).values[:-lag] if lag > 0 else data[variable].values

        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

        # Scatter plot
        ax.scatter(x, y, alpha=0.5, s=30)

        # Calculate correlation
        corr = np.corrcoef(x, y)[0, 1]

        # Add regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "r--", alpha=0.8, lw=2)

        ax.set_xlabel(variable, fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{variable} (lag={lag})', fontsize=11, fontweight='bold')
        ax.set_title(f'Lag {lag} | Correlation: {corr:.3f}',
                    fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    else:
        fig.suptitle(f'Lagged Scatter Plot: {variable}',
                    fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def lag_correlation(
    series,
    lags: int = 50,
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Enhanced autocorrelation and partial autocorrelation plots.

    Parameters:
    -----------
    series : pandas.Series
        Time series data.
    lags : int
        Number of lags to plot.
    figsize : tuple
        Figure size.
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save figure.

    Returns:
    --------
    matplotlib.figure.Figure
    """
    import statsmodels.api as sm

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # ACF
    sm.graphics.tsa.plot_acf(series, lags=lags, ax=axes[0], alpha=0.05)
    axes[0].set_xlabel('Lag', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Autocorrelation', fontsize=11, fontweight='bold')
    axes[0].set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)

    # PACF
    sm.graphics.tsa.plot_pacf(series, lags=lags, ax=axes[1], alpha=0.05)
    axes[1].set_xlabel('Lag', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Partial Autocorrelation', fontsize=11, fontweight='bold')
    axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def seasonal_decompose_plot(
    data,
    freq: int,
    model: str = 'additive',
    figsize: Tuple[int, int] = (14, 10),
    title: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Enhanced seasonal decomposition plot.

    Parameters:
    -----------
    data : pandas.Series
        Time series data with datetime index.
    freq : int
        Frequency of seasonality (e.g., 12 for monthly data with yearly seasonality).
    model : str
        'additive' or 'multiplicative'.
    figsize : tuple
        Figure size.
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save figure.

    Returns:
    --------
    matplotlib.figure.Figure
    """
    import statsmodels.api as sm

    # Decompose
    decomposition = sm.tsa.seasonal_decompose(data, model=model, period=freq)

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # Observed
    decomposition.observed.plot(ax=axes[0], color='blue', linewidth=1.5)
    axes[0].set_ylabel('Observed', fontsize=11, fontweight='bold')
    axes[0].set_title('Observed Data', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)

    # Trend
    decomposition.trend.plot(ax=axes[1], color='green', linewidth=1.5)
    axes[1].set_ylabel('Trend', fontsize=11, fontweight='bold')
    axes[1].set_title('Trend Component', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)

    # Seasonal
    decomposition.seasonal.plot(ax=axes[2], color='orange', linewidth=1.5)
    axes[2].set_ylabel('Seasonal', fontsize=11, fontweight='bold')
    axes[2].set_title('Seasonal Component', fontsize=12, fontweight='bold')
    axes[2].grid(alpha=0.3)

    # Residual
    decomposition.resid.plot(ax=axes[3], color='red', linewidth=1.5)
    axes[3].set_ylabel('Residual', fontsize=11, fontweight='bold')
    axes[3].set_title('Residual Component', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('Time', fontsize=11, fontweight='bold')
    axes[3].grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    else:
        fig.suptitle(f'Seasonal Decomposition ({model.capitalize()} Model)',
                    fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def plot_forecast(
    actual,
    forecast,
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    xlabel: str = 'Time',
    ylabel: str = 'Value',
    title: str = 'Forecast vs Actual',
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
    interactive: bool = False
):
    """
    Enhanced forecast plot with confidence intervals.

    Parameters:
    -----------
    actual : pandas.Series
        Actual time series data.
    forecast : pandas.Series
        Forecasted time series data.
    confidence_intervals : tuple, optional
        (lower_bound, upper_bound) arrays for confidence intervals.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    interactive : bool
        Create interactive Plotly version.

    Returns:
    --------
    matplotlib.figure.Figure or plotly.graph_objs.Figure

    Examples:
    ---------
    >>> # Basic forecast plot
    >>> plot_forecast(actual_series, forecast_series)
    >>>
    >>> # With confidence intervals
    >>> plot_forecast(actual_series, forecast_series,
    ...               confidence_intervals=(lower_ci, upper_ci))
    >>>
    >>> # Interactive version
    >>> plot_forecast(actual_series, forecast_series, interactive=True)
    """
    if interactive:
        import plotly.graph_objects as go

        fig = go.Figure()

        # Actual values
        fig.add_trace(go.Scatter(
            x=actual.index,
            y=actual.values,
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))

        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))

        # Confidence intervals
        if confidence_intervals:
            lower, upper = confidence_intervals
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=upper,
                mode='lines',
                name='Upper CI',
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=lower,
                mode='lines',
                name='Lower CI',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.2)',
                showlegend=True
            ))

        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            width=900,
            height=500,
            hovermode='x unified'
        )

        if save_path:
            fig.write_html(save_path)

        fig.show()
        return fig

    else:
        fig, ax = plt.subplots(figsize=figsize)

        # Actual
        ax.plot(actual.index, actual.values, label='Actual',
               color='blue', linewidth=2)

        # Forecast
        ax.plot(forecast.index, forecast.values, label='Forecast',
               color='red', linewidth=2, linestyle='--')

        # Confidence intervals
        if confidence_intervals:
            lower, upper = confidence_intervals
            ax.fill_between(forecast.index, lower, upper,
                           alpha=0.2, color='red', label='95% CI')

        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
        return fig


def plot_spectral_density(
    data,
    sampling_rate: float,
    figsize: Tuple[int, int] = (12, 6),
    title: str = 'Spectral Density Plot',
    save_path: Optional[str] = None
):
    """
    Enhanced spectral density plot with peak detection.

    Parameters:
    -----------
    data : pandas.Series or array
        Time series data.
    sampling_rate : float
        Sampling rate of the data.
    figsize : tuple
        Figure size.
    title : str
        Plot title.
    save_path : str, optional
        Path to save figure.

    Returns:
    --------
    matplotlib.figure.Figure
    """
    from scipy import signal

    freq, psd = signal.welch(data, fs=sampling_rate)

    fig, ax = plt.subplots(figsize=figsize)

    ax.semilogy(freq, psd, linewidth=2)

    # Find and mark peaks
    peaks, _ = signal.find_peaks(psd, prominence=np.max(psd) * 0.1)
    ax.plot(freq[peaks], psd[peaks], "rx", markersize=10, label='Peaks')

    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power Spectral Density', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)

    # Annotate dominant frequencies
    if len(peaks) > 0:
        top_peaks = peaks[np.argsort(psd[peaks])[-3:]]  # Top 3 peaks
        for peak in top_peaks:
            ax.annotate(
                f'{freq[peak]:.2f} Hz',
                xy=(freq[peak], psd[peak]),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


# ==================== CLUSTERING VISUALIZATIONS ====================

def plot_cluster(
    X,
    y,
    method: str = '2d',
    figsize: Tuple[int, int] = (10, 8),
    title: str = 'Cluster Visualization',
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    interactive: bool = False
):
    """
    Enhanced cluster visualization with multiple methods.

    Parameters:
    -----------
    X : array-like or DataFrame
        Feature data.
    y : array-like
        Cluster labels.
    method : str
        Visualization method: '2d' (PCA), '3d', 'tsne', or 'heatmap'.
    figsize : tuple
        Figure size.
    title : str
        Plot title.
    labels : list, optional
        Cluster labels for legend.
    save_path : str, optional
        Path to save figure.
    interactive : bool
        Create interactive Plotly version (for 2d/3d).

    Returns:
    --------
    matplotlib.figure.Figure or plotly.graph_objs.Figure

    Examples:
    ---------
    >>> # 2D PCA visualization
    >>> plot_cluster(X, cluster_labels, method='2d')
    >>>
    >>> # 3D visualization
    >>> plot_cluster(X, cluster_labels, method='3d', interactive=True)
    >>>
    >>> # t-SNE visualization
    >>> plot_cluster(X, cluster_labels, method='tsne')
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    unique_labels = np.unique(y)
    n_clusters = len(unique_labels)

    if labels is None:
        labels = [f'Cluster {i}' for i in unique_labels]

    if method == '2d':
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)

        if interactive:
            import plotly.express as px

            df_plot = pd.DataFrame({
                'PC1': X_reduced[:, 0],
                'PC2': X_reduced[:, 1],
                'Cluster': [labels[int(label)] for label in y]
            })

            fig = px.scatter(
                df_plot,
                x='PC1',
                y='PC2',
                color='Cluster',
                title=f'{title}<br>Explained Variance: {sum(pca.explained_variance_ratio_):.1%}',
                width=800,
                height=600
            )

            if save_path:
                fig.write_html(save_path)

            fig.show()
            return fig

        else:
            fig, ax = plt.subplots(figsize=figsize)

            colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))

            for idx, (label, color) in enumerate(zip(unique_labels, colors)):
                mask = y == label
                ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                          c=[color], label=labels[idx], s=50, alpha=0.6,
                          edgecolors='k', linewidth=0.5)

            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                         fontsize=12, fontweight='bold')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
                         fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            plt.show()
            return fig

    elif method == '3d':
        from sklearn.decomposition import PCA

        pca = PCA(n_components=3)
        X_reduced = pca.fit_transform(X)

        if interactive:
            import plotly.express as px

            df_plot = pd.DataFrame({
                'PC1': X_reduced[:, 0],
                'PC2': X_reduced[:, 1],
                'PC3': X_reduced[:, 2],
                'Cluster': [labels[int(label)] for label in y]
            })

            fig = px.scatter_3d(
                df_plot,
                x='PC1',
                y='PC2',
                z='PC3',
                color='Cluster',
                title=title,
                width=800,
                height=700
            )

            if save_path:
                fig.write_html(save_path)

            fig.show()
            return fig

        else:

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')

            colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))

            for idx, (label, color) in enumerate(zip(unique_labels, colors)):
                mask = y == label
                ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1], X_reduced[mask, 2],
                          c=[color], label=labels[idx], s=50, alpha=0.6,
                          edgecolors='k', linewidth=0.5)

            ax.set_xlabel('PC1', fontsize=11, fontweight='bold')
            ax.set_ylabel('PC2', fontsize=11, fontweight='bold')
            ax.set_zlabel('PC3', fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            plt.show()
            return fig

    elif method == 'tsne':
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
        X_reduced = tsne.fit_transform(X)

        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))

        for idx, (label, color) in enumerate(zip(unique_labels, colors)):
            mask = y == label
            ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                      c=[color], label=labels[idx], s=50, alpha=0.6,
                      edgecolors='k', linewidth=0.5)

        ax.set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
        return fig

    elif method == 'heatmap':
        fig, ax = plt.subplots(figsize=figsize)

        # Sort by cluster
        sorted_indices = np.argsort(y)
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]

        sns.heatmap(X_sorted.T, cmap='viridis', cbar=True, ax=ax,
                   yticklabels=False, xticklabels=False)

        # Add cluster boundaries
        cluster_boundaries = []
        current_cluster = y_sorted[0]
        for i, cluster in enumerate(y_sorted):
            if cluster != current_cluster:
                cluster_boundaries.append(i)
                current_cluster = cluster

        for boundary in cluster_boundaries:
            ax.axvline(x=boundary, color='red', linewidth=2, linestyle='--')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Samples (sorted by cluster)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Features', fontsize=12, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
        return fig


def silhouette(
    X,
    y_pred,
    figsize: Tuple[int, int] = (10, 7),
    title: str = 'Silhouette Analysis',
    save_path: Optional[str] = None
):
    """
    Enhanced silhouette plot.

    Parameters:
    -----------
    X : array-like
        Feature data.
    y_pred : array-like
        Cluster predictions.
    figsize : tuple
        Figure size.
    title : str
        Plot title.
    save_path : str, optional
        Path to save figure.

    Returns:
    --------
    matplotlib.figure.Figure
    """
    from sklearn.metrics import silhouette_samples, silhouette_score

    cluster_labels = np.unique(y_pred)
    n_clusters = len(cluster_labels)
    silhouette_vals = silhouette_samples(X, y_pred)
    silhouette_avg = silhouette_score(X, y_pred)

    fig, ax = plt.subplots(figsize=figsize)

    y_ax_lower, y_ax_upper = 0, 0
    yticks = []

    colors = plt.cm.Spectral(np.linspace(0, 1, n_clusters))

    for i, (cluster, color) in enumerate(zip(cluster_labels, colors)):
        c_silhouette_vals = silhouette_vals[y_pred == cluster]
        c_silhouette_vals.sort()

        y_ax_upper += len(c_silhouette_vals)

        ax.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals,
               height=1.0, edgecolor='none', color=color,
               label=f'Cluster {cluster} (n={len(c_silhouette_vals)})')

        yticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouette_vals)

    # Average silhouette score line
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2,
              label=f'Average Score: {silhouette_avg:.3f}')

    ax.set_xlabel('Silhouette Coefficient', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_title(f'{title}\\nAverage Silhouette Score: {silhouette_avg:.3f}',
                fontsize=14, fontweight='bold')
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{int(c)}' for c in cluster_labels])
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def dendrogram(
    X,
    method: str = 'ward',
    figsize: Tuple[int, int] = (14, 7),
    title: str = 'Hierarchical Clustering Dendrogram',
    save_path: Optional[str] = None,
    truncate_mode: Optional[str] = None,
    p: int = 30
):
    """
    Enhanced dendrogram with truncation options.

    Parameters:
    -----------
    X : array-like
        Data to be clustered.
    method : str
        Linkage method ('ward', 'complete', 'average', 'single').
    figsize : tuple
        Figure size.
    title : str
        Plot title.
    save_path : str, optional
        Path to save figure.
    truncate_mode : str, optional
        Truncation mode: 'lastp', 'level', or None.
    p : int
        Truncation parameter.

    Returns:
    --------
    matplotlib.figure.Figure
    """
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram, linkage

    # Calculate linkage matrix
    Z = linkage(X, method=method)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot dendrogram
    if truncate_mode:
        scipy_dendrogram(
            Z,
            ax=ax,
            truncate_mode=truncate_mode,
            p=p,
            leaf_font_size=10,
            show_leaf_counts=True
        )
    else:
        scipy_dendrogram(
            Z,
            ax=ax,
            leaf_font_size=8
        )

    ax.set_xlabel('Data Points / Cluster Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance', fontsize=12, fontweight='bold')
    ax.set_title(f'{title} (Method: {method.capitalize()})',
                fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def plot_elbow(
    X,
    max_k: int = 10,
    figsize: Tuple[int, int] = (10, 6),
    title: str = 'Elbow Method for Optimal K',
    save_path: Optional[str] = None,
    show_knee: bool = True
):
    """
    Enhanced elbow plot with knee detection.

    Parameters:
    -----------
    X : array-like
        Data to cluster.
    max_k : int
        Maximum number of clusters to test.
    figsize : tuple
        Figure size.
    title : str
        Plot title.
    save_path : str, optional
        Path to save figure.
    show_knee : bool
        Detect and show the knee point.

    Returns:
    --------
    matplotlib.figure.Figure
    """
    from sklearn.cluster import KMeans

    sse = []
    K_range = range(1, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(K_range, sse, marker='o', linewidth=2.5, markersize=8, color='blue')

    # Detect knee using the "elbow" method (second derivative)
    if show_knee and len(sse) > 2:
        try:
            from kneed import KneeLocator
            kl = KneeLocator(list(K_range), sse, curve='convex', direction='decreasing')
            if kl.elbow:
                ax.axvline(x=kl.elbow, color='red', linestyle='--', linewidth=2,
                          label=f'Optimal K = {kl.elbow}')
                ax.plot(kl.elbow, sse[kl.elbow-1], 'ro', markersize=12,
                       markeredgewidth=2, markeredgecolor='black')
        except ImportError:
            # Simple alternative: find maximum second derivative
            if len(sse) >= 3:
                second_deriv = np.diff(np.diff(sse))
                knee_idx = np.argmax(second_deriv) + 2
                ax.axvline(x=knee_idx, color='red', linestyle='--', linewidth=2,
                          label=f'Suggested K ≈ {knee_idx}')

    ax.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Within-Cluster Sum of Squares (SSE)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)

    # Annotate points
    for k, value in zip(K_range, sse):
        ax.annotate(f'{value:.0f}', xy=(k, value), xytext=(0, 10),
                   textcoords='offset points', ha='center', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def plot_pca(
    df,
    target_col: Optional[str] = None,
    n_components: int = 2,
    figsize: Tuple[int, int] = (10, 8),
    title: str = 'PCA Visualization',
    save_path: Optional[str] = None,
    show_loadings: bool = False,
    interactive: bool = False
):
    """
    Enhanced PCA visualization with variance explanation and loadings.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data.
    target_col : str, optional
        Name of target column for coloring.
    n_components : int
        Number of PCA components (2 or 3).
    figsize : tuple
        Figure size.
    title : str
        Plot title.
    save_path : str, optional
        Path to save figure.
    show_loadings : bool
        Show feature loadings as arrows.
    interactive : bool
        Create interactive Plotly version.

    Returns:
    --------
    matplotlib.figure.Figure or plotly.graph_objs.Figure
    """
    from sklearn.decomposition import PCA

    # Prepare data
    if target_col:
        X = df.drop(target_col, axis=1).values
        target = df[target_col].values
        feature_names = df.drop(target_col, axis=1).columns
    else:
        X = df.values
        target = None
        feature_names = df.columns

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    if n_components == 2:
        if interactive:
            import plotly.express as px

            df_plot = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1]
            })

            if target_col:
                df_plot['Target'] = target
                fig = px.scatter(
                    df_plot, x='PC1', y='PC2', color='Target',
                    title=f'{title}<br>Explained Variance: {sum(pca.explained_variance_ratio_):.1%}',
                    labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                           'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'},
                    width=800,
                    height=600
                )
            else:
                fig = px.scatter(
                    df_plot, x='PC1', y='PC2',
                    title=f'{title}<br>Explained Variance: {sum(pca.explained_variance_ratio_):.1%}',
                    width=800,
                    height=600
                )

            if save_path:
                fig.write_html(save_path)

            fig.show()
            return fig

        else:
            fig, ax = plt.subplots(figsize=figsize)

            if target_col:
                unique_targets = np.unique(target)
                colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_targets)))

                for target_val, color in zip(unique_targets, colors):
                    mask = target == target_val
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                             c=[color], label=str(target_val),
                             s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
                ax.legend(title=target_col, fontsize=10)
            else:
                ax.scatter(X_pca[:, 0], X_pca[:, 1], s=50, alpha=0.6,
                          edgecolors='k', linewidth=0.5)

            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                         fontsize=12, fontweight='bold')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
                         fontsize=12, fontweight='bold')
            ax.set_title(f'{title}\\nTotal Variance Explained: {sum(pca.explained_variance_ratio_):.1%}',
                        fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)

            # Show loadings
            if show_loadings:
                loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                for i, feature in enumerate(feature_names[:10]):  # Show top 10
                    ax.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                            head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.5)
                    ax.text(loadings[i, 0] * 1.1, loadings[i, 1] * 1.1, feature,
                           fontsize=8, ha='center', va='center')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            plt.show()
            return fig

    elif n_components == 3:
        if interactive:
            import plotly.express as px

            df_plot = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'PC3': X_pca[:, 2]
            })

            if target_col:
                df_plot['Target'] = target
                fig = px.scatter_3d(
                    df_plot, x='PC1', y='PC2', z='PC3', color='Target',
                    title=title,
                    width=800,
                    height=700
                )
            else:
                fig = px.scatter_3d(
                    df_plot, x='PC1', y='PC2', z='PC3',
                    title=title,
                    width=800,
                    height=700
                )

            if save_path:
                fig.write_html(save_path)

            fig.show()
            return fig


# ==================== KERAS/TENSORFLOW VISUALIZATIONS ====================

def loss_plot(
    history_or_model,
    metrics: List[str] = ['loss', 'val_loss'],
    start: int = 0,
    figsize: Tuple[int, int] = (12, 6),
    title: str = 'Training History',
    save_path: Optional[str] = None
):
    """
    Enhanced training history plot for Keras/TensorFlow models.

    Parameters:
    -----------
    history_or_model : keras.callbacks.History or keras.Model
        Training history or model with history attribute.
    metrics : list
        Metrics to plot.
    start : int
        Starting epoch to plot.
    figsize : tuple
        Figure size.
    title : str
        Plot title.
    save_path : str, optional
        Path to save figure.

    Returns:
    --------
    matplotlib.figure.Figure
    """
    # Get history
    if hasattr(history_or_model, 'history'):
        history_dict = history_or_model.history
    else:
        history_dict = history_or_model

    # Convert to DataFrame
    history_df = pd.DataFrame(history_dict)

    # Print best values
    if 'loss' in history_df.columns:
        print(f"[OK] Minimum loss: {history_df['loss'].min():.6f} at epoch {history_df['loss'].idxmin() + 1}")
    if 'val_loss' in history_df.columns:
        print(f"[OK] Minimum validation loss: {history_df['val_loss'].min():.6f} at epoch {history_df['val_loss'].idxmin() + 1}")

    # Determine number of subplots
    unique_metrics = set()
    for metric in metrics:
        base_metric = metric.replace('val_', '')
        unique_metrics.add(base_metric)

    n_plots = len(unique_metrics)

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    # Plot each metric
    for idx, base_metric in enumerate(unique_metrics):
        ax = axes[idx]

        # Plot training metric
        if base_metric in history_df.columns:
            ax.plot(history_df.index[start:], history_df[base_metric].values[start:],
                   label=f'Train {base_metric}', linewidth=2)

        # Plot validation metric
        val_metric = f'val_{base_metric}'
        if val_metric in history_df.columns:
            ax.plot(history_df.index[start:], history_df[val_metric].values[start:],
                   label=f'Val {base_metric}', linewidth=2, linestyle='--')

            # Mark best epoch
            best_epoch = history_df[val_metric][start:].idxmin()
            best_value = history_df[val_metric][start:].min()
            ax.plot(best_epoch, best_value, 'r*', markersize=15,
                   label=f'Best (epoch {best_epoch + 1})')

        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel(base_metric.capitalize(), fontsize=11, fontweight='bold')
        ax.set_title(f'{base_metric.capitalize()} Over Time', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig
