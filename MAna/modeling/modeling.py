from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import joblib
import json
from pathlib import Path
import time
from dataclasses import dataclass, field
from datetime import datetime
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # PyTorch is an optional modeling dependency.
    torch = None
    nn = None
    optim = None
    DataLoader = Any
    TensorDataset = None


def _require_pytorch() -> None:
    """Raise an actionable error when a PyTorch-only helper is requested."""
    if torch is None or nn is None or optim is None or TensorDataset is None:
        raise ImportError(
            "PyTorch helpers require the optional modeling dependencies. "
            "Install them with `pip install M_Ana_package[modeling]`."
        )


_TorchModuleBase = nn.Module if nn is not None else object


# ==================== MODEL REGISTRY & VERSIONING ====================

@dataclass
class ModelMetadata:
    """Metadata for model tracking and versioning."""

    model_name: str
    version: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    framework: str = 'unknown'  # 'sklearn', 'tensorflow', 'pytorch', etc.
    task_type: str = 'unknown'  # 'classification', 'regression', 'clustering'
    metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_time: float = 0.0
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    feature_names: Optional[List[str]] = None
    tags: List[str] = field(default_factory=list)
    description: str = ''

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'version': self.version,
            'created_at': self.created_at,
            'framework': self.framework,
            'task_type': self.task_type,
            'metrics': self.metrics,
            'hyperparameters': self.hyperparameters,
            'training_time': self.training_time,
            'dataset_info': self.dataset_info,
            'feature_names': self.feature_names,
            'tags': self.tags,
            'description': self.description
        }

    def to_json(self, filepath: Optional[str] = None) -> str:
        """Export metadata as JSON."""
        json_str = json.dumps(self.to_dict(), indent=2, default=str)
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
        return json_str


class ModelRegistry:
    """
    Central registry for model management, versioning, and deployment.

    Features:
    - Model versioning
    - Metadata tracking
    - Model comparison
    - Easy deployment
    - Rollback capabilities

    Examples:
    ---------
    >>> # Initialize registry
    >>> registry = ModelRegistry(base_dir='models/')
    >>>
    >>> # Save model with metadata
    >>> registry.save_model(
    ...     model=trained_model,
    ...     model_name='customer_churn',
    ...     version='v1.0',
    ...     metrics={'accuracy': 0.95, 'f1': 0.93},
    ...     hyperparameters={'n_estimators': 100, 'max_depth': 10},
    ...     description='Initial production model'
    ... )
    >>>
    >>> # Load latest model
    >>> model = registry.load_model('customer_churn', version='latest')
    >>>
    >>> # Compare model versions
    >>> comparison = registry.compare_models('customer_churn', ['v1.0', 'v2.0'])
    """

    def __init__(self, base_dir: str = 'model_registry'):
        """
        Initialize ModelRegistry.

        Parameters:
        -----------
        base_dir : str
            Base directory for storing models.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.base_dir / 'registry.json'
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """Load registry from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_registry(self):
        """Save registry to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)

    def save_model(
        self,
        model: Any,
        model_name: str,
        version: str = 'v1.0',
        framework: str = 'sklearn',
        task_type: str = 'classification',
        metrics: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        description: str = '',
        **kwargs
    ) -> str:
        """
        Save model with comprehensive metadata.

        Parameters:
        -----------
        model : any
            Trained model object.
        model_name : str
            Name of the model.
        version : str
            Version identifier.
        framework : str
            Framework used ('sklearn', 'tensorflow', 'pytorch', 'xgboost', etc.).
        task_type : str
            Type of task ('classification', 'regression', 'clustering').
        metrics : dict, optional
            Performance metrics.
        hyperparameters : dict, optional
            Model hyperparameters.
        feature_names : list, optional
            Names of input features.
        tags : list, optional
            Tags for categorization.
        description : str
            Model description.

        Returns:
        --------
        str
            Path to saved model.
        """
        # Create model directory
        model_dir = self.base_dir / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model based on framework
        model_path = model_dir / 'model'

        if framework == 'tensorflow' or framework == 'keras':
            # Save Keras/TensorFlow model
            model.save(str(model_path))
        elif framework == 'pytorch':
            # Save PyTorch model
            import torch
            torch.save(model.state_dict(), str(model_path) + '.pt')
        else:
            # Save sklearn-compatible model (default)
            joblib.dump(model, str(model_path) + '.pkl')

        # Create metadata
        metadata = ModelMetadata(
            model_name=model_name,
            version=version,
            framework=framework,
            task_type=task_type,
            metrics=metrics or {},
            hyperparameters=hyperparameters or {},
            feature_names=feature_names,
            tags=tags or [],
            description=description,
            **kwargs
        )

        # Save metadata
        metadata.to_json(str(model_dir / 'metadata.json'))

        # Update registry
        if model_name not in self.registry:
            self.registry[model_name] = {}

        self.registry[model_name][version] = {
            'path': str(model_path),
            'metadata': metadata.to_dict()
        }

        self._save_registry()

        version_label = str(version) if str(version).lower().startswith("v") else f"v{version}"
        print(f"[OK] Model saved: {model_name} {version_label}")
        print(f"  Path: {model_path}")
        print(f"  Framework: {framework}")
        if metrics:
            print(f"  Metrics: {metrics}")

        return str(model_path)

    def load_model(
        self,
        model_name: str,
        version: str = 'latest',
        framework: Optional[str] = None
    ) -> Any:
        """
        Load model from registry.

        Parameters:
        -----------
        model_name : str
            Name of the model.
        version : str
            Version to load ('latest' or specific version).
        framework : str, optional
            Framework (auto-detected if None).

        Returns:
        --------
        model
            Loaded model object.
        """
        if model_name not in self.registry:
            raise ValueError(f"Model '{model_name}' not found in registry")

        # Get version
        if version == 'latest':
            versions = list(self.registry[model_name].keys())
            version = sorted(versions)[-1]

        if version not in self.registry[model_name]:
            raise ValueError(f"Version '{version}' not found for model '{model_name}'")

        model_info = self.registry[model_name][version]
        model_path = model_info['path']
        metadata = model_info['metadata']

        # Auto-detect framework
        if framework is None:
            framework = metadata.get('framework', 'sklearn')

        # Load model based on framework
        if framework == 'tensorflow' or framework == 'keras':
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
        elif framework == 'pytorch':
            import torch
            model = torch.load(model_path + '.pt')
        else:
            # Load sklearn-compatible model
            model = joblib.load(model_path + '.pkl')

        version_label = str(version) if str(version).lower().startswith("v") else f"v{version}"
        print(f"[OK] Model loaded: {model_name} {version_label}")
        print(f"  Framework: {framework}")
        if metadata.get('metrics'):
            print(f"  Metrics: {metadata['metrics']}")

        return model

    def list_models(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """
        List all models in registry.

        Parameters:
        -----------
        model_name : str, optional
            Filter by model name.

        Returns:
        --------
        pd.DataFrame
            DataFrame with model information.
        """
        data = []

        models_to_list = [model_name] if model_name else self.registry.keys()

        for name in models_to_list:
            if name not in self.registry:
                continue

            for version, info in self.registry[name].items():
                metadata = info['metadata']
                data.append({
                    'model_name': name,
                    'version': version,
                    'framework': metadata.get('framework'),
                    'task_type': metadata.get('task_type'),
                    'created_at': metadata.get('created_at'),
                    **{f'metric_{k}': v for k, v in metadata.get('metrics', {}).items()}
                })

        return pd.DataFrame(data)

    def compare_models(
        self,
        model_name: str,
        versions: Optional[List[str]] = None,
        metric: str = 'accuracy'
    ) -> pd.DataFrame:
        """
        Compare different versions of a model.

        Parameters:
        -----------
        model_name : str
            Name of the model.
        versions : list, optional
            Versions to compare (all if None).
        metric : str
            Primary metric for comparison.

        Returns:
        --------
        pd.DataFrame
            Comparison table.
        """
        if model_name not in self.registry:
            raise ValueError(f"Model '{model_name}' not found")

        versions_to_compare = versions or list(self.registry[model_name].keys())

        data = []
        for version in versions_to_compare:
            if version not in self.registry[model_name]:
                continue

            metadata = self.registry[model_name][version]['metadata']
            metrics = metadata.get('metrics', {})

            data.append({
                'version': version,
                'created_at': metadata.get('created_at'),
                **metrics
            })

        df = pd.DataFrame(data)

        # Sort by primary metric if available
        if metric in df.columns:
            df = df.sort_values(metric, ascending=False)

        return df

    def delete_model(self, model_name: str, version: str):
        """Delete a model version."""
        if model_name in self.registry and version in self.registry[model_name]:
            model_dir = self.base_dir / model_name / version

            # Delete files
            import shutil
            if model_dir.exists():
                shutil.rmtree(model_dir)

            # Update registry
            del self.registry[model_name][version]
            if not self.registry[model_name]:
                del self.registry[model_name]

            self._save_registry()
            version_label = str(version) if str(version).lower().startswith("v") else f"v{version}"
            print(f"[OK] Deleted: {model_name} {version_label}")
        else:
            print(f"Model {model_name} v{version} not found")


# ==================== KERAS/TENSORFLOW UTILITIES ====================

def early_stopping(
    monitor: str = 'val_loss',
    min_delta: float = 0.001,
    patience: int = 20,
    mode: str = 'min',
    restore_best_weights: bool = True,
    verbose: int = 1
):
    """
    Create early stopping callback for Keras/TensorFlow models.

    Parameters:
    -----------
    monitor : str
        Metric to monitor ('val_loss', 'val_accuracy', etc.).
    min_delta : float
        Minimum change to qualify as an improvement.
    patience : int
        Number of epochs with no improvement before stopping.
    mode : str
        'min' for loss, 'max' for accuracy.
    restore_best_weights : bool
        Restore weights from best epoch.
    verbose : int
        Verbosity level.

    Returns:
    --------
    EarlyStopping callback

    Examples:
    ---------
    >>> from tensorflow.keras.models import Sequential
    >>> from tensorflow.keras.layers import Dense
    >>>
    >>> model = Sequential([Dense(64, activation='relu'), Dense(1)])
    >>> model.compile(optimizer='adam', loss='mse')
    >>>
    >>> # Use early stopping
    >>> callbacks = [early_stopping(patience=10)]
    >>> model.fit(X_train, y_train, validation_split=0.2,
    ...           epochs=100, callbacks=callbacks)
    """
    from tensorflow.keras import callbacks

    return callbacks.EarlyStopping(
        monitor=monitor,
        min_delta=min_delta,
        patience=patience,
        mode=mode,
        restore_best_weights=restore_best_weights,
        verbose=verbose
    )


def reduce_lr_on_plateau(
    monitor: str = 'val_loss',
    factor: float = 0.5,
    patience: int = 10,
    min_lr: float = 1e-7,
    mode: str = 'min',
    verbose: int = 1
):
    """
    Reduce learning rate when metric plateaus.

    Parameters:
    -----------
    monitor : str
        Metric to monitor.
    factor : float
        Factor by which to reduce learning rate.
    patience : int
        Number of epochs with no improvement.
    min_lr : float
        Minimum learning rate.
    mode : str
        'min' or 'max'.
    verbose : int
        Verbosity level.

    Returns:
    --------
    ReduceLROnPlateau callback
    """
    from tensorflow.keras import callbacks

    return callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        mode=mode,
        verbose=verbose
    )


def model_checkpoint(
    filepath: str,
    monitor: str = 'val_loss',
    save_best_only: bool = True,
    mode: str = 'min',
    verbose: int = 1
):
    """
    Save model checkpoints during training.

    Parameters:
    -----------
    filepath : str
        Path to save model (can include formatting like 'model_{epoch:02d}.h5').
    monitor : str
        Metric to monitor.
    save_best_only : bool
        Only save when model improves.
    mode : str
        'min' or 'max'.
    verbose : int
        Verbosity level.

    Returns:
    --------
    ModelCheckpoint callback
    """
    from tensorflow.keras import callbacks

    return callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor=monitor,
        save_best_only=save_best_only,
        mode=mode,
        verbose=verbose
    )


def get_training_callbacks(
    model_name: str = 'model',
    early_stop_patience: int = 20,
    reduce_lr_patience: int = 10,
    save_checkpoints: bool = True,
    tensorboard: bool = False,
    log_dir: str = './logs'
) -> List:
    """
    Get a comprehensive set of training callbacks.

    Parameters:
    -----------
    model_name : str
        Name for saving checkpoints.
    early_stop_patience : int
        Patience for early stopping.
    reduce_lr_patience : int
        Patience for learning rate reduction.
    save_checkpoints : bool
        Save model checkpoints.
    tensorboard : bool
        Enable TensorBoard logging.
    log_dir : str
        Directory for logs.

    Returns:
    --------
    list
        List of callbacks.

    Examples:
    ---------
    >>> callbacks = get_training_callbacks(
    ...     model_name='my_model',
    ...     early_stop_patience=15,
    ...     tensorboard=True
    ... )
    >>> model.fit(X_train, y_train, epochs=100, callbacks=callbacks)
    """
    from tensorflow.keras import callbacks as keras_callbacks

    callbacks_list = []

    # Early stopping
    callbacks_list.append(early_stopping(patience=early_stop_patience))

    # Reduce learning rate
    callbacks_list.append(reduce_lr_on_plateau(patience=reduce_lr_patience))

    # Model checkpoint
    if save_checkpoints:
        callbacks_list.append(model_checkpoint(
            filepath=f'{model_name}_best.h5',
            save_best_only=True
        ))

    # TensorBoard
    if tensorboard:
        callbacks_list.append(keras_callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        ))

    return callbacks_list


def save_keras_model(
    model,
    model_name: str,
    save_format: str = 'tf',
    save_weights_only: bool = False
):
    """
    Save Keras/TensorFlow model with multiple formats.

    Parameters:
    -----------
    model : keras.Model
        Trained model.
    model_name : str
        Name for saving (without extension).
    save_format : str
        Format: 'tf' (SavedModel), 'h5', or 'json'.
    save_weights_only : bool
        Save only weights (for 'h5' format).

    Examples:
    ---------
    >>> # Save as TensorFlow SavedModel (recommended)
    >>> save_keras_model(model, 'my_model', save_format='tf')
    >>>
    >>> # Save as HDF5
    >>> save_keras_model(model, 'my_model', save_format='h5')
    >>>
    >>> # Save architecture and weights separately
    >>> save_keras_model(model, 'my_model', save_format='json')
    """
    if save_format == 'tf':
        # TensorFlow SavedModel format (recommended)
        model.save(model_name)
        print(f"[OK] Saved model to {model_name}/ (TensorFlow SavedModel format)")

    elif save_format == 'h5':
        # HDF5 format
        if save_weights_only:
            model.save_weights(f"{model_name}.h5")
            print(f"[OK] Saved weights to {model_name}.h5")
        else:
            model.save(f"{model_name}.h5")
            print(f"[OK] Saved model to {model_name}.h5")

    elif save_format == 'json':
        # JSON + weights
        model_json = model.to_json()
        with open(f"{model_name}.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(f"{model_name}_weights.h5")
        print(f"[OK] Saved architecture to {model_name}.json")
        print(f"[OK] Saved weights to {model_name}_weights.h5")

    else:
        raise ValueError(f"Unsupported format: {save_format}")


def load_keras_model(
    model_name: str,
    load_format: str = 'auto',
    custom_objects: Optional[Dict] = None
):
    """
    Load Keras/TensorFlow model.

    Parameters:
    -----------
    model_name : str
        Name of saved model.
    load_format : str
        Format: 'auto', 'tf', 'h5', or 'json'.
    custom_objects : dict, optional
        Custom layers/functions.

    Returns:
    --------
    keras.Model
        Loaded model.

    Examples:
    ---------
    >>> # Auto-detect format
    >>> model = load_keras_model('my_model')
    >>>
    >>> # Load from JSON
    >>> model = load_keras_model('my_model', load_format='json')
    >>>
    >>> # With custom objects
    >>> model = load_keras_model('my_model', custom_objects={'CustomLayer': CustomLayer})
    """
    from tensorflow.keras.models import load_model, model_from_json
    from pathlib import Path

    # Auto-detect format
    if load_format == 'auto':
        if Path(model_name).is_dir():
            load_format = 'tf'
        elif Path(f"{model_name}.h5").exists():
            load_format = 'h5'
        elif Path(f"{model_name}.json").exists():
            load_format = 'json'
        else:
            raise ValueError(f"Could not find model: {model_name}")

    if load_format == 'tf' or load_format == 'h5':
        model = load_model(
            model_name if load_format == 'tf' else f"{model_name}.h5",
            custom_objects=custom_objects
        )
        print(f"[OK] Loaded model from {model_name}")

    elif load_format == 'json':
        # Load from JSON
        with open(f'{model_name}.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json, custom_objects=custom_objects)
        model.load_weights(f"{model_name}_weights.h5")
        print(f"[OK] Loaded model from {model_name}.json")

    else:
        raise ValueError(f"Unsupported format: {load_format}")

    return model


# ==================== SKLEARN MODEL UTILITIES ====================

def save_sklearn_model(
    model,
    filename: str,
    compress: int = 3
):
    """
    Save scikit-learn model efficiently.

    Parameters:
    -----------
    model : sklearn model
        Trained model.
    filename : str
        Output filename (without extension).
    compress : int
        Compression level (0-9, higher = more compression).

    Examples:
    ---------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier().fit(X, y)
    >>> save_sklearn_model(model, 'rf_model')
    """
    joblib.dump(model, f"{filename}.pkl", compress=compress)
    print(f"[OK] Saved model to {filename}.pkl")


def load_sklearn_model(filename: str):
    """
    Load scikit-learn model.

    Parameters:
    -----------
    filename : str
        Input filename (with or without .pkl extension).

    Returns:
    --------
    model
        Loaded sklearn model.
    """
    if not filename.endswith('.pkl'):
        filename += '.pkl'

    model = joblib.load(filename)
    print(f"[OK] Loaded model from {filename}")
    return model


# ==================== AUTO ML ====================

def auto_classifier(
    X_train,
    y_train,
    X_test,
    y_test,
    time_limit: int = 300,
    metric: str = 'accuracy',
    n_jobs: int = -1,
    verbose: bool = True
) -> Tuple[Any, Dict]:
    """
    Automatic classifier selection and hyperparameter tuning.

    Tries multiple algorithms and finds the best one automatically.

    Parameters:
    -----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
    X_test : array-like
        Test features.
    y_test : array-like
        Test labels.
    time_limit : int
        Time limit in seconds.
    metric : str
        Metric to optimize ('accuracy', 'f1', 'roc_auc', etc.).
    n_jobs : int
        Number of parallel jobs.
    verbose : bool
        Print progress.

    Returns:
    --------
    tuple
        (best_model, results_dict)

    Examples:
    ---------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>>
    >>> best_model, results = auto_classifier(
    ...     X_train, y_train, X_test, y_test,
    ...     time_limit=300
    ... )
    >>> print(f"Best model: {type(best_model).__name__}")
    >>> print(f"Test accuracy: {results['test_score']:.4f}")
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    # Define candidate models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=n_jobs),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Naive Bayes': GaussianNB(),
        'K-Nearest Neighbors': KNeighborsClassifier(n_jobs=n_jobs),
        'XGBoost': XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42, n_jobs=n_jobs)
    }

    results = {}
    best_score = -np.inf
    best_model = None
    best_model_name = None

    start_time = time.time()

    for name, model in models.items():
        if time.time() - start_time > time_limit:
            if verbose:
                print("[WARN] Time limit reached")
            break

        if verbose:
            print(f"Training {name}...")

        try:
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=5,
                scoring=metric, n_jobs=n_jobs
            )
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            # Train on full training set
            model.fit(X_train, y_train)

            # Test score
            y_pred = model.predict(X_test)

            if metric == 'accuracy':
                test_score = accuracy_score(y_test, y_pred)
            elif metric == 'f1':
                test_score = f1_score(y_test, y_pred, average='weighted')
            elif metric == 'roc_auc':
                y_prob = model.predict_proba(X_test)[:, 1]
                test_score = roc_auc_score(y_test, y_prob)
            else:
                test_score = accuracy_score(y_test, y_pred)

            results[name] = {
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'test_score': test_score,
                'model': model
            }

            if verbose:
                print(f"  CV: {cv_mean:.4f} +/- {cv_std:.4f} | Test: {test_score:.4f}")

            # Track best model
            if test_score > best_score:
                best_score = test_score
                best_model = model
                best_model_name = name

        except Exception as e:
            if verbose:
                print(f"  [FAILED] {e}")
            continue

    if verbose:
        print(f"\\n[OK] Best Model: {best_model_name}")
        print(f"   Score: {best_score:.4f}")

    return best_model, {
        'best_model_name': best_model_name,
        'best_score': best_score,
        'all_results': results
    }


def auto_regressor(
    X_train,
    y_train,
    X_test,
    y_test,
    time_limit: int = 300,
    metric: str = 'r2',
    n_jobs: int = -1,
    verbose: bool = True
) -> Tuple[Any, Dict]:
    """
    Automatic regressor selection and training.

    Parameters:
    -----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training targets.
    X_test : array-like
        Test features.
    y_test : array-like
        Test targets.
    time_limit : int
        Time limit in seconds.
    metric : str
        Metric to optimize ('r2', 'neg_mean_squared_error', etc.).
    n_jobs : int
        Number of parallel jobs.
    verbose : bool
        Print progress.

    Returns:
    --------
    tuple
        (best_model, results_dict)
    """
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import r2_score, mean_squared_error

    models = {
        'Linear Regression': LinearRegression(n_jobs=n_jobs),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=n_jobs),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(),
        'K-Nearest Neighbors': KNeighborsRegressor(n_jobs=n_jobs),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=n_jobs)
    }

    results = {}
    best_score = -np.inf
    best_model = None
    best_model_name = None

    start_time = time.time()

    for name, model in models.items():
        if time.time() - start_time > time_limit:
            if verbose:
                print("[WARN] Time limit reached")
            break

        if verbose:
            print(f"Training {name}...")

        try:
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=5,
                scoring=metric, n_jobs=n_jobs
            )
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            # Train and test
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if metric == 'r2':
                test_score = r2_score(y_test, y_pred)
            elif metric == 'neg_mean_squared_error':
                test_score = -mean_squared_error(y_test, y_pred)
            else:
                test_score = r2_score(y_test, y_pred)

            results[name] = {
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'test_score': test_score,
                'model': model
            }

            if verbose:
                print(f"  CV: {cv_mean:.4f} +/- {cv_std:.4f} | Test: {test_score:.4f}")

            if test_score > best_score:
                best_score = test_score
                best_model = model
                best_model_name = name

        except Exception as e:
            if verbose:
                print(f"  [FAILED] {e}")
            continue

    if verbose:
        print(f"\\n[OK] Best Model: {best_model_name}")
        print(f"   Score: {best_score:.4f}")

    return best_model, {
        'best_model_name': best_model_name,
        'best_score': best_score,
        'all_results': results
    }


# ==================== HYPERPARAMETER TUNING ====================

def hyperparameter_search(
    model,
    param_grid: Dict,
    X,
    y,
    search_type: str = 'grid',
    cv: int = 5,
    scoring: str = 'accuracy',
    n_iter: int = 50,
    n_jobs: int = -1,
    verbose: int = 1
):
    """
    Comprehensive hyperparameter search (Grid or Random).

    Parameters:
    -----------
    model : sklearn model
        Model to tune.
    param_grid : dict
        Parameter grid.
    X : array-like
        Features.
    y : array-like
        Target.
    search_type : str
        'grid' for GridSearchCV or 'random' for RandomizedSearchCV.
    cv : int
        Cross-validation folds.
    scoring : str
        Scoring metric.
    n_iter : int
        Number of iterations (for random search).
    n_jobs : int
        Parallel jobs.
    verbose : int
        Verbosity level.

    Returns:
    --------
    tuple
        (best_model, search_results)

    Examples:
    ---------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>>
    >>> model = RandomForestClassifier()
    >>> param_grid = {
    ...     'n_estimators': [50, 100, 200],
    ...     'max_depth': [10, 20, 30, None],
    ...     'min_samples_split': [2, 5, 10]
    ... }
    >>>
    >>> best_model, results = hyperparameter_search(
    ...     model, param_grid, X, y,
    ...     search_type='grid'
    ... )
    """
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    if search_type == 'grid':
        search = GridSearchCV(
            model, param_grid, cv=cv,
            scoring=scoring, n_jobs=n_jobs,
            verbose=verbose
        )
    elif search_type == 'random':
        search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter,
            cv=cv, scoring=scoring, n_jobs=n_jobs,
            verbose=verbose, random_state=42
        )
    else:
        raise ValueError(f"Unknown search_type: {search_type}")

    print(f"Starting {search_type} search...")
    search.fit(X, y)

    print("\n[OK] Search complete!")
    print(f"  Best score: {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")

    results = {
        'best_score': search.best_score_,
        'best_params': search.best_params_,
        'cv_results': pd.DataFrame(search.cv_results_)
    }

    return search.best_estimator_, results


# ==================== MODEL EXPORT ====================

def export_model_for_production(
    model,
    model_name: str,
    framework: str = 'sklearn',
    export_format: str = 'onnx',
    input_sample: Optional[np.ndarray] = None
):
    """
    Export model for production deployment.

    Supports ONNX, TensorFlow Lite, CoreML formats.

    Parameters:
    -----------
    model : trained model
        Model to export.
    model_name : str
        Output filename.
    framework : str
        'sklearn', 'tensorflow', 'pytorch'.
    export_format : str
        'onnx', 'tflite', 'coreml'.
    input_sample : array, optional
        Sample input for shape inference.

    Examples:
    ---------
    >>> # Export sklearn model to ONNX
    >>> export_model_for_production(
    ...     model, 'my_model',
    ...     framework='sklearn',
    ...     export_format='onnx',
    ...     input_sample=X_train[:1]
    ... )
    """
    if export_format == 'onnx':
        try:
            if framework == 'sklearn':
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType

                initial_type = [('float_input', FloatTensorType([None, input_sample.shape[1]]))]
                onnx_model = convert_sklearn(model, initial_types=initial_type)

                with open(f"{model_name}.onnx", "wb") as f:
                    f.write(onnx_model.SerializeToString())

                print(f"[OK] Exported to {model_name}.onnx")

            elif framework == 'tensorflow':
                if importlib.util.find_spec("tf2onnx") is None:
                    raise ImportError("tf2onnx is required for TensorFlow ONNX export")
                # TensorFlow to ONNX conversion
                print("TensorFlow to ONNX export requires additional setup")

        except ImportError as e:
            print(f"[FAILED] ONNX export requires additional packages: {e}")

    elif export_format == 'tflite':
        if framework == 'tensorflow':
            import tensorflow as tf
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()

            with open(f"{model_name}.tflite", 'wb') as f:
                f.write(tflite_model)

            print(f"[OK] Exported to {model_name}.tflite")
        else:
            print("TFLite export only works with TensorFlow models")

    else:
        print(f"Export format '{export_format}' not yet implemented")


# ==================== PYTORCH UTILITIES ====================

@dataclass
class PyTorchTrainingHistory:
    """Track training history for PyTorch models."""

    train_losses: List[float] = None
    val_losses: List[float] = None
    train_metrics: Dict[str, List[float]] = None
    val_metrics: Dict[str, List[float]] = None
    epoch_times: List[float] = None
    learning_rates: List[float] = None

    def __post_init__(self):
        if self.train_losses is None:
            self.train_losses = []
        if self.val_losses is None:
            self.val_losses = []
        if self.train_metrics is None:
            self.train_metrics = {}
        if self.val_metrics is None:
            self.val_metrics = {}
        if self.epoch_times is None:
            self.epoch_times = []
        if self.learning_rates is None:
            self.learning_rates = []

    def add_epoch(
        self,
        train_loss: float,
        val_loss: Optional[float] = None,
        train_metrics: Optional[Dict[str, float]] = None,
        val_metrics: Optional[Dict[str, float]] = None,
        epoch_time: Optional[float] = None,
        lr: Optional[float] = None
    ):
        """Add metrics for an epoch."""
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)

        if train_metrics:
            for key, value in train_metrics.items():
                if key not in self.train_metrics:
                    self.train_metrics[key] = []
                self.train_metrics[key].append(value)

        if val_metrics:
            for key, value in val_metrics.items():
                if key not in self.val_metrics:
                    self.val_metrics[key] = []
                self.val_metrics[key].append(value)

        if epoch_time is not None:
            self.epoch_times.append(epoch_time)

        if lr is not None:
            self.learning_rates.append(lr)

    def plot(self, figsize: Tuple[int, int] = (15, 5)):
        """Plot training history."""
        import matplotlib.pyplot as plt

        n_plots = 1 + len(self.train_metrics)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)

        if n_plots == 1:
            axes = [axes]

        # Loss plot
        axes[0].plot(self.train_losses, label='Train Loss', linewidth=2)
        if self.val_losses:
            axes[0].plot(self.val_losses, label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Metrics plots
        for idx, metric_name in enumerate(self.train_metrics.keys(), 1):
            axes[idx].plot(self.train_metrics[metric_name],
                          label=f'Train {metric_name}', linewidth=2)
            if metric_name in self.val_metrics:
                axes[idx].plot(self.val_metrics[metric_name],
                             label=f'Val {metric_name}', linewidth=2)
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric_name.capitalize())
            axes[idx].set_title(f'{metric_name.capitalize()} Over Time')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert history to DataFrame."""
        data = {
            'epoch': list(range(1, len(self.train_losses) + 1)),
            'train_loss': self.train_losses
        }

        if self.val_losses:
            data['val_loss'] = self.val_losses

        for metric, values in self.train_metrics.items():
            data[f'train_{metric}'] = values

        for metric, values in self.val_metrics.items():
            data[f'val_{metric}'] = values

        if self.epoch_times:
            data['epoch_time'] = self.epoch_times

        if self.learning_rates:
            data['learning_rate'] = self.learning_rates

        return pd.DataFrame(data)


class PyTorchTrainer:
    """
    Comprehensive PyTorch training manager with callbacks, logging, and utilities.

    Features:
    - Early stopping
    - Learning rate scheduling
    - Model checkpointing
    - Progress tracking
    - Metric calculation
    - GPU support
    - Mixed precision training

    Examples:
    ---------
    >>> # Define model
    >>> model = nn.Sequential(
    ...     nn.Linear(10, 64),
    ...     nn.ReLU(),
    ...     nn.Dropout(0.2),
    ...     nn.Linear(64, 32),
    ...     nn.ReLU(),
    ...     nn.Linear(32, 1),
    ...     nn.Sigmoid()
    ... )
    >>>
    >>> # Create trainer
    >>> trainer = PyTorchTrainer(
    ...     model=model,
    ...     criterion=nn.BCELoss(),
    ...     optimizer=optim.Adam(model.parameters(), lr=0.001),
    ...     device='cuda',
    ...     early_stopping_patience=10
    ... )
    >>>
    >>> # Train
    >>> history = trainer.fit(
    ...     train_loader=train_loader,
    ...     val_loader=val_loader,
    ...     epochs=100,
    ...     verbose=True
    ... )
    >>>
    >>> # Plot history
    >>> history.plot()
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str = 'auto',
        scheduler: Optional[Any] = None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 1e-4,
        checkpoint_dir: Optional[str] = None,
        save_best_only: bool = True,
        mixed_precision: bool = False,
        gradient_clip_val: Optional[float] = None,
        metrics: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize PyTorch Trainer.

        Parameters:
        -----------
        model : nn.Module
            PyTorch model.
        criterion : nn.Module
            Loss function.
        optimizer : optim.Optimizer
            Optimizer.
        device : str
            Device ('auto', 'cuda', 'cpu', 'mps').
        scheduler : optional
            Learning rate scheduler.
        early_stopping_patience : int, optional
            Patience for early stopping.
        early_stopping_min_delta : float
            Minimum change for improvement.
        checkpoint_dir : str, optional
            Directory for checkpoints.
        save_best_only : bool
            Only save best model.
        mixed_precision : bool
            Use automatic mixed precision.
        gradient_clip_val : float, optional
            Gradient clipping value.
        metrics : dict, optional
            Additional metrics {'name': metric_function}.
        """
        _require_pytorch()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.save_best_only = save_best_only
        self.mixed_precision = mixed_precision
        self.gradient_clip_val = gradient_clip_val
        self.metrics = metrics or {}

        # Auto-detect device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        # Create checkpoint directory
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.history = PyTorchTrainingHistory()

        print("[OK] Trainer initialized")
        print(f"  Device: {self.device}")
        print(f"  Mixed Precision: {mixed_precision}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        verbose: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        metric_totals = {name: 0.0 for name in self.metrics.keys()}
        n_batches = len(train_loader)

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)

                # Backward pass with scaling
                self.scaler.scale(loss).backward()

                if self.gradient_clip_val:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_val
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                # Backward pass
                loss.backward()

                if self.gradient_clip_val:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_val
                    )

                self.optimizer.step()

            total_loss += loss.item()

            # Calculate metrics
            with torch.no_grad():
                for metric_name, metric_fn in self.metrics.items():
                    metric_value = metric_fn(outputs, y_batch)
                    metric_totals[metric_name] += metric_value

            # Progress
            if verbose and (batch_idx + 1) % max(1, n_batches // 10) == 0:
                print(f"  Batch {batch_idx + 1}/{n_batches} - Loss: {loss.item():.4f}", end='\\r')

        avg_loss = total_loss / n_batches
        avg_metrics = {name: total / n_batches for name, total in metric_totals.items()}

        return avg_loss, avg_metrics

    def validate_epoch(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        metric_totals = {name: 0.0 for name in self.metrics.keys()}
        n_batches = len(val_loader)

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                total_loss += loss.item()

                # Calculate metrics
                for metric_name, metric_fn in self.metrics.items():
                    metric_value = metric_fn(outputs, y_batch)
                    metric_totals[metric_name] += metric_value

        avg_loss = total_loss / n_batches
        avg_metrics = {name: total / n_batches for name, total in metric_totals.items()}

        return avg_loss, avg_metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        verbose: bool = True
    ) -> PyTorchTrainingHistory:
        """
        Train the model.

        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader.
        val_loader : DataLoader, optional
            Validation data loader.
        epochs : int
            Number of epochs.
        verbose : bool
            Print progress.

        Returns:
        --------
        PyTorchTrainingHistory
            Training history.
        """
        print(f"\\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}\\n")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss, train_metrics = self.train_epoch(train_loader, epoch, verbose)

            # Validate
            val_loss = None
            val_metrics = None
            if val_loader:
                val_loss, val_metrics = self.validate_epoch(val_loader)

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loss else train_loss)
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            epoch_time = time.time() - epoch_start

            # Add to history
            self.history.add_epoch(
                train_loss=train_loss,
                val_loss=val_loss,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                epoch_time=epoch_time,
                lr=current_lr
            )

            # Print progress
            if verbose:
                print(f"\\nEpoch {epoch}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}", end='')
                if val_loss:
                    print(f" | Val Loss: {val_loss:.4f}", end='')

                for metric_name in train_metrics.keys():
                    print(f" | Train {metric_name}: {train_metrics[metric_name]:.4f}", end='')
                    if val_metrics and metric_name in val_metrics:
                        print(f" | Val {metric_name}: {val_metrics[metric_name]:.4f}", end='')

                print(f" | LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")

            # Save checkpoint
            if self.checkpoint_dir:
                if self.save_best_only:
                    if val_loss and val_loss < self.best_val_loss - self.early_stopping_min_delta:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(epoch, is_best=True)
                        if verbose:
                            print(f"  [OK] Saved best model (val_loss: {val_loss:.4f})")
                else:
                    self.save_checkpoint(epoch, is_best=False)

            # Early stopping
            if self.early_stopping_patience and val_loss:
                if val_loss < self.best_val_loss - self.early_stopping_min_delta:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                    if verbose:
                        print(f"  Early stopping: {self.epochs_without_improvement}/{self.early_stopping_patience}")

                    if self.epochs_without_improvement >= self.early_stopping_patience:
                        print(f"\\n⏹️  Early stopping triggered after {epoch} epochs")
                        break

        print(f"\\n{'='*60}")
        print("Training Complete")
        print(f"{'='*60}\\n")

        return self.history

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.history.train_losses[-1],
            'val_loss': self.history.val_losses[-1] if self.history.val_losses else None,
            'best_val_loss': self.best_val_loss
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        filename = 'best_model.pt' if is_best else f'checkpoint_epoch_{epoch}.pt'
        filepath = self.checkpoint_dir / filename

        torch.save(checkpoint, filepath)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"[OK] Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Best Val Loss: {self.best_val_loss:.4f}")

    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """Make predictions."""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for X_batch, _ in data_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                predictions.append(outputs.cpu().numpy())

        return np.concatenate(predictions, axis=0)


# ==================== PYTORCH MODEL SAVING/LOADING ====================

def save_pytorch_model(
    model: nn.Module,
    model_name: str,
    optimizer: Optional[optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict] = None,
    save_format: str = 'state_dict'
):
    """
    Save PyTorch model with multiple formats.

    Parameters:
    -----------
    model : nn.Module
        PyTorch model.
    model_name : str
        Name for saving.
    optimizer : Optimizer, optional
        Optimizer state.
    epoch : int, optional
        Current epoch.
    metrics : dict, optional
        Training metrics.
    save_format : str
        'state_dict' (recommended), 'full_model', or 'scripted'.

    Examples:
    ---------
    >>> # Save state dict (recommended)
    >>> save_pytorch_model(model, 'my_model', save_format='state_dict')
    >>>
    >>> # Save with optimizer and metrics
    >>> save_pytorch_model(
    ...     model, 'my_model',
    ...     optimizer=optimizer,
    ...     epoch=50,
    ...     metrics={'loss': 0.15, 'accuracy': 0.95}
    ... )
    >>>
    >>> # Save as TorchScript
    >>> save_pytorch_model(model, 'my_model', save_format='scripted')
    """
    _require_pytorch()
    if save_format == 'state_dict':
        # Save state dict (recommended - most flexible)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_architecture': str(model)
        }

        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if metrics:
            checkpoint['metrics'] = metrics

        torch.save(checkpoint, f"{model_name}.pt")
        print(f"[OK] Saved model state dict to {model_name}.pt")

    elif save_format == 'full_model':
        # Save entire model
        torch.save(model, f"{model_name}_full.pt")
        print(f"[OK] Saved full model to {model_name}_full.pt")

    elif save_format == 'scripted':
        # Save as TorchScript (for production)
        scripted_model = torch.jit.script(model)
        scripted_model.save(f"{model_name}_scripted.pt")
        print(f"[OK] Saved TorchScript model to {model_name}_scripted.pt")

    else:
        raise ValueError(f"Unknown save_format: {save_format}")


def load_pytorch_model(
    model_class: Optional[nn.Module] = None,
    model_name: str = 'model',
    load_format: str = 'state_dict',
    device: str = 'cpu',
    **model_kwargs
) -> nn.Module:
    """
    Load PyTorch model.

    Parameters:
    -----------
    model_class : nn.Module, optional
        Model class (for state_dict loading).
    model_name : str
        Name of saved model.
    load_format : str
        'state_dict', 'full_model', or 'scripted'.
    device : str
        Device to load to.
    **model_kwargs :
        Arguments for model initialization.

    Returns:
    --------
    nn.Module
        Loaded model.

    Examples:
    ---------
    >>> # Load state dict
    >>> model = load_pytorch_model(
    ...     model_class=MyModel,
    ...     model_name='my_model',
    ...     load_format='state_dict',
    ...     input_size=10,
    ...     hidden_size=64
    ... )
    >>>
    >>> # Load full model
    >>> model = load_pytorch_model(
    ...     model_name='my_model_full',
    ...     load_format='full_model'
    ... )
    >>>
    >>> # Load TorchScript
    >>> model = load_pytorch_model(
    ...     model_name='my_model_scripted',
    ...     load_format='scripted'
    ... )
    """
    _require_pytorch()
    device = torch.device(device)

    if load_format == 'state_dict':
        if model_class is None:
            raise ValueError("model_class is required for state_dict loading")

        # Initialize model
        model = model_class(**model_kwargs)

        # Load checkpoint
        checkpoint = torch.load(f"{model_name}.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        model.to(device)
        model.eval()

        print(f"[OK] Loaded model from {model_name}.pt")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            print(f"  Metrics: {checkpoint['metrics']}")

        return model

    elif load_format == 'full_model':
        model = torch.load(f"{model_name}_full.pt", map_location=device)
        model.to(device)
        model.eval()
        print(f"[OK] Loaded full model from {model_name}_full.pt")
        return model

    elif load_format == 'scripted':
        model = torch.jit.load(f"{model_name}_scripted.pt", map_location=device)
        model.to(device)
        model.eval()
        print(f"[OK] Loaded TorchScript model from {model_name}_scripted.pt")
        return model

    else:
        raise ValueError(f"Unknown load_format: {load_format}")


# ==================== PYTORCH UTILITIES ====================

def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count model parameters.

    Parameters:
    -----------
    model : nn.Module
        PyTorch model.
    trainable_only : bool
        Count only trainable parameters.

    Returns:
    --------
    int
        Number of parameters.

    Examples:
    ---------
    >>> model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
    >>> print(f"Total parameters: {count_parameters(model):,}")
    >>> print(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def freeze_layers(model: nn.Module, layer_names: Optional[List[str]] = None):
    """
    Freeze model layers.

    Parameters:
    -----------
    model : nn.Module
        PyTorch model.
    layer_names : list, optional
        Names of layers to freeze. If None, freezes all.

    Examples:
    ---------
    >>> # Freeze all layers
    >>> freeze_layers(model)
    >>>
    >>> # Freeze specific layers
    >>> freeze_layers(model, layer_names=['layer1', 'layer2'])
    """
    if layer_names is None:
        # Freeze all
        for param in model.parameters():
            param.requires_grad = False
        print("[OK] Froze all layers")
    else:
        # Freeze specific layers
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
                print(f"  Froze: {name}")


def unfreeze_layers(model: nn.Module, layer_names: Optional[List[str]] = None):
    """
    Unfreeze model layers.

    Parameters:
    -----------
    model : nn.Module
        PyTorch model.
    layer_names : list, optional
        Names of layers to unfreeze. If None, unfreezes all.
    """
    if layer_names is None:
        for param in model.parameters():
            param.requires_grad = True
        print("[OK] Unfroze all layers")
    else:
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
                print(f"  Unfroze: {name}")


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get optimal device for PyTorch.

    Parameters:
    -----------
    prefer_gpu : bool
        Prefer GPU if available.

    Returns:
    --------
    torch.device
        Device object.

    Examples:
    ---------
    >>> device = get_device()
    >>> model.to(device)
    """
    _require_pytorch()
    if prefer_gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"[OK] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("[OK] Using Apple MPS (Metal)")
        else:
            device = torch.device('cpu')
            print("[WARN] No GPU available, using CPU")
    else:
        device = torch.device('cpu')
        print("[OK] Using CPU")

    return device


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Parameters:
    -----------
    seed : int
        Random seed.

    Examples:
    ---------
    >>> set_seed(42)  # Ensures reproducible results
    """
    _require_pytorch()
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[OK] Set random seed to {seed}")


def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """
    Create PyTorch DataLoaders from numpy arrays.

    Parameters:
    -----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    X_val : np.ndarray, optional
        Validation features.
    y_val : np.ndarray, optional
        Validation labels.
    batch_size : int
        Batch size.
    shuffle : bool
        Shuffle training data.
    num_workers : int
        Number of data loading workers.

    Returns:
    --------
    DataLoader or tuple
        Train loader, or (train_loader, val_loader) if validation data provided.

    Examples:
    ---------
    >>> # Create train and validation loaders
    >>> train_loader, val_loader = create_data_loaders(
    ...     X_train, y_train, X_val, y_val,
    ...     batch_size=64
    ... )
    >>>
    >>> # Create only train loader
    >>> train_loader = create_data_loaders(X_train, y_train, batch_size=32)
    """
    # Convert to tensors
    _require_pytorch()
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    if X_val is not None and y_val is not None:
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)

        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        return train_loader, val_loader

    return train_loader


# ==================== COMMON PYTORCH METRICS ====================

def accuracy_pytorch(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate accuracy for PyTorch tensors."""
    _require_pytorch()
    if outputs.shape[1] > 1:  # Multi-class
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
    else:  # Binary
        predicted = (outputs > 0.5).float()
        correct = (predicted == targets).sum().item()

    return correct / targets.size(0)


def f1_score_pytorch(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate F1 score for PyTorch tensors."""
    _require_pytorch()
    predicted = (outputs > 0.5).float() if outputs.shape[1] == 1 else torch.argmax(outputs, dim=1)

    tp = ((predicted == 1) & (targets == 1)).sum().float()
    fp = ((predicted == 1) & (targets == 0)).sum().float()
    fn = ((predicted == 0) & (targets == 1)).sum().float()

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return f1.item()


# ==================== PRE-BUILT MODELS ====================

class SimpleClassifier(_TorchModuleBase):
    """Simple fully-connected classifier."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [64, 32],
        output_size: int = 1,
        dropout: float = 0.2,
        activation: str = 'relu'
    ):
        _require_pytorch()
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        if output_size == 1:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class SimpleRegressor(_TorchModuleBase):
    """Simple fully-connected regressor."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [64, 32],
        output_size: int = 1,
        dropout: float = 0.2
    ):
        _require_pytorch()
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
