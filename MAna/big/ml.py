"""Small PySpark ML pipeline builders for mixed tabular features."""

from __future__ import annotations

from typing import Any, Sequence


def _ml_classes():
    try:
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import (
            Imputer,
            OneHotEncoder,
            StandardScaler,
            StringIndexer,
            VectorAssembler,
        )
    except ImportError as exc:
        raise ImportError(
            "Spark ML support requires: pip install 'M_Ana_package[big]'"
        ) from exc
    return Pipeline, Imputer, OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler


def build_feature_pipeline(
    *,
    numeric_columns: Sequence[str] = (),
    categorical_columns: Sequence[str] = (),
    output_column: str = "features",
    impute_strategy: str = "median",
    handle_invalid: str = "keep",
    scale_features: bool = False,
) -> Any:
    """Build an unfitted Spark ML pipeline for numeric and categorical columns."""
    numeric = list(numeric_columns)
    categorical = list(categorical_columns)
    if not numeric and not categorical:
        raise ValueError("At least one feature column is required")
    if len(set([*numeric, *categorical])) != len(numeric) + len(categorical):
        raise ValueError("feature columns must be unique")
    if handle_invalid not in {"error", "skip", "keep"}:
        raise ValueError("handle_invalid must be 'error', 'skip', or 'keep'")
    if impute_strategy not in {"mean", "median", "mode"}:
        raise ValueError("impute_strategy must be 'mean', 'median', or 'mode'")

    Pipeline, Imputer, OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler = _ml_classes()
    stages = []
    assembled_inputs = []

    if numeric:
        imputed = [f"__mana_imputed_{column}" for column in numeric]
        stages.append(
            Imputer(
                inputCols=numeric,
                outputCols=imputed,
                strategy=impute_strategy,
            )
        )
        assembled_inputs.extend(imputed)

    if categorical:
        indexed = [f"__mana_indexed_{column}" for column in categorical]
        encoded = [f"__mana_encoded_{column}" for column in categorical]
        for source, destination in zip(categorical, indexed):
            stages.append(
                StringIndexer(
                    inputCol=source,
                    outputCol=destination,
                    handleInvalid=handle_invalid,
                )
            )
        stages.append(
            OneHotEncoder(
                inputCols=indexed,
                outputCols=encoded,
                handleInvalid="keep" if handle_invalid == "keep" else "error",
            )
        )
        assembled_inputs.extend(encoded)

    assembler_output = "__mana_unscaled_features" if scale_features else output_column
    stages.append(
        VectorAssembler(
            inputCols=assembled_inputs,
            outputCol=assembler_output,
            handleInvalid=handle_invalid,
        )
    )
    if scale_features:
        stages.append(
            StandardScaler(
                inputCol=assembler_output,
                outputCol=output_column,
                withMean=False,
                withStd=True,
            )
        )
    return Pipeline(stages=stages)


def random_split_dataframe(
    frame: Any,
    *,
    weights: Sequence[float] = (0.8, 0.2),
    seed: int = 42,
) -> Sequence[Any]:
    """Create reproducible Spark train/test or train/validation/test splits."""
    values = [float(weight) for weight in weights]
    if len(values) < 2 or any(weight <= 0 for weight in values):
        raise ValueError("weights must contain at least two positive values")
    return frame.randomSplit(values, seed=seed)


__all__ = ["build_feature_pipeline", "random_split_dataframe"]
