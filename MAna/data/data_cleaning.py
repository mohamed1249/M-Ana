import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def _make_hashable(value: Any) -> Any:
    """Return a stable, hashable representation of nested cell values."""
    if isinstance(value, dict):
        items = (
            (_make_hashable(key), _make_hashable(item))
            for key, item in value.items()
        )
        return tuple(sorted(items, key=repr))
    if isinstance(value, np.ndarray):
        return tuple(_make_hashable(item) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_make_hashable(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_make_hashable(item) for item in value), key=repr))

    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _comparable_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Copy a frame and normalize unhashable object cells for comparisons."""
    comparable = df.copy()
    for column in comparable.select_dtypes(include=["object"]).columns:
        comparable[column] = comparable[column].map(_make_hashable)
    return comparable


def _safe_nunique(series: pd.Series) -> int:
    """Count unique values even when an object series contains nested values."""
    if pd.api.types.is_object_dtype(series.dtype):
        series = series.map(_make_hashable)
    return int(series.nunique(dropna=True))


def drop_missing_values(df, threshold=0.5, axis=0, subset=None) -> pd.DataFrame:
    """
    Drops missing values from a dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to drop missing values from.
    threshold : float, optional (default=0.5)
        The threshold for the proportion of missing values in a row/column to be dropped.
        If a row/column has missing values in more than `threshold` proportion, it will be dropped.
    axis : int, optional (default=0)
        The axis along which to drop missing values. 0 for rows, 1 for columns.
    subset : list or tuple, optional (default=None)
        The list of column names to consider for dropping missing values.
        If None, all columns are considered.

    Returns:
    --------
    pandas.DataFrame
        The dataframe with missing values dropped.
    """
    if subset is not None:
        df = df[subset]
    if axis == 0:
        # Drop rows with missing values
        return df.dropna(thresh=int(threshold * len(df.columns)))
    elif axis == 1:
        # Drop columns with missing values
        return df.dropna(axis=1, thresh=int(threshold * len(df.index)))
    else:
        raise ValueError("Axis must be 0 or 1.")


def fill_missing_values(df, fill_value=None, method='mean', subset=None) -> pd.DataFrame:
    """
    Fills missing values in a dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to fill missing values in.
    fill_value : scalar or dict, optional (default=None)
        The value or dictionary of values to use for filling missing values.
        If None, `method` is used to fill missing values.
    method : str, optional (default='mean')
        The method used to fill missing values. Supported methods are 'mean', 'median', 'mode',
        and 'ffill' (forward fill). Ignored if `fill_value` is not None. Default is 'mean'.
    subset : list or tuple, optional (default=None)
        The list of column names to consider for filling missing values.
        If None, all columns are considered.

    Returns:
    --------
    pandas.DataFrame
        The dataframe with missing values filled.
    """
    if subset is not None:
        df = df[subset]
    if fill_value is not None:
        return df.fillna(fill_value)
    elif method == 'mean':
        return df.fillna(df.mean())
    elif method == 'median':
        return df.fillna(df.median())
    elif method == 'mode':
        return df.fillna(df.mode().iloc[0])
    elif method == 'ffill':
        return df.ffill()
    else:
        raise ValueError(f"Unsupported method: {method}. Supported methods are 'mean', 'median', 'mode', and 'ffill'.")


def z_remove_outliers(df, threshold=3) -> pd.DataFrame:
    """
    Remove outliers from a Pandas DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to remove outliers from.
    threshold : int or float, optional (default=3)
        The threshold for identifying outliers using the z-score.
        Observations with a z-score greater than or equal to
        `threshold` will be considered outliers.

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with outliers removed.
    """

    from scipy.stats import zscore

    # Calculate the z-score for each observation in the DataFrame
    zscores = df.apply(zscore)

    # Identify outliers based on the z-score
    outliers = (zscores.abs() >= threshold).any(axis=1)

    # Remove outliers from the DataFrame
    df_clean = df[~outliers]

    return df_clean

def iqr_remove_outliers(df, factor=1.5) -> pd.DataFrame:
    """
    Remove outliers using the IQR (Interquartile Range) method.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to remove outliers from.
    factor : float, optional (default=1.5)
        The IQR multiplier. 1.5 is standard, 3.0 is more conservative.

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with outliers removed.
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # Define outlier bounds
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    # Filter outliers
    mask = ~((df < lower_bound) | (df > upper_bound)).any(axis=1)
    return df[mask]


def iso_remove_outliers(df, contamination=0.05, random_state=42) -> pd.DataFrame:
    """
    Remove outliers using Isolation Forest algorithm.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to remove outliers from.
    contamination : float, optional (default=0.05)
        Expected proportion of outliers (0.05 = 5%).
    random_state : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with outliers removed.
    """
    from sklearn.ensemble import IsolationForest

    # Only use numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns

    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100
    )

    predictions = iso_forest.fit_predict(df[numeric_cols])

    # -1 indicates outlier, 1 indicates inlier
    return df[predictions == 1]


def lof_remove_outliers(df, n_neighbors=20, contamination=0.1) -> pd.DataFrame:
    """
    Remove outliers using Local Outlier Factor.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to remove outliers from.
    n_neighbors : int, optional (default=20)
        Number of neighbors to consider.
    contamination : float, optional (default=0.1)
        Expected proportion of outliers.

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with outliers removed.
    """
    from sklearn.neighbors import LocalOutlierFactor

    numeric_cols = df.select_dtypes(include=['number']).columns

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination
    )

    predictions = lof.fit_predict(df[numeric_cols])
    return df[predictions == 1]



def mad_remove_outliers(df, threshold=3.5) -> pd.DataFrame:
    """
    Remove outliers using Modified Z-score (MAD method).
    More robust than standard z-score for skewed distributions.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to remove outliers from.
    threshold : float, optional (default=3.5)
        Threshold for modified z-score (3.5 is recommended).

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with outliers removed.
    """
    import numpy as np

    def modified_z_scores(series):
        median = series.median()
        mad = np.median(np.abs(series - median))
        # Avoid division by zero
        if mad == 0:
            mad = np.mean(np.abs(series - median))
        modified_z = 0.6745 * (series - median) / mad
        return modified_z

    numeric_df = df.select_dtypes(include=['number'])
    mod_z_scores = numeric_df.apply(modified_z_scores)

    outliers = (mod_z_scores.abs() >= threshold).any(axis=1)
    return df[~outliers]


def dbscan_remove_outliers(df, eps=0.5, min_samples=5) -> pd.DataFrame:
    """
    Remove outliers using DBSCAN clustering.
    Points labeled as noise (-1) are considered outliers.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to remove outliers from.
    eps : float, optional (default=0.5)
        Maximum distance between samples.
    min_samples : int, optional (default=5)
        Minimum samples in a neighborhood.

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with outliers removed.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    numeric_cols = df.select_dtypes(include=['number']).columns

    # Scale data for DBSCAN
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_cols])

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(scaled_data)

    # Keep only points that belong to a cluster (not noise)
    return df[labels != -1]


def percentile_remove_outliers(df, lower=0.01, upper=0.99) -> pd.DataFrame:
    """
    Remove outliers based on percentile thresholds.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to remove outliers from.
    lower : float, optional (default=0.01)
        Lower percentile threshold (0.01 = 1st percentile).
    upper : float, optional (default=0.99)
        Upper percentile threshold (0.99 = 99th percentile).

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with outliers removed.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns

    mask = pd.Series([True] * len(df), index=df.index)

    for col in numeric_cols:
        lower_bound = df[col].quantile(lower)
        upper_bound = df[col].quantile(upper)
        mask &= (df[col] >= lower_bound) & (df[col] <= upper_bound)

    return df[mask]


def remove_outliers(df, method='zscore', **kwargs) -> pd.DataFrame:
    """
    Remove outliers using various methods.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to remove outliers from.
    method : str, optional (default='zscore')
        Method to use: 'zscore', 'iqr', 'isolation_forest', 'lof',
        'mad', 'dbscan', 'percentile'
    **kwargs :
        Method-specific parameters

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with outliers removed.

    Examples:
    ---------
    >>> df_clean = remove_outliers(df, method='iqr', factor=1.5)
    >>> df_clean = remove_outliers(df, method='isolation_forest', contamination=0.05)
    """
    methods = {
        'zscore': z_remove_outliers,
        'iqr': iqr_remove_outliers,
        'isolation_forest': iso_remove_outliers,
        'lof': lof_remove_outliers,
        'mad': mad_remove_outliers,
        'dbscan': dbscan_remove_outliers,
        'percentile': percentile_remove_outliers
    }

    if method not in methods:
        raise ValueError(f"Method '{method}' not supported. Choose from: {list(methods.keys())}")

    return methods[method](df, **kwargs)




def group_and_aggregate(
    df: pd.DataFrame,
    group_cols: List[str],
    agg_dict: Dict[str, Any],
) -> pd.DataFrame:
    """
    Group and aggregate data in a dataframe.

    Parameters:
        df (pandas.DataFrame): The input dataframe.
        group_cols (list): A list of column names to group by.
        agg_dict (dict): A dictionary of columns to aggregate and their respective aggregate functions.

    Returns:
        pandas.DataFrame: The grouped and aggregated dataframe.
    """
    # Group by the specified columns and apply the specified aggregate functions
    grouped_df = df.groupby(group_cols).agg(agg_dict)

    # Flatten the column index of the resulting dataframe
    grouped_df.columns = ['_'.join(col).strip() for col in grouped_df.columns.values]

    # Reset the index to turn the group columns back into regular columns
    grouped_df.reset_index(inplace=True)

    return grouped_df


def solve_data_entry_errors(
    df,
    column=None,
    expected_values=None,
    auto_fix=True,
    case_sensitive=False,
    strip_whitespace=True,
    remove_special_chars=False,
    standardize_spacing=True,
    fix_common_typos=True,
    fuzzy_match=False,
    fuzzy_threshold=80,
    return_report=False,
    inplace=False
):
    """
    Ultimate data entry error detection and correction function for TEXT data.
    Handles text normalization, typo correction, and fuzzy matching.

    Note: For outlier detection in numeric columns, use the dedicated remove_outliers() functions.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to check for data entry errors.
    column : str or list, optional
        Column name(s) to check. If None, checks all object/string columns.
    expected_values : list, dict, or callable, optional
        - list: Expected valid values for the column
        - dict: Mapping of incorrect values to correct values
        - callable: Custom validation function that returns bool
    auto_fix : bool, optional (default=True)
        Automatically fix common issues (whitespace, case, special chars).
    case_sensitive : bool, optional (default=False)
        Whether to consider case when comparing values.
    strip_whitespace : bool, optional (default=True)
        Remove leading/trailing whitespace.
    remove_special_chars : bool, optional (default=False)
        Remove special characters (keep only alphanumeric and spaces).
    standardize_spacing : bool, optional (default=True)
        Replace multiple spaces with single space.
    fix_common_typos : bool, optional (default=True)
        Attempt to fix common typos (e.g., 'teh' -> 'the').
    fuzzy_match : bool, optional (default=False)
        Use fuzzy string matching to suggest corrections.
    fuzzy_threshold : int, optional (default=80)
        Minimum similarity score (0-100) for fuzzy matching.
    return_report : bool, optional (default=False)
        Return detailed report of changes made.
    inplace : bool, optional (default=False)
        Modify DataFrame in place and return None.

    Returns:
    --------
    pandas.DataFrame or (pandas.DataFrame, dict)
        The cleaned DataFrame, optionally with a report dict if return_report=True.
        If inplace=True, returns None.

    Examples:
    ---------
    >>> # Basic usage with expected values
    >>> df_clean = solve_data_entry_errors(df, 'gender', expected_values=['male', 'female'])

    >>> # Auto-fix with typo correction
    >>> df_clean = solve_data_entry_errors(df, 'country', auto_fix=True, fix_common_typos=True)

    >>> # Fuzzy matching for similar values
    >>> df_clean = solve_data_entry_errors(df, 'city', expected_values=['New York', 'Los Angeles'],
    ...                                      fuzzy_match=True, fuzzy_threshold=85)

    >>> # Custom mapping for common variations
    >>> mapping = {'M': 'male', 'F': 'female', 'm': 'male', 'f': 'female'}
    >>> df_clean = solve_data_entry_errors(df, 'gender', expected_values=mapping)

    >>> # Get detailed report of changes
    >>> df_clean, report = solve_data_entry_errors(df, 'status', return_report=True)
    >>> print(f"Fixed {report['total_fixes']} errors")

    >>> # Custom validation function
    >>> df_clean = solve_data_entry_errors(df, 'email',
    ...     expected_values=lambda x: '@' in str(x) and '.' in str(x))
    """
    import pandas as pd
    import re

    # Create copy unless inplace
    df_clean = df if inplace else df.copy()

    # Initialize report
    report = {
        'columns_processed': [],
        'total_fixes': 0,
        'errors_by_column': {},
        'corrections_made': {},
        'unfixable_values': {}
    }

    # Determine which columns to process
    if column is None:
        # Process all string/object columns
        columns_to_process = df_clean.select_dtypes(include=['object', 'string']).columns.tolist()
    elif isinstance(column, str):
        columns_to_process = [column]
    else:
        columns_to_process = column

    # Common typo mappings
    common_typos = {
        'teh': 'the',
        'adn': 'and',
        'recieve': 'receive',
        'occured': 'occurred',
        'seperate': 'separate',
        'definately': 'definitely',
        'accomodate': 'accommodate',
        'occassion': 'occasion',
        'publically': 'publicly',
        'untill': 'until',
        'thier': 'their',
        'wich': 'which',
        'wierd': 'weird',
        'beleive': 'believe',
        'reccomend': 'recommend'
    }

    def normalize_text(text, strip_ws=True, remove_special=False, standardize_space=True, lower=True):
        """Normalize text value."""
        if pd.isna(text) or not isinstance(text, str):
            return text

        result = text

        # Strip whitespace
        if strip_ws:
            result = result.strip()

        # Convert to lowercase
        if lower and not case_sensitive:
            result = result.lower()

        # Remove special characters
        if remove_special:
            result = re.sub(r'[^a-zA-Z0-9\s]', '', result)

        # Standardize spacing
        if standardize_space:
            result = re.sub(r'\s+', ' ', result)

        return result

    def fix_typos(text, typo_dict):
        """Fix common typos in text."""
        if pd.isna(text) or not isinstance(text, str):
            return text

        words = text.split()
        fixed_words = [typo_dict.get(word, word) for word in words]
        return ' '.join(fixed_words)

    def fuzzy_match_value(value, expected_list, threshold=80):
        """Find best fuzzy match from expected values."""
        try:
            from fuzzywuzzy import fuzz
            use_fuzzywuzzy = True
        except ImportError:
            # Fallback: simple character-based similarity
            use_fuzzywuzzy = False

        if pd.isna(value):
            return None

        best_match = None
        best_score = 0

        for expected in expected_list:
            if use_fuzzywuzzy:
                score = fuzz.ratio(str(value), str(expected))
            else:
                # Simple similarity calculation
                s1, s2 = str(value).lower(), str(expected).lower()
                matches = sum(1 for a, b in zip(s1, s2) if a == b)
                score = (matches / max(len(s1), len(s2))) * 100

            if score > best_score and score >= threshold:
                best_score = score
                best_match = expected

        return best_match

    # Process each column
    for col in columns_to_process:
        if col not in df_clean.columns:
            continue

        report['columns_processed'].append(col)
        report['errors_by_column'][col] = 0
        report['corrections_made'][col] = []
        report['unfixable_values'][col] = []

        # Skip numeric columns (use remove_outliers() for those)
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            continue

        # Convert column to string for processing
        df_clean[col] = df_clean[col].astype(str)

        # Replace 'None' and 'NaN' strings with actual NaN
        df_clean.loc[df_clean[col].isin(['None', 'nan', 'NaN', 'null', 'NULL']), col] = pd.NA

        # Auto-fix: normalize text
        if auto_fix:
            df_clean[col] = df_clean[col].apply(
                lambda x: normalize_text(
                    x,
                    strip_ws=strip_whitespace,
                    remove_special=remove_special_chars,
                    standardize_space=standardize_spacing,
                    lower=not case_sensitive
                )
            )

        # Fix common typos
        if fix_common_typos:
            df_clean[col] = df_clean[col].apply(lambda x: fix_typos(x, common_typos))

        # Handle expected values
        if expected_values is not None:
            # Case 1: Dictionary mapping (incorrect -> correct)
            if isinstance(expected_values, dict):
                for incorrect, correct in expected_values.items():
                    incorrect_norm = normalize_text(
                        str(incorrect),
                        strip_ws=strip_whitespace,
                        standardize_space=standardize_spacing,
                        lower=not case_sensitive
                    )
                    mask = df_clean[col] == incorrect_norm
                    count = mask.sum()

                    if count > 0:
                        df_clean.loc[mask, col] = correct
                        report['errors_by_column'][col] += count
                        report['total_fixes'] += count
                        report['corrections_made'][col].append(f'{incorrect} -> {correct} ({count}x)')

            # Case 2: List of valid values
            elif isinstance(expected_values, list):
                expected_normalized = [
                    normalize_text(
                        str(v),
                        strip_ws=strip_whitespace,
                        standardize_space=standardize_spacing,
                        lower=not case_sensitive
                    )
                    for v in expected_values
                ]

                # Find invalid values
                mask = ~df_clean[col].isin(expected_normalized + ['nan'])
                invalid_values = df_clean.loc[mask, col].unique()

                for invalid_val in invalid_values:
                    if pd.isna(invalid_val) or invalid_val == 'nan':
                        continue

                    # Try fuzzy matching
                    if fuzzy_match:
                        matched = fuzzy_match_value(invalid_val, expected_normalized, fuzzy_threshold)
                        if matched:
                            val_mask = df_clean[col] == invalid_val
                            count = val_mask.sum()
                            df_clean.loc[val_mask, col] = matched
                            report['errors_by_column'][col] += count
                            report['total_fixes'] += count
                            report['corrections_made'][col].append(
                                f'Fuzzy matched: {invalid_val} -> {matched} ({count}x)'
                            )
                        else:
                            # Replace with NaN
                            val_mask = df_clean[col] == invalid_val
                            count = val_mask.sum()
                            df_clean.loc[val_mask, col] = pd.NA
                            report['errors_by_column'][col] += count
                            report['total_fixes'] += count
                            report['unfixable_values'][col].append(invalid_val)
                    else:
                        # Replace with NaN
                        val_mask = df_clean[col] == invalid_val
                        count = val_mask.sum()
                        df_clean.loc[val_mask, col] = pd.NA
                        report['errors_by_column'][col] += count
                        report['total_fixes'] += count
                        report['unfixable_values'][col].append(invalid_val)

            # Case 3: Callable validation function
            elif callable(expected_values):
                invalid_mask = ~df_clean[col].apply(expected_values)
                count = invalid_mask.sum()

                if count > 0:
                    invalid_vals = df_clean.loc[invalid_mask, col].unique()[:5]  # First 5 examples
                    df_clean.loc[invalid_mask, col] = pd.NA
                    report['errors_by_column'][col] += count
                    report['total_fixes'] += count
                    report['corrections_made'][col].append(
                        f'Failed custom validation ({count}x)'
                    )
                    report['unfixable_values'][col].extend(invalid_vals)

        # Convert 'nan' strings back to actual NaN
        df_clean.loc[df_clean[col] == 'nan', col] = pd.NA

    # Generate summary
    if return_report:
        report['summary'] = (
            f"Processed {len(report['columns_processed'])} columns, "
            f"made {report['total_fixes']} corrections"
        )
        return (None if inplace else df_clean), report

    return None if inplace else df_clean


# Convenience functions for common use cases
def quick_clean_text_entries(df, column, expected_values=None):
    """Quick text-entry normalization with sensible defaults."""
    return solve_data_entry_errors(
        df, column, expected_values,
        auto_fix=True,
        fix_common_typos=True,
        fuzzy_match=True,
        fuzzy_threshold=85
    )


def strict_validate(df, column, expected_values):
    """Strict validation - no auto-fixing, exact matches only."""
    return solve_data_entry_errors(
        df, column, expected_values,
        auto_fix=False,
        case_sensitive=True,
        fuzzy_match=False
    )


def smart_clean(df, column=None, return_report=True):
    """Smart cleaning with comprehensive report."""
    return solve_data_entry_errors(
        df, column,
        auto_fix=True,
        fix_common_typos=True,
        return_report=return_report
    )



@dataclass
class CleaningReport:
    """Detailed report of all cleaning operations performed."""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    original_shape: Tuple[int, int] = (0, 0)
    final_shape: Tuple[int, int] = (0, 0)
    rows_removed: int = 0
    columns_removed: int = 0

    # Detailed tracking
    missing_values_filled: Dict[str, int] = field(default_factory=dict)
    outliers_removed: Dict[str, int] = field(default_factory=dict)
    duplicates_removed: int = 0
    data_types_changed: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    text_corrections: Dict[str, List[str]] = field(default_factory=dict)
    columns_created: List[str] = field(default_factory=list)
    columns_dropped: List[str] = field(default_factory=list)
    encoding_applied: Dict[str, str] = field(default_factory=dict)
    scaling_applied: Dict[str, str] = field(default_factory=dict)

    # Quality metrics
    data_quality_before: Dict[str, float] = field(default_factory=dict)
    data_quality_after: Dict[str, float] = field(default_factory=dict)

    # Warnings and errors
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Processing time
    processing_time: float = 0.0

    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        return {
            'timestamp': self.timestamp,
            'original_shape': self.original_shape,
            'final_shape': self.final_shape,
            'rows_removed': self.rows_removed,
            'columns_removed': self.columns_removed,
            'missing_values_filled': self.missing_values_filled,
            'outliers_removed': self.outliers_removed,
            'duplicates_removed': self.duplicates_removed,
            'data_types_changed': self.data_types_changed,
            'text_corrections': self.text_corrections,
            'columns_created': self.columns_created,
            'columns_dropped': self.columns_dropped,
            'encoding_applied': self.encoding_applied,
            'scaling_applied': self.scaling_applied,
            'data_quality_before': self.data_quality_before,
            'data_quality_after': self.data_quality_after,
            'warnings': self.warnings,
            'errors': self.errors,
            'processing_time': self.processing_time
        }

    def to_json(self, filepath: Optional[str] = None) -> str:
        """Export report as JSON."""
        json_str = json.dumps(self.to_dict(), indent=2)
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
        return json_str

    def summary(self) -> str:
        """Generate human-readable summary."""
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║              DATA CLEANING REPORT                            ║
╚══════════════════════════════════════════════════════════════╝

📊 DATASET TRANSFORMATION
  Original Shape: {self.original_shape[0]:,} rows × {self.original_shape[1]} columns
  Final Shape:    {self.final_shape[0]:,} rows × {self.final_shape[1]} columns
  Rows Removed:   {self.rows_removed:,} ({self._percent(self.rows_removed, self.original_shape[0])})
  Columns Removed: {self.columns_removed}

🧹 CLEANING OPERATIONS
  Missing Values Filled:  {sum(self.missing_values_filled.values()):,} cells
  Outliers Removed:       {sum(self.outliers_removed.values()):,} values
  Duplicate Rows Removed: {self.duplicates_removed:,}
  Text Corrections:       {sum(len(v) for v in self.text_corrections.values())}

🔧 TRANSFORMATIONS
  Data Types Changed:  {len(self.data_types_changed)}
  Columns Created:     {len(self.columns_created)}
  Encoding Applied:    {len(self.encoding_applied)} columns
  Scaling Applied:     {len(self.scaling_applied)} columns

📈 DATA QUALITY
  Completeness: {self.data_quality_before.get('completeness', 0):.1%} → {self.data_quality_after.get('completeness', 0):.1%}
  Validity:     {self.data_quality_before.get('validity', 0):.1%} → {self.data_quality_after.get('validity', 0):.1%}

⚠️  WARNINGS: {len(self.warnings)}
❌ ERRORS:   {len(self.errors)}

⏱️  Processing Time: {self.processing_time:.2f} seconds
"""
        return summary

    def _percent(self, part: int, total: int) -> str:
        """Calculate percentage string."""
        if total == 0:
            return "0.0%"
        return f"{(part / total * 100):.1f}%"


class DataCleaner:
    """
    Ultimate automatic data cleaning and preprocessing class.

    This class provides comprehensive data cleaning capabilities with detailed
    reporting, logging, and customization options for data science workflows.

    Features:
    ---------
    ✅ Automatic data type detection and conversion
    ✅ Missing value imputation (multiple strategies)
    ✅ Outlier detection and removal (7+ methods)
    ✅ Duplicate detection and handling
    ✅ Text data cleaning and normalization
    ✅ Date/time parsing and feature extraction
    ✅ Categorical encoding (one-hot, label, target)
    ✅ Numerical scaling and normalization
    ✅ Column name standardization
    ✅ Data quality profiling
    ✅ Comprehensive reporting and logging
    ✅ Pipeline persistence (save/load configurations)

    Examples:
    ---------
    >>> # Basic usage - auto-clean everything
    >>> cleaner = DataCleaner(df)
    >>> df_clean = cleaner.clean_all()
    >>> print(cleaner.report.summary())

    >>> # Customized cleaning
    >>> cleaner = DataCleaner(df, verbose=True)
    >>> cleaner.fix_missing_values(strategy='knn')
    >>> cleaner.remove_outliers(method='isolation_forest')
    >>> cleaner.encode_categorical(method='target', target='price')
    >>> df_clean = cleaner.get_cleaned_data()

    >>> # Save configuration for reproducibility
    >>> cleaner.save_pipeline('cleaning_pipeline.json')
    >>>
    >>> # Apply same cleaning to new data
    >>> new_cleaner = DataCleaner.from_pipeline('cleaning_pipeline.json', new_df)
    >>> new_df_clean = new_cleaner.apply_pipeline()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        date_columns: Optional[List[str]] = None,
        text_columns: Optional[List[str]] = None,
        id_columns: Optional[List[str]] = None,
        drop_columns: Optional[List[str]] = None,
        missing_threshold: float = 0.5,
        outlier_threshold: float = 0.1,
        verbose: bool = True,
        log_file: Optional[str] = None
    ):
        """
        Initialize DataCleaner with a raw DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            Raw input DataFrame to be cleaned.
        target_column : str, optional
            Name of target variable (for supervised learning).
        categorical_columns : list, optional
            Explicitly specify categorical columns (auto-detected if None).
        numerical_columns : list, optional
            Explicitly specify numerical columns (auto-detected if None).
        date_columns : list, optional
            Explicitly specify date columns (auto-detected if None).
        text_columns : list, optional
            Explicitly specify text columns (auto-detected if None).
        id_columns : list, optional
            Columns that are IDs (will be excluded from processing).
        drop_columns : list, optional
            Columns to drop immediately.
        missing_threshold : float, optional (default=0.5)
            Drop columns with missing values above this threshold.
        outlier_threshold : float, optional (default=0.1)
            Maximum proportion of outliers to remove per column.
        verbose : bool, optional (default=True)
            Print progress messages.
        log_file : str, optional
            Path to save detailed logs.
        """
        self.df_original = df.copy()
        self.df = df.copy()
        self.target_column = target_column
        self.id_columns = id_columns or []
        self.columns_to_drop = drop_columns or []
        self.missing_threshold = missing_threshold
        self.outlier_threshold = outlier_threshold
        self.verbose = verbose

        # Initialize report
        self.report = CleaningReport(
            original_shape=df.shape,
            data_quality_before=self._calculate_data_quality(df)
        )

        # Setup logging
        self.logger = self._setup_logger(log_file)

        # Track column types
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.date_columns = date_columns
        self.text_columns = text_columns

        # Pipeline tracking for reproducibility
        self.pipeline_steps = []

        # Auto-detect column types if not specified
        if not all([categorical_columns, numerical_columns]):
            self._auto_detect_column_types()

        self._log("DataCleaner initialized successfully")
        self._log(f"Original shape: {df.shape}")

    def _setup_logger(self, log_file: Optional[str]) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('DataCleaner')
        logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        logger.handlers.clear()

        # Console handler
        if self.verbose:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(levelname)s: %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        return logger

    def _log(self, message: str, level: str = 'info'):
        """Log message with specified level."""
        if level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
            self.report.warnings.append(message)
        elif level == 'error':
            self.logger.error(message)
            self.report.errors.append(message)
        elif level == 'debug':
            self.logger.debug(message)

    def _calculate_data_quality(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate data quality metrics."""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        unique_rows = _comparable_frame(df).drop_duplicates().shape[0]

        return {
            'completeness': 1 - (missing_cells / total_cells) if total_cells > 0 else 0,
            'validity': 1.0,  # Will be updated during validation
            'consistency': 1.0,  # Will be updated during cleaning
            'uniqueness': unique_rows / df.shape[0] if df.shape[0] > 0 else 1.0
        }

    def _auto_detect_column_types(self):
        """Automatically detect column types."""
        self._log("Auto-detecting column types...", 'debug')

        if self.numerical_columns is None:
            self.numerical_columns = self.df.select_dtypes(
                include=['int64', 'float64', 'int32', 'float32']
            ).columns.tolist()
            # Remove target and ID columns
            self.numerical_columns = [
                col for col in self.numerical_columns
                if col != self.target_column and col not in self.id_columns
            ]

        if self.categorical_columns is None:
            self.categorical_columns = []
            for col in self.df.select_dtypes(
                include=['object', 'string', 'category']
            ).columns:
                if col in self.id_columns or col == self.target_column:
                    continue
                # Consider as categorical if unique values < 50% of total rows
                unique_ratio = _safe_nunique(self.df[col]) / len(self.df)
                if unique_ratio < 0.5:
                    self.categorical_columns.append(col)
                elif self.text_columns is None:
                    # Might be text column
                    if col not in (self.text_columns or []):
                        if self.text_columns is None:
                            self.text_columns = []
                        self.text_columns.append(col)

        if self.date_columns is None:
            self.date_columns = []
            for col in self.df.columns:
                if col in self.id_columns:
                    continue
                # Try to detect date columns
                if 'date' in col.lower() or 'time' in col.lower():
                    self.date_columns.append(col)

        self._log(f"Detected {len(self.numerical_columns)} numerical columns")
        self._log(f"Detected {len(self.categorical_columns)} categorical columns")
        self._log(f"Detected {len(self.date_columns or [])} date columns")
        self._log(f"Detected {len(self.text_columns or [])} text columns")

    # ==================== COLUMN MANAGEMENT ====================

    def standardize_column_names(self,
                                 lowercase: bool = True,
                                 remove_special: bool = True,
                                 snake_case: bool = True) -> 'DataCleaner':
        """
        Standardize column names for consistency.

        Parameters:
        -----------
        lowercase : bool
            Convert to lowercase.
        remove_special : bool
            Remove special characters.
        snake_case : bool
            Convert to snake_case.
        """
        import re

        self._log("Standardizing column names...")
        original_columns = self.df.columns.tolist()

        new_columns = []
        for col in self.df.columns:
            new_col = col

            if lowercase:
                new_col = new_col.lower()

            if remove_special:
                new_col = re.sub(r'[^a-zA-Z0-9_\s]', '', new_col)

            if snake_case:
                new_col = re.sub(r'\s+', '_', new_col)
                new_col = re.sub(r'_{2,}', '_', new_col)
                new_col = new_col.strip('_')

            new_columns.append(new_col)

        self.df.columns = new_columns
        rename_map = dict(zip(original_columns, new_columns))
        self.target_column = rename_map.get(self.target_column, self.target_column)
        self.id_columns = [rename_map.get(col, col) for col in self.id_columns]
        self.columns_to_drop = [rename_map.get(col, col) for col in self.columns_to_drop]
        if self.categorical_columns:
            self.categorical_columns = [
                rename_map.get(col, col) for col in self.categorical_columns
            ]
        if self.numerical_columns:
            self.numerical_columns = [
                rename_map.get(col, col) for col in self.numerical_columns
            ]
        if self.date_columns:
            self.date_columns = [rename_map.get(col, col) for col in self.date_columns]
        if self.text_columns:
            self.text_columns = [rename_map.get(col, col) for col in self.text_columns]
        self._log(f"Renamed {len([i for i, (o, n) in enumerate(zip(original_columns, new_columns)) if o != n])} columns")

        self.pipeline_steps.append({
            'step': 'standardize_column_names',
            'params': {'lowercase': lowercase, 'remove_special': remove_special, 'snake_case': snake_case}
        })

        return self

    def rename_columns(self, mapping: Dict[str, str]) -> 'DataCleaner':
        """Rename columns while keeping DataCleaner metadata synchronized.

        Parameters:
        -----------
        mapping : dict
            Mapping from existing names to new names. Missing source columns are
            ignored, matching pandas ``DataFrame.rename`` behavior.
        """
        if not isinstance(mapping, dict):
            raise TypeError("mapping must be a dictionary")

        effective_mapping = {
            old: new for old, new in mapping.items() if old in self.df.columns
        }
        renamed_columns = [effective_mapping.get(col, col) for col in self.df.columns]
        if len(set(renamed_columns)) != len(renamed_columns):
            raise ValueError("Column mapping would create duplicate column names")

        self.df = self.df.rename(columns=effective_mapping)
        self.target_column = effective_mapping.get(self.target_column, self.target_column)
        self.id_columns = [effective_mapping.get(col, col) for col in self.id_columns]
        self.columns_to_drop = [
            effective_mapping.get(col, col) for col in self.columns_to_drop
        ]
        for attribute in (
            'categorical_columns',
            'numerical_columns',
            'date_columns',
            'text_columns',
        ):
            values = getattr(self, attribute)
            if values:
                setattr(
                    self,
                    attribute,
                    [effective_mapping.get(col, col) for col in values],
                )

        changed = sum(old != new for old, new in effective_mapping.items())
        self._log(f"Renamed {changed} columns")
        self.pipeline_steps.append({
            'step': 'rename_columns',
            'params': {'mapping': mapping}
        })
        return self

    def drop_columns(self, columns: Optional[List[str]] = None,
                    missing_threshold: Optional[float] = None) -> 'DataCleaner':
        """
        Drop specified columns or columns with too many missing values.

        Parameters:
        -----------
        columns : list, optional
            Specific columns to drop.
        missing_threshold : float, optional
            Drop columns with missing ratio above this threshold.
        """
        self._log("Dropping columns...")

        to_drop = set()

        # Drop specified columns
        explicit_columns = self.columns_to_drop if columns is None else columns
        if explicit_columns:
            to_drop.update([col for col in explicit_columns if col in self.df.columns])

        # Drop columns with too many missing values
        threshold = missing_threshold or self.missing_threshold
        missing_ratios = self.df.isnull().sum() / len(self.df)
        high_missing = missing_ratios[missing_ratios > threshold].index.tolist()
        to_drop.update(high_missing)

        # Don't drop target column
        if self.target_column and self.target_column in to_drop:
            to_drop.remove(self.target_column)
            self._log(f"Keeping target column '{self.target_column}' despite high missing values", 'warning')

        if to_drop:
            self.df = self.df.drop(columns=list(to_drop))
            self.report.columns_dropped.extend(list(to_drop))
            self.report.columns_removed += len(to_drop)
            self._log(f"Dropped {len(to_drop)} columns: {list(to_drop)}")
        else:
            self._log("No columns to drop")

        self.pipeline_steps.append({
            'step': 'drop_columns',
            'params': {'columns': columns, 'missing_threshold': threshold}
        })

        return self

    def drop_missing_rows(
        self,
        subset: Optional[List[str]] = None,
        how: str = 'any',
        min_non_null: Optional[int] = None,
        treat_blank_as_missing: bool = False,
    ) -> 'DataCleaner':
        """Drop rows missing required values, optionally treating blanks as nulls.

        Parameters:
        -----------
        subset : list, optional
            Columns used to determine whether a row is missing.
        how : {'any', 'all'}
            Drop a row when any or all selected values are missing.
        min_non_null : int, optional
            Keep rows with at least this many non-null selected values. Cannot
            be combined with a non-default ``how`` value.
        treat_blank_as_missing : bool
            Treat empty or whitespace-only strings as missing before filtering.
        """
        if how not in {'any', 'all'}:
            raise ValueError("how must be 'any' or 'all'")
        if min_non_null is not None and how != 'any':
            raise ValueError("min_non_null cannot be combined with how='all'")
        selected = list(subset) if subset is not None else list(self.df.columns)
        missing_columns = [col for col in selected if col not in self.df.columns]
        if missing_columns:
            raise KeyError(f"Columns not found: {missing_columns}")

        if treat_blank_as_missing:
            for col in selected:
                if pd.api.types.is_object_dtype(self.df[col]) or pd.api.types.is_string_dtype(self.df[col]):
                    blank_mask = self.df[col].astype('string').str.strip().eq('')
                    self.df.loc[blank_mask.fillna(False), col] = pd.NA

        before_count = len(self.df)
        if min_non_null is not None:
            if min_non_null < 0 or min_non_null > len(selected):
                raise ValueError("min_non_null must be between 0 and the number of selected columns")
            self.df = self.df.dropna(subset=selected, thresh=min_non_null)
        else:
            self.df = self.df.dropna(subset=selected, how=how)
        removed = before_count - len(self.df)
        self.report.rows_removed += removed
        self._log(f"Removed {removed} rows with missing required values")
        self.pipeline_steps.append({
            'step': 'drop_missing_rows',
            'params': {
                'subset': subset,
                'how': how,
                'min_non_null': min_non_null,
                'treat_blank_as_missing': treat_blank_as_missing,
            }
        })
        return self

    # ==================== MISSING VALUES ====================

    def fix_missing_values(self,
                          strategy: Union[str, Dict[str, str]] = 'auto',
                          fill_value: Any = None) -> 'DataCleaner':
        """
        Handle missing values with various strategies.

        Parameters:
        -----------
        strategy : str or dict
            Strategy for filling missing values:
            - 'auto': Automatically choose best strategy per column
            - 'mean', 'median', 'mode': Statistical imputation
            - 'ffill', 'bfill': Forward/backward fill
            - 'knn': K-Nearest Neighbors imputation
            - 'iterative': Iterative imputation
            - dict: Column-specific strategies {'col1': 'mean', 'col2': 'mode'}
        fill_value : any, optional
            Specific value to fill (overrides strategy).
        """
        self._log(f"Fixing missing values with strategy: {strategy}...")

        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import KNNImputer, IterativeImputer
        missing_before = self.df.isnull().sum().sum()

        if fill_value is not None:
            # Simple fill with specific value
            self.df = self.df.fillna(fill_value)
            filled = missing_before - self.df.isnull().sum().sum()
            self._log(f"Filled {filled} missing values with {fill_value}")
            self.report.missing_values_filled['all_columns'] = filled

        elif strategy == 'auto':
            # Auto-select strategy per column type
            for col in self.df.columns:
                if col in self.id_columns or col == self.target_column:
                    continue

                missing_count = self.df[col].isnull().sum()
                if missing_count == 0:
                    continue

                if col in self.numerical_columns:
                    # Use median for numerical
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                    self.report.missing_values_filled[col] = missing_count
                    self._log(f"  {col}: filled {missing_count} with median", 'debug')

                elif col in self.categorical_columns:
                    # Use mode for categorical
                    mode_val = self.df[col].mode()
                    if len(mode_val) > 0:
                        self.df[col] = self.df[col].fillna(mode_val[0])
                        self.report.missing_values_filled[col] = missing_count
                        self._log(f"  {col}: filled {missing_count} with mode", 'debug')

        elif isinstance(strategy, dict):
            # Column-specific strategies
            for col, col_strategy in strategy.items():
                if col not in self.df.columns:
                    continue
                missing_count = self.df[col].isnull().sum()
                if missing_count == 0:
                    continue

                if col_strategy in ['mean', 'median']:
                    self.df[col] = self.df[col].fillna(self.df[col].agg(col_strategy))
                elif col_strategy == 'mode':
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                elif col_strategy == 'ffill':
                    self.df[col] = self.df[col].fillna(method='ffill')
                elif col_strategy == 'bfill':
                    self.df[col] = self.df[col].fillna(method='bfill')

                self.report.missing_values_filled[col] = missing_count
                self._log(f"  {col}: filled {missing_count} with {col_strategy}", 'debug')

        elif strategy == 'knn':
            # KNN imputation for numerical columns
            if self.numerical_columns:
                imputer = KNNImputer(n_neighbors=5)
                self.df[self.numerical_columns] = imputer.fit_transform(
                    self.df[self.numerical_columns]
                )
                filled = sum(self.report.missing_values_filled.values())
                self._log(f"Applied KNN imputation to {len(self.numerical_columns)} numerical columns")

        elif strategy == 'iterative':
            # Iterative imputation
            if self.numerical_columns:
                imputer = IterativeImputer(max_iter=10, random_state=42)
                self.df[self.numerical_columns] = imputer.fit_transform(
                    self.df[self.numerical_columns]
                )
                self._log(f"Applied iterative imputation to {len(self.numerical_columns)} numerical columns")

        else:
            # Simple strategy for all columns
            for col in self.df.columns:
                if col in self.id_columns:
                    continue
                missing_count = self.df[col].isnull().sum()
                if missing_count == 0:
                    continue

                if strategy in ['mean', 'median']:
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        self.df[col] = self.df[col].fillna(self.df[col].agg(strategy))
                        self.report.missing_values_filled[col] = missing_count
                elif strategy == 'mode':
                    mode_val = self.df[col].mode()
                    if len(mode_val) > 0:
                        self.df[col] = self.df[col].fillna(mode_val[0])
                        self.report.missing_values_filled[col] = missing_count
                elif strategy in ['ffill', 'bfill']:
                    self.df[col] = self.df[col].fillna(method=strategy)
                    self.report.missing_values_filled[col] = missing_count

        missing_after = self.df.isnull().sum().sum()
        self._log(f"Missing values: {missing_before} → {missing_after}")

        self.pipeline_steps.append({
            'step': 'fix_missing_values',
            'params': {'strategy': strategy, 'fill_value': fill_value}
        })

        return self

    # ==================== OUTLIERS ====================

    def remove_outliers(self,
                       method: str = 'iqr',
                       columns: Optional[List[str]] = None,
                       **kwargs) -> 'DataCleaner':
        """
        Remove outliers from numerical columns.

        Parameters:
        -----------
        method : str
            Outlier detection method: 'iqr', 'zscore', 'isolation_forest',
            'lof', 'mad', 'percentile'.
        columns : list, optional
            Specific columns to check (default: all numerical).
        **kwargs :
            Method-specific parameters.
        """
        self._log(f"Removing outliers using {method} method...")

        from scipy.stats import zscore
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor

        cols_to_check = columns or self.numerical_columns
        initial_rows = len(self.df)

        for col in cols_to_check:
            if col not in self.df.columns or col == self.target_column:
                continue

            before_count = len(self.df)

            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                factor = kwargs.get('factor', 1.5)
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]

            elif method == 'zscore':
                threshold = kwargs.get('threshold', 3)
                self.df = self.df[np.abs(zscore(self.df[col].fillna(self.df[col].median()))) < threshold]

            elif method == 'mad':
                threshold = kwargs.get('threshold', 3.5)
                median = self.df[col].median()
                mad = np.median(np.abs(self.df[col] - median))
                if mad != 0:
                    modified_z = 0.6745 * (self.df[col] - median) / mad
                    self.df = self.df[np.abs(modified_z) < threshold]

            elif method == 'percentile':
                lower = kwargs.get('lower', 0.01)
                upper = kwargs.get('upper', 0.99)
                lower_bound = self.df[col].quantile(lower)
                upper_bound = self.df[col].quantile(upper)
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]

            after_count = len(self.df)
            removed = before_count - after_count

            if removed > 0:
                self.report.outliers_removed[col] = removed
                self._log(f"  {col}: removed {removed} outliers", 'debug')

        # Handle multivariate methods
        if method == 'isolation_forest':
            contamination = kwargs.get('contamination', 0.1)
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            predictions = iso_forest.fit_predict(self.df[cols_to_check].fillna(0))
            outlier_count = (predictions == -1).sum()
            self.df = self.df[predictions == 1]
            self.report.outliers_removed['multivariate'] = outlier_count
            self._log(f"Removed {outlier_count} outliers using Isolation Forest")

        elif method == 'lof':
            n_neighbors = kwargs.get('n_neighbors', 20)
            contamination = kwargs.get('contamination', 0.1)
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            predictions = lof.fit_predict(self.df[cols_to_check].fillna(0))
            outlier_count = (predictions == -1).sum()
            self.df = self.df[predictions == 1]
            self.report.outliers_removed['multivariate'] = outlier_count
            self._log(f"Removed {outlier_count} outliers using LOF")

        total_removed = initial_rows - len(self.df)
        self.report.rows_removed += total_removed
        self._log(f"Total outliers removed: {total_removed}")

        self.pipeline_steps.append({
            'step': 'remove_outliers',
            'params': {'method': method, 'columns': columns, **kwargs}
        })

        return self

    # ==================== DUPLICATES ====================

    def remove_duplicates(self,
                         subset: Optional[List[str]] = None,
                         keep: str = 'first') -> 'DataCleaner':
        """
        Remove duplicate rows.

        Parameters:
        -----------
        subset : list, optional
            Only consider certain columns for identifying duplicates.
        keep : str
            Which duplicates to keep: 'first', 'last', or False (remove all).
        """
        self._log("Removing duplicates...")

        before_count = len(self.df)
        duplicate_mask = _comparable_frame(self.df).duplicated(
            subset=subset,
            keep=keep,
        )
        self.df = self.df.loc[~duplicate_mask].copy()
        after_count = len(self.df)

        duplicates_removed = before_count - after_count
        self.report.duplicates_removed = duplicates_removed
        self.report.rows_removed += duplicates_removed

        self._log(f"Removed {duplicates_removed} duplicate rows")

        self.pipeline_steps.append({
            'step': 'remove_duplicates',
            'params': {'subset': subset, 'keep': keep}
        })

        return self

    # ==================== TEXT CLEANING ====================

    def clean_text_columns(self,
                          columns: Optional[List[str]] = None,
                          lowercase: bool = True,
                          remove_special: bool = True,
                          remove_numbers: bool = False,
                          fix_typos: bool = True) -> 'DataCleaner':
        """
        Clean text columns.

        Parameters:
        -----------
        columns : list, optional
            Columns to clean (default: all text columns).
        lowercase : bool
            Convert to lowercase.
        remove_special : bool
            Remove special characters.
        remove_numbers : bool
            Remove numbers.
        fix_typos : bool
            Fix common typos.
        """
        import re

        self._log("Cleaning text columns...")

        cols_to_clean = columns or self.text_columns or []

        common_typos = {
            'teh': 'the', 'adn': 'and', 'recieve': 'receive',
            'occured': 'occurred', 'seperate': 'separate'
        }

        for col in cols_to_clean:
            if col not in self.df.columns:
                continue

            corrections = 0

            for idx, val in self.df[col].items():
                if pd.isna(val):
                    continue

                original = str(val)
                cleaned = original

                if lowercase:
                    cleaned = cleaned.lower()

                if remove_special:
                    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned)

                if remove_numbers:
                    cleaned = re.sub(r'\d+', '', cleaned)

                # Fix multiple spaces
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()

                if fix_typos:
                    words = cleaned.split()
                    fixed_words = [common_typos.get(word, word) for word in words]
                    cleaned = ' '.join(fixed_words)

                if cleaned != original:
                    self.df.at[idx, col] = cleaned
                    corrections += 1

            if corrections > 0:
                if col not in self.report.text_corrections:
                    self.report.text_corrections[col] = []
                self.report.text_corrections[col].append(f"{corrections} values cleaned")
                self._log(f"  {col}: cleaned {corrections} text values", 'debug')

        self.pipeline_steps.append({
            'step': 'clean_text_columns',
            'params': {
                'columns': columns, 'lowercase': lowercase,
                'remove_special': remove_special, 'remove_numbers': remove_numbers,
                'fix_typos': fix_typos
            }
        })

        return self

    # ==================== DATA TYPE CONVERSION ====================

    def coerce_numeric(
        self,
        columns: List[str],
        fill_value: Any = None,
        downcast: Optional[str] = None,
        errors: str = 'coerce',
    ) -> 'DataCleaner':
        """Convert selected columns with ``pandas.to_numeric`` semantics."""
        if errors not in {'raise', 'coerce'}:
            raise ValueError("errors must be 'raise' or 'coerce'")
        missing_columns = [col for col in columns if col not in self.df.columns]
        if missing_columns:
            raise KeyError(f"Columns not found: {missing_columns}")

        for col in columns:
            original_dtype = str(self.df[col].dtype)
            self.df[col] = pd.to_numeric(
                self.df[col],
                errors=errors,
                downcast=downcast,
            )
            if isinstance(fill_value, dict):
                if col in fill_value:
                    self.df[col] = self.df[col].fillna(fill_value[col])
            elif fill_value is not None:
                self.df[col] = self.df[col].fillna(fill_value)
            new_dtype = str(self.df[col].dtype)
            if original_dtype != new_dtype:
                self.report.data_types_changed[col] = (original_dtype, new_dtype)

        self._log(f"Coerced {len(columns)} columns to numeric values")
        self.pipeline_steps.append({
            'step': 'coerce_numeric',
            'params': {
                'columns': columns,
                'fill_value': fill_value,
                'downcast': downcast,
                'errors': errors,
            }
        })
        return self

    def convert_data_types(self, auto_convert: bool = True,
                          conversions: Optional[Dict[str, str]] = None) -> 'DataCleaner':
        """
        Convert data types optimally.

        Parameters:
        -----------
        auto_convert : bool
            Automatically optimize data types.
        conversions : dict, optional
            Explicit conversions {'column': 'dtype'}.
        """
        self._log("Converting data types...")

        if auto_convert:
            # Optimize numerical columns
            for col in self.numerical_columns:
                if col not in self.df.columns:
                    continue

                original_dtype = str(self.df[col].dtype)

                # Try to downcast integers
                if 'int' in original_dtype:
                    self.df[col] = pd.to_numeric(self.df[col], downcast='integer')

                # Try to downcast floats
                elif 'float' in original_dtype:
                    self.df[col] = pd.to_numeric(self.df[col], downcast='float')

                new_dtype = str(self.df[col].dtype)
                if original_dtype != new_dtype:
                    self.report.data_types_changed[col] = (original_dtype, new_dtype)
                    self._log(f"  {col}: {original_dtype} → {new_dtype}", 'debug')

            # Convert categorical columns
            for col in self.categorical_columns:
                if col not in self.df.columns:
                    continue

                original_dtype = str(self.df[col].dtype)
                if self.df[col].dtype == 'object':
                    # Convert to category if cardinality is low
                    if _safe_nunique(self.df[col]) / len(self.df) < 0.5:
                        self.df[col] = self.df[col].astype('category')
                        new_dtype = 'category'
                        self.report.data_types_changed[col] = (original_dtype, new_dtype)
                        self._log(f"  {col}: {original_dtype} → {new_dtype}", 'debug')

        if conversions:
            for col, dtype in conversions.items():
                if col not in self.df.columns:
                    continue

                try:
                    original_dtype = str(self.df[col].dtype)
                    self.df[col] = self.df[col].astype(dtype)
                    new_dtype = str(self.df[col].dtype)
                    self.report.data_types_changed[col] = (original_dtype, new_dtype)
                    self._log(f"  {col}: {original_dtype} → {new_dtype}", 'debug')
                except Exception as e:
                    self._log(f"Failed to convert {col} to {dtype}: {e}", 'warning')

        self.pipeline_steps.append({
            'step': 'convert_data_types',
            'params': {'auto_convert': auto_convert, 'conversions': conversions}
        })

        return self

    # ==================== DATE HANDLING ====================

    def parse_dates(
        self,
        columns: Optional[List[str]] = None,
        extract_features: bool = True,
        date_format: Optional[Union[str, Dict[str, str]]] = None,
        utc: Union[bool, Dict[str, bool]] = False,
        errors: str = 'coerce',
    ) -> 'DataCleaner':
        """
        Parse date columns and extract features.

        Parameters:
        -----------
        columns : list, optional
            Date columns to parse (auto-detected if None).
        extract_features : bool
            Extract year, month, day, etc. as separate columns.
        date_format : str or dict, optional
            One format for all columns or a mapping of column to format.
        utc : bool or dict
            Parse all columns as UTC or configure UTC per column.
        errors : {'raise', 'coerce'}
            Invalid parsing behavior.
        """
        self._log("Parsing date columns...")

        date_cols = columns or self.date_columns or []

        for col in date_cols:
            if col not in self.df.columns:
                continue

            try:
                column_format = (
                    date_format.get(col)
                    if isinstance(date_format, dict)
                    else date_format
                )
                column_utc = utc.get(col, False) if isinstance(utc, dict) else utc
                self.df[col] = pd.to_datetime(
                    self.df[col],
                    errors=errors,
                    format=column_format,
                    utc=column_utc,
                )
                self._log(f"  {col}: parsed as datetime", 'debug')

                if extract_features:
                    self.df[f'{col}_year'] = self.df[col].dt.year
                    self.df[f'{col}_month'] = self.df[col].dt.month
                    self.df[f'{col}_day'] = self.df[col].dt.day
                    self.df[f'{col}_dayofweek'] = self.df[col].dt.dayofweek
                    self.df[f'{col}_quarter'] = self.df[col].dt.quarter

                    new_features = [f'{col}_year', f'{col}_month', f'{col}_day',
                                  f'{col}_dayofweek', f'{col}_quarter']
                    self.report.columns_created.extend(new_features)
                    self._log(f"    Extracted {len(new_features)} date features", 'debug')

            except Exception as e:
                self._log(f"Failed to parse {col} as date: {e}", 'warning')

        self.pipeline_steps.append({
            'step': 'parse_dates',
            'params': {
                'columns': columns,
                'extract_features': extract_features,
                'date_format': date_format,
                'utc': utc,
                'errors': errors,
            }
        })

        return self

    # ==================== ENCODING ====================

    def encode_categorical(self,
                          method: str = 'onehot',
                          columns: Optional[List[str]] = None,
                          drop_first: bool = False,
                          **kwargs) -> 'DataCleaner':
        """
        Encode categorical variables.

        Parameters:
        -----------
        method : str
            Encoding method: 'onehot', 'label', 'target', 'binary', 'ordinal'.
        columns : list, optional
            Columns to encode (default: all categorical).
        drop_first : bool
            Drop first category for one-hot encoding (avoid multicollinearity).
        **kwargs :
            Method-specific parameters.
        """
        self._log(f"Encoding categorical variables using {method} method...")

        from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

        cols_to_encode = columns or self.categorical_columns

        if method == 'onehot':
            # One-hot encoding
            encoded_df = pd.get_dummies(
                self.df[cols_to_encode],
                prefix=cols_to_encode,
                drop_first=drop_first
            )

            # Drop original columns and add encoded ones
            self.df = self.df.drop(columns=cols_to_encode)
            self.df = pd.concat([self.df, encoded_df], axis=1)

            new_cols = encoded_df.columns.tolist()
            self.report.columns_created.extend(new_cols)
            self.report.columns_dropped.extend(cols_to_encode)

            for col in cols_to_encode:
                self.report.encoding_applied[col] = 'onehot'

            self._log(f"Created {len(new_cols)} one-hot encoded columns")

        elif method == 'label':
            # Label encoding
            le = LabelEncoder()
            for col in cols_to_encode:
                if col not in self.df.columns:
                    continue
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.report.encoding_applied[col] = 'label'
                self._log(f"  {col}: label encoded", 'debug')

        elif method == 'target':
            # Target encoding (requires target column)
            if not self.target_column or self.target_column not in self.df.columns:
                self._log("Target encoding requires a valid target column", 'error')
                return self

            for col in cols_to_encode:
                if col not in self.df.columns or col == self.target_column:
                    continue

                # Calculate mean target per category
                target_means = self.df.groupby(col)[self.target_column].mean()
                self.df[f'{col}_target_enc'] = self.df[col].map(target_means)
                self.report.columns_created.append(f'{col}_target_enc')
                self.report.encoding_applied[col] = 'target'
                self._log(f"  {col}: target encoded", 'debug')

        elif method == 'ordinal':
            # Ordinal encoding
            categories = kwargs.get('categories', None)
            oe = OrdinalEncoder(categories=categories if categories else 'auto')

            for col in cols_to_encode:
                if col not in self.df.columns:
                    continue
                self.df[[col]] = oe.fit_transform(self.df[[col]])
                self.report.encoding_applied[col] = 'ordinal'
                self._log(f"  {col}: ordinal encoded", 'debug')

        self.pipeline_steps.append({
            'step': 'encode_categorical',
            'params': {'method': method, 'columns': columns, 'drop_first': drop_first, **kwargs}
        })

        return self

    # ==================== SCALING ====================

    def scale_features(self,
                      method: str = 'standard',
                      columns: Optional[List[str]] = None) -> 'DataCleaner':
        """
        Scale numerical features.

        Parameters:
        -----------
        method : str
            Scaling method: 'standard', 'minmax', 'robust', 'maxabs'.
        columns : list, optional
            Columns to scale (default: all numerical).
        """
        self._log(f"Scaling features using {method} method...")

        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

        cols_to_scale = columns or self.numerical_columns
        cols_to_scale = [c for c in cols_to_scale if c in self.df.columns and c != self.target_column]

        if not cols_to_scale:
            self._log("No columns to scale", 'warning')
            return self

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'maxabs':
            scaler = MaxAbsScaler()
        else:
            self._log(f"Unknown scaling method: {method}", 'error')
            return self

        self.df[cols_to_scale] = scaler.fit_transform(self.df[cols_to_scale])

        for col in cols_to_scale:
            self.report.scaling_applied[col] = method

        self._log(f"Scaled {len(cols_to_scale)} numerical columns")

        self.pipeline_steps.append({
            'step': 'scale_features',
            'params': {'method': method, 'columns': columns}
        })

        return self

    # ==================== ALL-IN-ONE CLEANING ====================

    def clean_all(self,
                 missing_strategy: str = 'auto',
                 outlier_method: str = 'iqr',
                 encode_method: str = 'onehot',
                 scale_method: str = 'standard',
                 remove_duplicates: bool = True,
                 standardize_columns: bool = True,
                 parse_dates: bool = True) -> pd.DataFrame:
        """
        Perform comprehensive data cleaning with sensible defaults.

        This is the one-stop method that applies all cleaning operations.

        Parameters:
        -----------
        missing_strategy : str
            Strategy for missing values.
        outlier_method : str
            Method for outlier detection.
        encode_method : str
            Categorical encoding method.
        scale_method : str
            Feature scaling method.
        remove_duplicates : bool
            Remove duplicate rows.
        standardize_columns : bool
            Standardize column names.
        parse_dates : bool
            Parse and extract date features.

        Returns:
        --------
        pd.DataFrame
            Cleaned DataFrame.
        """
        import time
        start_time = time.time()

        self._log("=" * 60)
        self._log("STARTING COMPREHENSIVE DATA CLEANING")
        self._log("=" * 60)

        # Step 1: Column management
        if standardize_columns:
            self.standardize_column_names()

        self.drop_columns()

        # Step 2: Remove duplicates
        if remove_duplicates:
            self.remove_duplicates()

        # Step 3: Handle missing values
        self.fix_missing_values(strategy=missing_strategy)

        # Step 4: Data type conversion
        self.convert_data_types()

        # Step 5: Parse dates
        if parse_dates and self.date_columns:
            self.parse_dates()

        # Step 6: Clean text columns
        if self.text_columns:
            self.clean_text_columns()

        # Step 7: Remove outliers
        if self.numerical_columns:
            self.remove_outliers(method=outlier_method)

        # Step 8: Encode categorical variables
        if self.categorical_columns and encode_method:
            self.encode_categorical(method=encode_method)

        # Step 9: Scale features
        if self.numerical_columns and scale_method:
            self.scale_features(method=scale_method)

        # Finalize report
        self.report.final_shape = self.df.shape
        self.report.data_quality_after = self._calculate_data_quality(self.df)
        self.report.processing_time = time.time() - start_time

        self._log("=" * 60)
        self._log("CLEANING COMPLETED")
        self._log("=" * 60)

        if self.verbose:
            print(self.report.summary())

        return self.df

    # ==================== UTILITY METHODS ====================

    def get_cleaned_data(self) -> pd.DataFrame:
        """Get the cleaned DataFrame."""
        return self.df.copy()

    def get_report(self) -> CleaningReport:
        """Get the cleaning report."""
        return self.report

    def profile_data(self) -> Dict[str, Any]:
        """
        Generate comprehensive data profile.

        Returns:
        --------
        dict
            Data profile with statistics, quality metrics, and recommendations.
        """
        profile = {
            'shape': self.df.shape,
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'columns': {},
            'missing_values': {},
            'duplicates': int(_comparable_frame(self.df).duplicated().sum()),
            'data_quality': self._calculate_data_quality(self.df)
        }

        for col in self.df.columns:
            unique_count = _safe_nunique(self.df[col])
            col_profile = {
                'dtype': str(self.df[col].dtype),
                'missing': self.df[col].isnull().sum(),
                'missing_percent': self.df[col].isnull().sum() / len(self.df) * 100,
                'unique': unique_count,
                'unique_percent': unique_count / len(self.df) * 100
            }

            if pd.api.types.is_numeric_dtype(self.df[col]):
                col_profile.update({
                    'mean': self.df[col].mean(),
                    'std': self.df[col].std(),
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'median': self.df[col].median()
                })

            profile['columns'][col] = col_profile

        return profile

    def save_pipeline(self, filepath: str):
        """Save cleaning pipeline configuration for reproducibility."""
        pipeline_config = {
            'steps': self.pipeline_steps,
            'target_column': self.target_column,
            'id_columns': self.id_columns,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'date_columns': self.date_columns,
            'text_columns': self.text_columns,
            'missing_threshold': self.missing_threshold,
            'outlier_threshold': self.outlier_threshold
        }

        with open(filepath, 'w') as f:
            json.dump(pipeline_config, f, indent=2)

        self._log(f"Pipeline saved to {filepath}")

    @classmethod
    def from_pipeline(cls, filepath: str, df: pd.DataFrame) -> 'DataCleaner':
        """Load and apply saved pipeline to new data."""
        with open(filepath, 'r') as f:
            config = json.load(f)

        cleaner = cls(
            df,
            target_column=config.get('target_column'),
            categorical_columns=config.get('categorical_columns'),
            numerical_columns=config.get('numerical_columns'),
            date_columns=config.get('date_columns'),
            text_columns=config.get('text_columns'),
            id_columns=config.get('id_columns'),
            missing_threshold=config.get('missing_threshold', 0.5),
            outlier_threshold=config.get('outlier_threshold', 0.1)
        )

        return cleaner

    def apply_pipeline(self) -> pd.DataFrame:
        """Apply saved pipeline steps."""
        for step_config in self.pipeline_steps:
            step = step_config['step']
            params = step_config['params']

            method = getattr(self, step, None)
            if method:
                method(**params)

        return self.df

    def reset(self):
        """Reset to original DataFrame."""
        self.df = self.df_original.copy()
        self.report = CleaningReport(
            original_shape=self.df_original.shape,
            data_quality_before=self._calculate_data_quality(self.df_original)
        )
        self.pipeline_steps = []
        self._log("Reset to original DataFrame")

    def compare_with_original(self) -> pd.DataFrame:
        """Compare cleaned data with original."""
        comparison = pd.DataFrame({
            'Metric': [
                'Rows',
                'Columns',
                'Missing Values',
                'Duplicates',
                'Memory (MB)'
            ],
            'Original': [
                self.df_original.shape[0],
                self.df_original.shape[1],
                self.df_original.isnull().sum().sum(),
                _comparable_frame(self.df_original).duplicated().sum(),
                self.df_original.memory_usage(deep=True).sum() / 1024**2
            ],
            'Cleaned': [
                self.df.shape[0],
                self.df.shape[1],
                self.df.isnull().sum().sum(),
                _comparable_frame(self.df).duplicated().sum(),
                self.df.memory_usage(deep=True).sum() / 1024**2
            ]
        })

        comparison['Change'] = comparison['Cleaned'] - comparison['Original']
        comparison['Change %'] = (comparison['Change'] / comparison['Original'] * 100).round(2)

        return comparison


# ==================== CONVENIENCE FUNCTIONS ====================

def quick_clean(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Quick cleaning with default settings."""
    cleaner = DataCleaner(df, **kwargs)
    return cleaner.clean_all()


def advanced_clean(df: pd.DataFrame,
                  target: Optional[str] = None,
                  report_path: Optional[str] = None) -> Tuple[pd.DataFrame, CleaningReport]:
    """Advanced cleaning with full report."""
    cleaner = DataCleaner(df, target_column=target, verbose=True)
    df_clean = cleaner.clean_all()

    if report_path:
        cleaner.report.to_json(report_path)

    return df_clean, cleaner.get_report()
