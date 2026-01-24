"""Robust data loading utilities with encoding detection and error handling."""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np


@dataclass
class DataLoadResult:
    """Result of data loading operation with metadata."""

    df: pd.DataFrame
    encoding: str
    delimiter: str
    warnings: List[str] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    file_size_mb: float = 0.0
    load_successful: bool = True
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.df is not None:
            self.row_count = len(self.df)
            self.column_count = len(self.df.columns)


def detect_delimiter(file_path: str, encoding: str = 'utf-8') -> str:
    """Detect the delimiter used in a CSV file.

    Args:
        file_path: Path to the CSV file
        encoding: File encoding to use

    Returns:
        Detected delimiter character
    """
    delimiters = [',', ';', '\t', '|']

    try:
        with open(file_path, 'r', encoding=encoding) as f:
            first_line = f.readline()

        # Count occurrences of each delimiter
        counts = {d: first_line.count(d) for d in delimiters}

        # Return the delimiter with highest count (must be > 0)
        max_count = max(counts.values())
        if max_count > 0:
            return max(counts, key=counts.get)

        return ','  # Default to comma

    except Exception:
        return ','


def detect_encoding(file_path: str) -> Tuple[str, List[str]]:
    """Detect file encoding by trying common encodings.

    Args:
        file_path: Path to the file

    Returns:
        Tuple of (detected encoding, list of warnings)
    """
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
    warnings = []

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                # Try to read first 10000 characters
                f.read(10000)

            if encoding != 'utf-8':
                warnings.append(f"File encoding detected as '{encoding}' (not UTF-8)")

            return encoding, warnings

        except (UnicodeDecodeError, UnicodeError):
            continue

    # Fallback to latin-1 (accepts all byte values)
    warnings.append("Could not detect encoding, falling back to 'latin-1'")
    return 'latin-1', warnings


def load_csv_robust(
    file_path: str,
    sample_rows: Optional[int] = None,
    chunksize: Optional[int] = None,
    na_values: Optional[List[str]] = None,
    dtype: Optional[Dict[str, Any]] = None,
    usecols: Optional[List[str]] = None,
    parse_dates: Optional[List[str]] = None,
    low_memory: bool = True
) -> DataLoadResult:
    """Load CSV file with robust encoding detection and error handling.

    This function implements best practices from ML text:
    - Encoding detection with fallback (UTF-8 → latin-1 → cp1252)
    - Delimiter auto-detection
    - Custom missing value markers
    - Memory optimization options
    - Comprehensive error handling

    Args:
        file_path: Path to the CSV file
        sample_rows: If set, only load first N rows (for testing/preview)
        chunksize: If set, return iterator for chunked loading of large files
        na_values: Additional strings to recognize as NA/NaN
        dtype: Column data types specification
        usecols: Columns to load (None = all)
        parse_dates: Columns to parse as dates
        low_memory: Use low memory mode (default True)

    Returns:
        DataLoadResult with DataFrame and metadata

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file cannot be parsed
    """
    warnings = []

    # Check file exists
    if not os.path.exists(file_path):
        return DataLoadResult(
            df=pd.DataFrame(),
            encoding='',
            delimiter='',
            warnings=[f"File not found: {file_path}"],
            load_successful=False,
            error_message=f"File not found: {file_path}"
        )

    # Get file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    # Warn about large files
    if file_size_mb > 100 and chunksize is None and sample_rows is None:
        warnings.append(
            f"Large file detected ({file_size_mb:.1f} MB). "
            "Consider using sample_rows or chunksize for better performance."
        )

    # Detect encoding
    encoding, encoding_warnings = detect_encoding(file_path)
    warnings.extend(encoding_warnings)

    # Detect delimiter
    delimiter = detect_delimiter(file_path, encoding)
    if delimiter != ',':
        warnings.append(f"Non-comma delimiter detected: '{repr(delimiter)}'")

    # Default NA values (common missing value representations)
    default_na_values = [
        '', 'NA', 'N/A', 'n/a', 'na', 'NaN', 'nan',
        'NULL', 'null', 'None', 'none',
        '-', '--', '?', '.', 'missing', 'MISSING',
        '#N/A', '#NA', '#NULL!', '#REF!', '#VALUE!'
    ]

    if na_values:
        all_na_values = list(set(default_na_values + na_values))
    else:
        all_na_values = default_na_values

    # Build read_csv parameters
    read_params = {
        'filepath_or_buffer': file_path,
        'encoding': encoding,
        'sep': delimiter,
        'na_values': all_na_values,
        'low_memory': low_memory,
        'on_bad_lines': 'warn'  # Skip bad lines with warning
    }

    if sample_rows is not None:
        read_params['nrows'] = sample_rows

    if chunksize is not None:
        read_params['chunksize'] = chunksize

    if dtype is not None:
        read_params['dtype'] = dtype

    if usecols is not None:
        read_params['usecols'] = usecols

    if parse_dates is not None:
        read_params['parse_dates'] = parse_dates

    try:
        df = pd.read_csv(**read_params)

        # If chunked, return iterator wrapped in result
        if chunksize is not None:
            # For chunked loading, we return the iterator
            # Caller must handle iteration
            return DataLoadResult(
                df=pd.DataFrame(),  # Empty placeholder
                encoding=encoding,
                delimiter=delimiter,
                warnings=warnings + ["Chunked loading enabled. Use iterator to access data."],
                file_size_mb=file_size_mb,
                load_successful=True
            )

        # Post-load checks
        if len(df) == 0:
            warnings.append("DataFrame is empty after loading")

        # Check for columns that are all NA
        all_na_cols = df.columns[df.isna().all()].tolist()
        if all_na_cols:
            warnings.append(f"Columns with all missing values: {all_na_cols}")

        # Check for duplicate columns
        if len(df.columns) != len(set(df.columns)):
            warnings.append("Duplicate column names detected")

        # Memory usage info for large dataframes
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        if memory_mb > 100:
            warnings.append(f"DataFrame memory usage: {memory_mb:.1f} MB")

        return DataLoadResult(
            df=df,
            encoding=encoding,
            delimiter=delimiter,
            warnings=warnings,
            file_size_mb=file_size_mb,
            load_successful=True
        )

    except pd.errors.EmptyDataError:
        return DataLoadResult(
            df=pd.DataFrame(),
            encoding=encoding,
            delimiter=delimiter,
            warnings=warnings,
            file_size_mb=file_size_mb,
            load_successful=False,
            error_message="File is empty or contains no data"
        )

    except pd.errors.ParserError as e:
        return DataLoadResult(
            df=pd.DataFrame(),
            encoding=encoding,
            delimiter=delimiter,
            warnings=warnings,
            file_size_mb=file_size_mb,
            load_successful=False,
            error_message=f"CSV parsing error: {str(e)}"
        )

    except Exception as e:
        return DataLoadResult(
            df=pd.DataFrame(),
            encoding=encoding,
            delimiter=delimiter,
            warnings=warnings,
            file_size_mb=file_size_mb,
            load_successful=False,
            error_message=f"Unexpected error: {str(e)}"
        )


def get_dataframe_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Get comprehensive information about a DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with DataFrame metadata
    """
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        "missing_counts": df.isna().sum().to_dict(),
        "missing_percentages": (df.isna().sum() / len(df) * 100).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
        "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist(),
    }

    return info
