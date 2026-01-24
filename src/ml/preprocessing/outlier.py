"""Outlier detection and handling transformers.

Implements outlier handling methods from ML text Pages 127-141:
- IQR Method (Interquartile Range)
- Z-Score Method
- Percentile/Winsorization Method

Reference formulas:
- IQR: Lower = Q1 - 1.5*IQR, Upper = Q3 + 1.5*IQR
- Z-Score: |z| > 3 is considered outlier
- Percentile: Cap at 1st and 99th percentile
"""

from typing import Optional, List, Dict, Any, Literal
import pandas as pd
import numpy as np
from scipy import stats

from .base import BaseTransformer


class OutlierDetector:
    """Detect outliers using various methods.

    From ML text Pages 127-141:
    - IQR Method: Robust for skewed distributions
    - Z-Score Method: For normally distributed data
    - Percentile Method: For any distribution
    """

    def __init__(
        self,
        method: Literal["iqr", "zscore", "percentile"] = "iqr",
        threshold: float = 1.5,
        columns: Optional[List[str]] = None
    ):
        """Initialize outlier detector.

        Args:
            method: Detection method ('iqr', 'zscore', 'percentile')
            threshold: Threshold for detection
                - IQR: multiplier (default 1.5, use 3 for extreme outliers)
                - Z-Score: number of standard deviations (default 3)
                - Percentile: percentile value (default 1, meaning 1st and 99th)
            columns: Columns to check (None = all numeric)
        """
        self.method = method
        self.threshold = threshold
        self.columns = columns
        self.bounds_: Dict[str, Dict[str, float]] = {}

    def fit(self, X: pd.DataFrame) -> "OutlierDetector":
        """Calculate bounds for outlier detection.

        Args:
            X: Input DataFrame

        Returns:
            Self
        """
        if self.columns is None:
            cols = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = [c for c in self.columns if c in X.columns]

        for col in cols:
            series = X[col].dropna()
            if len(series) == 0:
                continue

            if self.method == "iqr":
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                self.bounds_[col] = {
                    "lower": q1 - self.threshold * iqr,
                    "upper": q3 + self.threshold * iqr,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr
                }

            elif self.method == "zscore":
                mean = series.mean()
                std = series.std()
                self.bounds_[col] = {
                    "lower": mean - self.threshold * std,
                    "upper": mean + self.threshold * std,
                    "mean": mean,
                    "std": std
                }

            elif self.method == "percentile":
                lower_pct = self.threshold / 100
                upper_pct = 1 - (self.threshold / 100)
                self.bounds_[col] = {
                    "lower": series.quantile(lower_pct),
                    "upper": series.quantile(upper_pct),
                    "lower_percentile": lower_pct * 100,
                    "upper_percentile": upper_pct * 100
                }

        return self

    def detect(self, X: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers and return boolean mask.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with True where outliers exist
        """
        outlier_mask = pd.DataFrame(False, index=X.index, columns=X.columns)

        for col, bounds in self.bounds_.items():
            if col in X.columns:
                lower = bounds["lower"]
                upper = bounds["upper"]
                outlier_mask[col] = (X[col] < lower) | (X[col] > upper)

        return outlier_mask

    def get_outlier_summary(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of outliers per column.

        Args:
            X: Input DataFrame

        Returns:
            Dictionary with outlier counts and percentages
        """
        mask = self.detect(X)
        summary = {}

        for col in mask.columns:
            if col in self.bounds_:
                count = int(mask[col].sum())
                total = len(mask[col].dropna())
                summary[col] = {
                    "count": count,
                    "percentage": round(count / total * 100, 2) if total > 0 else 0,
                    "bounds": self.bounds_[col]
                }

        return summary


class OutlierHandler(BaseTransformer):
    """Handle outliers through capping, removal, or flagging.

    From ML text Pages 131-141:
    - Capping (Winsorization): Replace outliers with bounds
    - Trimming (Removal): Remove outlier rows
    - Flagging: Add indicator columns for outliers
    """

    def __init__(
        self,
        method: Literal["iqr", "zscore", "percentile"] = "iqr",
        action: Literal["cap", "remove", "flag"] = "cap",
        threshold: float = 1.5,
        columns: Optional[List[str]] = None
    ):
        """Initialize outlier handler.

        Args:
            method: Detection method ('iqr', 'zscore', 'percentile')
            action: How to handle outliers ('cap', 'remove', 'flag')
            threshold: Threshold for detection
            columns: Columns to process (None = all numeric)
        """
        super().__init__(columns)
        self.method = method
        self.action = action
        self.threshold = threshold
        self.detector = OutlierDetector(method=method, threshold=threshold, columns=columns)
        self.outlier_columns_created: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "OutlierHandler":
        """Fit outlier detector on data.

        Args:
            X: Input DataFrame
            y: Optional target (not used)

        Returns:
            Self
        """
        self.detector.fit(X)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by handling outliers.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with outliers handled
        """
        if not self.fitted:
            raise ValueError("Transformer must be fitted before transform")

        X = X.copy()

        if self.action == "cap":
            # Winsorization: cap values at bounds
            for col, bounds in self.detector.bounds_.items():
                if col in X.columns:
                    X[col] = X[col].clip(lower=bounds["lower"], upper=bounds["upper"])

        elif self.action == "remove":
            # Remove rows with any outliers
            outlier_mask = self.detector.detect(X)
            rows_with_outliers = outlier_mask.any(axis=1)
            X = X[~rows_with_outliers]

        elif self.action == "flag":
            # Add indicator columns for outliers
            outlier_mask = self.detector.detect(X)
            for col in self.detector.bounds_.keys():
                if col in X.columns:
                    flag_col = f"{col}_is_outlier"
                    X[flag_col] = outlier_mask[col].astype(int)
                    self.outlier_columns_created.append(flag_col)

        return X

    def get_bounds(self) -> Dict[str, Dict[str, float]]:
        """Get the calculated bounds for each column.

        Returns:
            Dictionary with bounds per column
        """
        return self.detector.bounds_


class IQROutlierHandler(OutlierHandler):
    """Convenience class for IQR-based outlier handling.

    From ML text Page 134:
    IQR = Q3 - Q1
    Lower = Q1 - 1.5 * IQR
    Upper = Q3 + 1.5 * IQR
    """

    def __init__(
        self,
        action: Literal["cap", "remove", "flag"] = "cap",
        threshold: float = 1.5,
        columns: Optional[List[str]] = None
    ):
        super().__init__(
            method="iqr",
            action=action,
            threshold=threshold,
            columns=columns
        )


class ZScoreOutlierHandler(OutlierHandler):
    """Convenience class for Z-score based outlier handling.

    From ML text Page 130:
    Z-score = (x - mean) / std
    Outlier if |Z| > threshold (default 3)
    """

    def __init__(
        self,
        action: Literal["cap", "remove", "flag"] = "cap",
        threshold: float = 3.0,
        columns: Optional[List[str]] = None
    ):
        super().__init__(
            method="zscore",
            action=action,
            threshold=threshold,
            columns=columns
        )


class PercentileOutlierHandler(OutlierHandler):
    """Convenience class for percentile-based outlier handling.

    From ML text Page 138:
    Cap values at specified percentiles (e.g., 1st and 99th)
    """

    def __init__(
        self,
        action: Literal["cap", "remove", "flag"] = "cap",
        percentile: float = 1.0,
        columns: Optional[List[str]] = None
    ):
        super().__init__(
            method="percentile",
            action=action,
            threshold=percentile,
            columns=columns
        )
