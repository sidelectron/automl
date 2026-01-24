"""Feature scaling transformers."""

from typing import Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from sklearn.preprocessing import RobustScaler as SklearnRobustScaler

from .base import BaseTransformer


class StandardScaler(BaseTransformer):
    """Standardize features by removing mean and scaling to unit variance."""

    def __init__(self, columns: Optional[list] = None):
        """Initialize standard scaler.

        Args:
            columns: Columns to scale. If None, scales all numeric columns.
        """
        super().__init__(columns)
        self.scaler = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "StandardScaler":
        """Fit scaler on data.

        Args:
            X: Input DataFrame
            y: Optional target (not used)

        Returns:
            Self
        """
        cols = self._get_columns(X)
        numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(X[col])]

        if numeric_cols:
            self.scaler = SklearnStandardScaler()
            self.scaler.fit(X[numeric_cols])

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by scaling.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with scaled values
        """
        if not self.fitted:
            raise ValueError("Transformer must be fitted before transform")

        if self.scaler is None:
            return X.copy()

        X = X.copy()
        cols = self._get_columns(X)
        numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(X[col])]

        if numeric_cols:
            X[numeric_cols] = self.scaler.transform(X[numeric_cols])
            X[numeric_cols] = pd.DataFrame(
                X[numeric_cols],
                columns=numeric_cols,
                index=X.index
            )

        return X


class MinMaxScaler(BaseTransformer):
    """Scale features to a given range (default: 0-1)."""

    def __init__(self, columns: Optional[list] = None, feature_range=(0, 1)):
        """Initialize min-max scaler.

        Args:
            columns: Columns to scale. If None, scales all numeric columns.
            feature_range: Desired range of transformed data
        """
        super().__init__(columns)
        self.feature_range = feature_range
        self.scaler = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "MinMaxScaler":
        """Fit scaler on data.

        Args:
            X: Input DataFrame
            y: Optional target (not used)

        Returns:
            Self
        """
        cols = self._get_columns(X)
        numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(X[col])]

        if numeric_cols:
            self.scaler = SklearnMinMaxScaler(feature_range=self.feature_range)
            self.scaler.fit(X[numeric_cols])

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by scaling.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with scaled values
        """
        if not self.fitted:
            raise ValueError("Transformer must be fitted before transform")

        if self.scaler is None:
            return X.copy()

        X = X.copy()
        cols = self._get_columns(X)
        numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(X[col])]

        if numeric_cols:
            X[numeric_cols] = self.scaler.transform(X[numeric_cols])
            X[numeric_cols] = pd.DataFrame(
                X[numeric_cols],
                columns=numeric_cols,
                index=X.index
            )

        return X


class RobustScaler(BaseTransformer):
    """Scale features using statistics that are robust to outliers."""

    def __init__(self, columns: Optional[list] = None):
        """Initialize robust scaler.

        Args:
            columns: Columns to scale. If None, scales all numeric columns.
        """
        super().__init__(columns)
        self.scaler = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "RobustScaler":
        """Fit scaler on data.

        Args:
            X: Input DataFrame
            y: Optional target (not used)

        Returns:
            Self
        """
        cols = self._get_columns(X)
        numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(X[col])]

        if numeric_cols:
            self.scaler = SklearnRobustScaler()
            self.scaler.fit(X[numeric_cols])

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by scaling.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with scaled values
        """
        if not self.fitted:
            raise ValueError("Transformer must be fitted before transform")

        if self.scaler is None:
            return X.copy()

        X = X.copy()
        cols = self._get_columns(X)
        numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(X[col])]

        if numeric_cols:
            X[numeric_cols] = self.scaler.transform(X[numeric_cols])
            X[numeric_cols] = pd.DataFrame(
                X[numeric_cols],
                columns=numeric_cols,
                index=X.index
            )

        return X
