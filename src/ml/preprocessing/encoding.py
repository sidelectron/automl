"""Categorical encoding transformers."""

from typing import Optional, Dict
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder

from .base import BaseTransformer


class OneHotEncoder(BaseTransformer):
    """One-hot encode categorical variables."""

    def __init__(self, columns: Optional[list] = None, drop: str = "first"):
        """Initialize one-hot encoder.

        Args:
            columns: Columns to encode. If None, encodes all categorical columns.
            drop: Strategy for dropping categories ('first', 'if_binary', or None)
        """
        super().__init__(columns)
        self.drop = drop
        self.encoder = None
        self.feature_names = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "OneHotEncoder":
        """Fit encoder on data.

        Args:
            X: Input DataFrame
            y: Optional target (not used)

        Returns:
            Self
        """
        cols = self._get_columns(X)
        categorical_cols = [col for col in cols if not pd.api.types.is_numeric_dtype(X[col])]

        if categorical_cols:
            self.encoder = SklearnOneHotEncoder(drop=self.drop, sparse_output=False)
            self.encoder.fit(X[categorical_cols])
            # Store feature names for transform
            self.feature_names = self.encoder.get_feature_names_out(categorical_cols)

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by one-hot encoding.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with encoded columns
        """
        if not self.fitted:
            raise ValueError("Transformer must be fitted before transform")

        if self.encoder is None:
            return X.copy()

        X = X.copy()
        cols = self._get_columns(X)
        categorical_cols = [col for col in cols if not pd.api.types.is_numeric_dtype(X[col])]

        if categorical_cols:
            # Encode categorical columns
            encoded = self.encoder.transform(X[categorical_cols])
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.feature_names,
                index=X.index
            )

            # Drop original categorical columns and add encoded ones
            X = X.drop(columns=categorical_cols)
            X = pd.concat([X, encoded_df], axis=1)

        return X


class LabelEncoder(BaseTransformer):
    """Label encode categorical variables (for target variable or ordinal categories)."""

    def __init__(self, columns: Optional[list] = None):
        """Initialize label encoder.

        Args:
            columns: Columns to encode. If None, encodes all categorical columns.
        """
        super().__init__(columns)
        self.encoders = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "LabelEncoder":
        """Fit encoder on data.

        Args:
            X: Input DataFrame
            y: Optional target (not used)

        Returns:
            Self
        """
        cols = self._get_columns(X)
        categorical_cols = [col for col in cols if not pd.api.types.is_numeric_dtype(X[col])]

        for col in categorical_cols:
            encoder = SklearnLabelEncoder()
            encoder.fit(X[col].astype(str))
            self.encoders[col] = encoder

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by label encoding.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with encoded columns
        """
        if not self.fitted:
            raise ValueError("Transformer must be fitted before transform")

        X = X.copy()
        for col, encoder in self.encoders.items():
            if col in X.columns:
                X[col] = encoder.transform(X[col].astype(str))

        return X


class TargetEncoder(BaseTransformer):
    """Target (mean) encoding for high-cardinality categorical variables.

    Replaces each category with the mean of the target variable for that category.
    Uses smoothing to handle rare categories and prevent overfitting.

    From ML text: Target encoding is useful for high-cardinality categoricals
    where one-hot encoding would create too many features.
    """

    def __init__(
        self,
        columns: Optional[list] = None,
        smoothing: float = 1.0,
        min_samples_leaf: int = 1
    ):
        """Initialize target encoder.

        Args:
            columns: Columns to encode. If None, encodes all categorical columns.
            smoothing: Smoothing factor for regularization. Higher values give more
                      weight to the global mean (prevents overfitting on rare categories).
            min_samples_leaf: Minimum samples required to use category mean.
                             Categories with fewer samples use global mean.
        """
        super().__init__(columns)
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.encoding_maps: Dict[str, Dict] = {}
        self.global_mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TargetEncoder":
        """Fit encoder on data.

        Args:
            X: Input DataFrame
            y: Target variable (required for target encoding)

        Returns:
            Self

        Raises:
            ValueError: If y is not provided
        """
        if y is None:
            raise ValueError("Target variable y is required for TargetEncoder")

        cols = self._get_columns(X)
        categorical_cols = [col for col in cols if not pd.api.types.is_numeric_dtype(X[col])]

        # Calculate global mean
        self.global_mean = float(y.mean())

        for col in categorical_cols:
            # Calculate category statistics
            df_temp = pd.DataFrame({col: X[col], 'target': y})
            agg = df_temp.groupby(col)['target'].agg(['mean', 'count'])

            # Apply smoothing: smoothed_mean = (count * mean + smoothing * global_mean) / (count + smoothing)
            # This gives more weight to global mean for rare categories
            smoothed_means = {}
            for category in agg.index:
                count = agg.loc[category, 'count']
                cat_mean = agg.loc[category, 'mean']

                if count >= self.min_samples_leaf:
                    # Smoothed mean calculation
                    smoothed_mean = (count * cat_mean + self.smoothing * self.global_mean) / (count + self.smoothing)
                else:
                    # Use global mean for rare categories
                    smoothed_mean = self.global_mean

                smoothed_means[category] = smoothed_mean

            self.encoding_maps[col] = smoothed_means

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by target encoding.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with encoded columns
        """
        if not self.fitted:
            raise ValueError("Transformer must be fitted before transform")

        X = X.copy()
        for col, encoding_map in self.encoding_maps.items():
            if col in X.columns:
                # Map known categories, use global mean for unknown
                X[col] = X[col].map(encoding_map).fillna(self.global_mean)

        return X

    def get_encoding_map(self, column: str) -> Dict:
        """Get the encoding map for a specific column.

        Args:
            column: Column name

        Returns:
            Dictionary mapping categories to encoded values
        """
        return self.encoding_maps.get(column, {})
