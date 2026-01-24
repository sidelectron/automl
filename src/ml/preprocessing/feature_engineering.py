"""Feature engineering transformers.

From ML text Pages 57-58, 99-102, 142-145:
- Interaction features
- Polynomial features
- Log transformations
- Date/time feature extraction
"""

from typing import Optional, List
import pandas as pd
import numpy as np

from .base import BaseTransformer


class FeatureEngineer(BaseTransformer):
    """Create new features through transformations.

    Supported methods:
    - interaction: Create product features (x1 * x2)
    - polynomial: Create squared features (x^2)
    - polynomial_full: Full polynomial features using sklearn
    - log_transform: Log transformation (log1p)
    - datetime: Extract date/time components
    - binning: Bin numeric features into categories
    """

    def __init__(
        self,
        columns: Optional[list] = None,
        method: str = "interaction",
        **kwargs
    ):
        """Initialize feature engineer.

        Args:
            columns: Columns to use for feature engineering
            method: Method to use ('interaction', 'polynomial', 'polynomial_full',
                   'log_transform', 'datetime', 'binning')
            **kwargs: Additional parameters for specific methods
                - degree: Polynomial degree (default 2)
                - n_bins: Number of bins for binning (default 5)
        """
        super().__init__(columns)
        self.method = method
        self.created_features = []
        self.degree = kwargs.get("degree", 2)
        self.n_bins = kwargs.get("n_bins", 5)
        self.poly_transformer = None
        self.bin_edges_ = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        """Fit engineer on data.

        Args:
            X: Input DataFrame
            y: Optional target (not used)

        Returns:
            Self
        """
        # For feature engineering, fit is mostly a no-op
        # Actual feature creation happens in transform
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by creating new features.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with new features added
        """
        if not self.fitted:
            raise ValueError("Transformer must be fitted before transform")

        X = X.copy()
        cols = self._get_columns(X)
        numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(X[col])]

        if self.method == "interaction" and len(numeric_cols) >= 2:
            # Create interaction features (product of pairs)
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    feature_name = f"{col1}_x_{col2}"
                    X[feature_name] = X[col1] * X[col2]
                    self.created_features.append(feature_name)

        elif self.method == "polynomial" and numeric_cols:
            # Create polynomial features (squares)
            for col in numeric_cols[:5]:  # Limit to avoid explosion
                feature_name = f"{col}_squared"
                X[feature_name] = X[col] ** 2
                self.created_features.append(feature_name)

        elif self.method == "log_transform" and numeric_cols:
            # Create log-transformed features
            for col in numeric_cols:
                if (X[col] > 0).all():  # Only if all values positive
                    feature_name = f"{col}_log"
                    X[feature_name] = np.log1p(X[col])
                    self.created_features.append(feature_name)

        elif self.method == "polynomial_full" and numeric_cols:
            # Full polynomial features using sklearn (from ML text)
            from sklearn.preprocessing import PolynomialFeatures

            if self.poly_transformer is None:
                self.poly_transformer = PolynomialFeatures(
                    degree=self.degree,
                    include_bias=False,
                    interaction_only=False
                )
                self.poly_transformer.fit(X[numeric_cols[:10]])  # Limit columns

            poly_features = self.poly_transformer.transform(X[numeric_cols[:10]])
            poly_names = self.poly_transformer.get_feature_names_out(numeric_cols[:10])

            # Add new polynomial features (skip original features)
            for i, name in enumerate(poly_names):
                if name not in numeric_cols:
                    X[name] = poly_features[:, i]
                    self.created_features.append(name)

        elif self.method == "datetime":
            # Extract date/time features (from ML text Pages 99-102)
            datetime_cols = [col for col in cols
                           if pd.api.types.is_datetime64_any_dtype(X[col])]

            for col in datetime_cols:
                dt = X[col]

                # Year, month, day
                X[f"{col}_year"] = dt.dt.year
                X[f"{col}_month"] = dt.dt.month
                X[f"{col}_day"] = dt.dt.day
                X[f"{col}_dayofweek"] = dt.dt.dayofweek
                X[f"{col}_dayofyear"] = dt.dt.dayofyear

                # Quarter and week
                X[f"{col}_quarter"] = dt.dt.quarter
                X[f"{col}_weekofyear"] = dt.dt.isocalendar().week.astype(int)

                # Is weekend
                X[f"{col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

                # Hour if available
                if hasattr(dt.dt, 'hour'):
                    X[f"{col}_hour"] = dt.dt.hour
                    X[f"{col}_minute"] = dt.dt.minute

                self.created_features.extend([
                    f"{col}_year", f"{col}_month", f"{col}_day",
                    f"{col}_dayofweek", f"{col}_dayofyear",
                    f"{col}_quarter", f"{col}_weekofyear", f"{col}_is_weekend"
                ])

        elif self.method == "binning" and numeric_cols:
            # Bin numeric features into categories (discretization)
            for col in numeric_cols:
                if col not in self.bin_edges_:
                    # Calculate bin edges during first transform
                    self.bin_edges_[col] = np.percentile(
                        X[col].dropna(),
                        np.linspace(0, 100, self.n_bins + 1)
                    )

                feature_name = f"{col}_binned"
                X[feature_name] = pd.cut(
                    X[col],
                    bins=self.bin_edges_[col],
                    labels=range(self.n_bins),
                    include_lowest=True
                ).astype(float)
                self.created_features.append(feature_name)

        return X
