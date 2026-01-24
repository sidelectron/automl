"""Dimensionality reduction transformers.

From ML text Pages 146-151: PCA for dimensionality reduction.
Reduces features while preserving explained variance.
"""

from typing import Optional, List, Union
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from .base import BaseTransformer


class PCATransformer(BaseTransformer):
    """PCA (Principal Component Analysis) for dimensionality reduction.

    From ML text Pages 146-151:
    - Reduces dimensions while preserving maximum variance
    - Useful for high-dimensional data
    - Helps visualize data in 2D/3D
    - Can speed up training for some algorithms

    Usage:
        # Keep components explaining 95% of variance
        pca = PCATransformer(variance_threshold=0.95)

        # Or specify exact number of components
        pca = PCATransformer(n_components=10)

        # Or reduce to 2D for visualization
        pca = PCATransformer(n_components=2)
    """

    def __init__(
        self,
        n_components: Optional[Union[int, float]] = None,
        variance_threshold: float = 0.95,
        columns: Optional[List[str]] = None
    ):
        """Initialize PCA transformer.

        Args:
            n_components: Number of components to keep.
                - If int: exact number of components
                - If float (0-1): fraction of variance to retain
                - If None: use variance_threshold
            variance_threshold: Minimum variance to retain (default 0.95)
                Only used if n_components is None
            columns: Columns to transform (None = all numeric)
        """
        super().__init__(columns)
        self.variance_threshold = variance_threshold

        if n_components is not None:
            self.n_components = n_components
        else:
            self.n_components = variance_threshold

        self.pca: Optional[PCA] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.cumulative_variance_: Optional[np.ndarray] = None
        self.n_components_selected_: int = 0
        self.original_columns_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "PCATransformer":
        """Fit PCA on data.

        Args:
            X: Input DataFrame
            y: Optional target (not used)

        Returns:
            Self
        """
        cols = self._get_columns(X)
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(X[c])]

        if not numeric_cols:
            raise ValueError("No numeric columns found for PCA")

        self.original_columns_ = numeric_cols
        X_numeric = X[numeric_cols].values

        # Handle missing values for PCA
        if np.isnan(X_numeric).any():
            # Fill NaN with column means for PCA fitting
            col_means = np.nanmean(X_numeric, axis=0)
            X_numeric = np.where(np.isnan(X_numeric), col_means, X_numeric)

        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_numeric)

        self.explained_variance_ = self.pca.explained_variance_
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.cumulative_variance_ = np.cumsum(self.explained_variance_ratio_)
        self.n_components_selected_ = self.pca.n_components_

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted PCA.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with PCA components
        """
        if not self.fitted:
            raise ValueError("Transformer must be fitted before transform")

        X_numeric = X[self.original_columns_].values

        # Handle missing values
        if np.isnan(X_numeric).any():
            col_means = np.nanmean(X_numeric, axis=0)
            X_numeric = np.where(np.isnan(X_numeric), col_means, X_numeric)

        X_pca = self.pca.transform(X_numeric)

        # Create DataFrame with PCA component columns
        pca_columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        X_pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)

        # Keep non-numeric columns unchanged
        non_numeric_cols = [c for c in X.columns if c not in self.original_columns_]
        if non_numeric_cols:
            X_other = X[non_numeric_cols].copy()
            return pd.concat([X_pca_df, X_other], axis=1)

        return X_pca_df

    def get_explained_variance_report(self) -> dict:
        """Get detailed report on explained variance.

        Returns:
            Dictionary with variance information
        """
        if not self.fitted:
            raise ValueError("Transformer must be fitted first")

        return {
            "n_components": self.n_components_selected_,
            "total_variance_explained": float(self.cumulative_variance_[-1]),
            "variance_per_component": self.explained_variance_ratio_.tolist(),
            "cumulative_variance": self.cumulative_variance_.tolist(),
            "original_features": len(self.original_columns_),
            "reduction_ratio": len(self.original_columns_) / self.n_components_selected_
        }

    def get_component_loadings(self) -> pd.DataFrame:
        """Get the loadings (coefficients) for each component.

        Returns:
            DataFrame with loadings (features x components)
        """
        if not self.fitted:
            raise ValueError("Transformer must be fitted first")

        loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f"PC{i+1}" for i in range(self.n_components_selected_)],
            index=self.original_columns_
        )
        return loadings

    def get_top_features_per_component(self, n_top: int = 5) -> dict:
        """Get top contributing features for each component.

        Args:
            n_top: Number of top features to return

        Returns:
            Dictionary mapping component name to top features
        """
        if not self.fitted:
            raise ValueError("Transformer must be fitted first")

        loadings = self.get_component_loadings()
        result = {}

        for col in loadings.columns:
            abs_loadings = loadings[col].abs().sort_values(ascending=False)
            top_features = abs_loadings.head(n_top)
            result[col] = {
                feat: float(loadings.loc[feat, col])
                for feat in top_features.index
            }

        return result
