"""Missing value imputation transformers."""

from typing import Optional, Literal
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer as SklearnKNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer as SklearnIterativeImputer

from .base import BaseTransformer


class MedianImputer(BaseTransformer):
    """Impute missing values with median (for numeric columns)."""

    def __init__(self, columns: Optional[list] = None):
        """Initialize median imputer.

        Args:
            columns: Columns to impute. If None, imputes all numeric columns.
        """
        super().__init__(columns)
        self.imputer = None
        self.median_values = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "MedianImputer":
        """Fit imputer on data.

        Args:
            X: Input DataFrame
            y: Optional target (not used)

        Returns:
            Self
        """
        cols = self._get_columns(X)
        numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(X[col])]

        for col in numeric_cols:
            self.median_values[col] = X[col].median()

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by imputing missing values.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with imputed values
        """
        if not self.fitted:
            raise ValueError("Transformer must be fitted before transform")

        X = X.copy()
        for col, median_val in self.median_values.items():
            if col in X.columns:
                X[col] = X[col].fillna(median_val)

        return X


class ModeImputer(BaseTransformer):
    """Impute missing values with mode (for categorical columns)."""

    def __init__(self, columns: Optional[list] = None):
        """Initialize mode imputer.

        Args:
            columns: Columns to impute. If None, imputes all categorical columns.
        """
        super().__init__(columns)
        self.mode_values = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ModeImputer":
        """Fit imputer on data.

        Args:
            X: Input DataFrame
            y: Optional target (not used)

        Returns:
            Self
        """
        cols = self._get_columns(X)
        categorical_cols = [col for col in cols if not pd.api.types.is_numeric_dtype(X[col])]

        for col in categorical_cols:
            mode_val = X[col].mode()
            self.mode_values[col] = mode_val[0] if len(mode_val) > 0 else None

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by imputing missing values.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with imputed values
        """
        if not self.fitted:
            raise ValueError("Transformer must be fitted before transform")

        X = X.copy()
        for col, mode_val in self.mode_values.items():
            if col in X.columns and mode_val is not None:
                X[col] = X[col].fillna(mode_val)

        return X


class KNNImputer(BaseTransformer):
    """Impute missing values using K-Nearest Neighbors."""

    def __init__(self, columns: Optional[list] = None, n_neighbors: int = 5):
        """Initialize KNN imputer.

        Args:
            columns: Columns to impute. If None, imputes all numeric columns.
            n_neighbors: Number of neighbors for KNN
        """
        super().__init__(columns)
        self.n_neighbors = n_neighbors
        self.imputer = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "KNNImputer":
        """Fit imputer on data.

        Args:
            X: Input DataFrame
            y: Optional target (not used)

        Returns:
            Self
        """
        cols = self._get_columns(X)
        numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(X[col])]

        if numeric_cols:
            self.imputer = SklearnKNNImputer(n_neighbors=self.n_neighbors)
            self.imputer.fit(X[numeric_cols])

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by imputing missing values.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with imputed values
        """
        if not self.fitted:
            raise ValueError("Transformer must be fitted before transform")

        if self.imputer is None:
            return X.copy()

        X = X.copy()
        cols = self._get_columns(X)
        numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(X[col])]

        if numeric_cols:
            X[numeric_cols] = self.imputer.transform(X[numeric_cols])
            # Convert back to DataFrame to preserve column names
            X[numeric_cols] = pd.DataFrame(
                X[numeric_cols],
                columns=numeric_cols,
                index=X.index
            )

        return X


class IterativeImputer(BaseTransformer):
    """Impute missing values using iterative multivariate imputation (MICE).

    Models each feature with missing values as a function of other features,
    and uses that estimate for imputation. It does so in an iterated round-robin
    fashion: at each step, a feature column is designated as output y and the
    other feature columns are treated as inputs X. A regressor is fit on (X, y)
    for known y. Then, the regressor is used to predict the missing values of y.

    From ML text: Iterative imputation is more sophisticated than simple
    mean/median imputation and can capture relationships between features.
    """

    def __init__(
        self,
        columns: Optional[list] = None,
        max_iter: int = 10,
        estimator: Literal["bayesian_ridge", "decision_tree", "extra_trees", "knn"] = "bayesian_ridge",
        initial_strategy: str = "mean",
        random_state: int = 42
    ):
        """Initialize iterative imputer.

        Args:
            columns: Columns to impute. If None, imputes all numeric columns.
            max_iter: Maximum number of imputation rounds.
            estimator: Estimator to use for imputation:
                - "bayesian_ridge": BayesianRidge (default, good for linear relationships)
                - "decision_tree": DecisionTreeRegressor (captures non-linear patterns)
                - "extra_trees": ExtraTreesRegressor (ensemble, more robust)
                - "knn": KNeighborsRegressor (local patterns)
            initial_strategy: Strategy for initial imputation before iterating.
                Options: "mean", "median", "most_frequent", "constant"
            random_state: Random seed for reproducibility.
        """
        super().__init__(columns)
        self.max_iter = max_iter
        self.estimator_type = estimator
        self.initial_strategy = initial_strategy
        self.random_state = random_state
        self.imputer = None
        self._numeric_cols = []

    def _get_estimator(self):
        """Get the estimator object based on estimator type."""
        if self.estimator_type == "bayesian_ridge":
            from sklearn.linear_model import BayesianRidge
            return BayesianRidge()
        elif self.estimator_type == "decision_tree":
            from sklearn.tree import DecisionTreeRegressor
            return DecisionTreeRegressor(max_features='sqrt', random_state=self.random_state)
        elif self.estimator_type == "extra_trees":
            from sklearn.ensemble import ExtraTreesRegressor
            return ExtraTreesRegressor(n_estimators=10, random_state=self.random_state, n_jobs=-1)
        elif self.estimator_type == "knn":
            from sklearn.neighbors import KNeighborsRegressor
            return KNeighborsRegressor(n_neighbors=5)
        else:
            # Default to BayesianRidge
            from sklearn.linear_model import BayesianRidge
            return BayesianRidge()

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "IterativeImputer":
        """Fit imputer on data.

        Args:
            X: Input DataFrame
            y: Optional target (not used)

        Returns:
            Self
        """
        cols = self._get_columns(X)
        self._numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(X[col])]

        if self._numeric_cols:
            estimator = self._get_estimator()
            self.imputer = SklearnIterativeImputer(
                estimator=estimator,
                max_iter=self.max_iter,
                initial_strategy=self.initial_strategy,
                random_state=self.random_state
            )
            self.imputer.fit(X[self._numeric_cols])

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by imputing missing values.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with imputed values
        """
        if not self.fitted:
            raise ValueError("Transformer must be fitted before transform")

        if self.imputer is None:
            return X.copy()

        X = X.copy()

        if self._numeric_cols:
            # Get numeric columns that exist in X
            cols_to_impute = [col for col in self._numeric_cols if col in X.columns]

            if cols_to_impute:
                imputed_values = self.imputer.transform(X[cols_to_impute])
                X[cols_to_impute] = pd.DataFrame(
                    imputed_values,
                    columns=cols_to_impute,
                    index=X.index
                )

        return X

    def get_convergence_info(self) -> dict:
        """Get information about imputation convergence.

        Returns:
            Dictionary with convergence statistics
        """
        if self.imputer is None:
            return {"fitted": False}

        return {
            "fitted": True,
            "n_iter": getattr(self.imputer, 'n_iter_', None),
            "estimator_type": self.estimator_type,
            "max_iter": self.max_iter
        }
