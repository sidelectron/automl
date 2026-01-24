"""Base transformer class for preprocessing steps."""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
import numpy as np


class BaseTransformer(ABC):
    """Abstract base class for preprocessing transformers."""

    def __init__(self, columns: Optional[list] = None):
        """Initialize transformer.

        Args:
            columns: List of column names to transform. If None, transforms all applicable columns.
        """
        self.columns = columns
        self.fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BaseTransformer":
        """Fit the transformer on data.

        Args:
            X: Input DataFrame
            y: Optional target Series

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.

        Args:
            X: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        pass

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            X: Input DataFrame
            y: Optional target Series

        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y).transform(X)

    def _get_columns(self, X: pd.DataFrame) -> list:
        """Get columns to transform.

        Args:
            X: Input DataFrame

        Returns:
            List of column names
        """
        if self.columns is None:
            return list(X.columns)
        return [col for col in self.columns if col in X.columns]
