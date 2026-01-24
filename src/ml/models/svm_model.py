"""Support Vector Machine model implementation.

From ML text Pages 37, 184, 298, 1261: SVM for classification and regression.
Uses kernel trick for non-linear decision boundaries.
"""

from typing import Optional, Literal
import pandas as pd
import numpy as np
from sklearn.svm import SVC, SVR

from .base import BaseModel


class SVMModel(BaseModel):
    """Support Vector Machine model wrapper for classification and regression.

    From ML text:
    - Effective in high-dimensional spaces
    - Memory efficient (uses support vectors)
    - Requires feature scaling
    - Various kernels available (linear, rbf, poly)
    """

    def __init__(self, **hyperparameters):
        """Initialize SVM model.

        Args:
            task_type: 'binary_classification', 'multiclass_classification', or 'regression'
            kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient ('scale', 'auto', or float)
            **hyperparameters: Additional model hyperparameters
        """
        super().__init__(**hyperparameters)
        self.task_type = hyperparameters.pop("task_type", "binary_classification")

        # Set default hyperparameters
        defaults = {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "random_state": 42,
        }
        defaults.update(hyperparameters)

        if self.task_type == "regression":
            self.model = SVR(**defaults)
        else:
            # For classification, enable probability estimates
            defaults["probability"] = True
            self.model = SVC(**defaults)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple] = None
    ) -> "SVMModel":
        """Train SVM model.

        Note: Scaling is highly recommended before training SVM.

        Args:
            X: Training features
            y: Training target
            validation_data: Optional (X_val, y_val) tuple (not used)

        Returns:
            Self
        """
        self.model.fit(X, y)
        self.fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.

        Args:
            X: Input features

        Returns:
            Predicted values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before predict")

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities (for classification).

        Note: Only available for SVC with probability=True.

        Args:
            X: Input features

        Returns:
            Probability array
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before predict_proba")

        if self.task_type == "regression":
            raise NotImplementedError("predict_proba not available for regression")

        return self.model.predict_proba(X)

    def get_support_vectors_count(self) -> int:
        """Get the number of support vectors.

        Returns:
            Number of support vectors
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        return len(self.model.support_vectors_)
