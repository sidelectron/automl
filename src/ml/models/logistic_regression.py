"""Logistic Regression model implementation."""

from typing import Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import BaseModel


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model wrapper."""

    def __init__(self, **hyperparameters):
        """Initialize Logistic Regression model.

        Args:
            **hyperparameters: Logistic Regression hyperparameters
        """
        super().__init__(**hyperparameters)
        self.task_type = hyperparameters.pop("task_type", "binary_classification")

        if self.task_type == "regression":
            raise ValueError("LogisticRegressionModel only supports classification tasks")

        # Set default hyperparameters
        defaults = {
            "max_iter": 1000,
            "random_state": 42,
            "solver": "lbfgs",
        }
        defaults.update(hyperparameters)

        self.model = LogisticRegression(**defaults)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple] = None
    ) -> "LogisticRegressionModel":
        """Train Logistic Regression model.

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
        """Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Probability array
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before predict_proba")

        return self.model.predict_proba(X)
