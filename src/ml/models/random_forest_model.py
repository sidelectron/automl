"""Random Forest model implementation."""

from typing import Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .base import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest model wrapper."""

    def __init__(self, **hyperparameters):
        """Initialize Random Forest model.

        Args:
            **hyperparameters: Random Forest hyperparameters
        """
        super().__init__(**hyperparameters)
        self.task_type = hyperparameters.pop("task_type", "binary_classification")

        # Set default hyperparameters
        defaults = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
        }
        defaults.update(hyperparameters)

        if self.task_type == "regression":
            self.model = RandomForestRegressor(**defaults)
        else:
            self.model = RandomForestClassifier(**defaults)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple] = None
    ) -> "RandomForestModel":
        """Train Random Forest model.

        Args:
            X: Training features
            y: Training target
            validation_data: Optional (X_val, y_val) tuple (not used for RF)

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

        if self.task_type == "regression":
            raise ValueError("predict_proba not available for regression")

        return self.model.predict_proba(X)
