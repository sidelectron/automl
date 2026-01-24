"""XGBoost model implementation."""

from typing import Optional
import pandas as pd
import numpy as np

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from .base import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost model wrapper."""

    def __init__(self, **hyperparameters):
        """Initialize XGBoost model.

        Args:
            **hyperparameters: XGBoost hyperparameters
        """
        if xgb is None:
            raise ImportError("xgboost package not installed. Install with: pip install xgboost")

        super().__init__(**hyperparameters)
        self.task_type = hyperparameters.pop("task_type", "binary_classification")

        # Set default hyperparameters
        defaults = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        defaults.update(hyperparameters)

        if self.task_type == "binary_classification":
            self.model = xgb.XGBClassifier(**defaults)
        elif self.task_type == "multiclass_classification":
            self.model = xgb.XGBClassifier(**defaults)
        else:  # regression
            self.model = xgb.XGBRegressor(**defaults)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple] = None
    ) -> "XGBoostModel":
        """Train XGBoost model.

        Args:
            X: Training features
            y: Training target
            validation_data: Optional (X_val, y_val) tuple

        Returns:
            Self
        """
        if validation_data:
            X_val, y_val = validation_data
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
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
