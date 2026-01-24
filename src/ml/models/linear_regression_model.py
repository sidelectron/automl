"""Linear Regression model implementation.

From ML text Page 13: Linear Regression for regression tasks.
Used as a baseline model for regression problems.
"""

from typing import Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from .base import BaseModel


class LinearRegressionModel(BaseModel):
    """Linear Regression model wrapper.

    Supports standard Linear Regression and regularized variants:
    - Linear Regression (OLS)
    - Ridge (L2 regularization)
    - Lasso (L1 regularization)
    - ElasticNet (L1 + L2 regularization)
    """

    def __init__(self, **hyperparameters):
        """Initialize Linear Regression model.

        Args:
            regularization: Type of regularization ('none', 'ridge', 'lasso', 'elasticnet')
            alpha: Regularization strength (for ridge/lasso/elasticnet)
            l1_ratio: L1/L2 ratio for elasticnet (0-1)
            **hyperparameters: Additional model hyperparameters
        """
        super().__init__(**hyperparameters)
        self.task_type = hyperparameters.pop("task_type", "regression")

        if self.task_type != "regression":
            raise ValueError("LinearRegressionModel only supports regression tasks")

        regularization = hyperparameters.pop("regularization", "none")
        alpha = hyperparameters.pop("alpha", 1.0)
        l1_ratio = hyperparameters.pop("l1_ratio", 0.5)

        if regularization == "ridge":
            self.model = Ridge(alpha=alpha, **hyperparameters)
        elif regularization == "lasso":
            self.model = Lasso(alpha=alpha, **hyperparameters)
        elif regularization == "elasticnet":
            self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **hyperparameters)
        else:
            self.model = LinearRegression(**hyperparameters)

        self.regularization = regularization

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple] = None
    ) -> "LinearRegressionModel":
        """Train Linear Regression model.

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
        """Not applicable for regression.

        Raises:
            NotImplementedError: Regression models don't have predict_proba
        """
        raise NotImplementedError("predict_proba not available for regression models")

    def get_coefficients(self) -> dict:
        """Get model coefficients.

        Returns:
            Dictionary with intercept and coefficients
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        return {
            "intercept": float(self.model.intercept_),
            "coefficients": self.model.coef_.tolist()
        }
