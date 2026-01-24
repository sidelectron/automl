"""K-Nearest Neighbors model implementation.

From ML text Pages 37, 63, 184: KNN for classification and regression.
Instance-based learning that uses distance metrics.
"""

from typing import Optional, Literal
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from .base import BaseModel


class KNNModel(BaseModel):
    """K-Nearest Neighbors model wrapper for classification and regression.

    From ML text:
    - Instance-based learning (lazy learning)
    - Requires feature scaling for distance-based metrics
    - Computationally expensive for large datasets
    """

    def __init__(self, **hyperparameters):
        """Initialize KNN model.

        Args:
            task_type: 'binary_classification', 'multiclass_classification', or 'regression'
            n_neighbors: Number of neighbors to use (default 5)
            weights: Weight function ('uniform' or 'distance')
            metric: Distance metric ('euclidean', 'manhattan', 'minkowski')
            **hyperparameters: Additional model hyperparameters
        """
        super().__init__(**hyperparameters)
        self.task_type = hyperparameters.pop("task_type", "binary_classification")

        # Set default hyperparameters
        defaults = {
            "n_neighbors": 5,
            "weights": "distance",  # Closer neighbors have more influence
            "metric": "minkowski",
            "n_jobs": -1,  # Use all cores
        }
        defaults.update(hyperparameters)

        if self.task_type == "regression":
            self.model = KNeighborsRegressor(**defaults)
        else:
            self.model = KNeighborsClassifier(**defaults)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple] = None
    ) -> "KNNModel":
        """Train KNN model.

        Note: KNN is lazy - it just stores the training data.
        Scaling is recommended before training.

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

    def get_neighbors(self, X: pd.DataFrame, n_neighbors: Optional[int] = None) -> tuple:
        """Get the neighbors for given samples.

        Args:
            X: Input features
            n_neighbors: Number of neighbors (default: model's n_neighbors)

        Returns:
            Tuple of (distances, indices)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        if n_neighbors is None:
            n_neighbors = self.model.n_neighbors

        return self.model.kneighbors(X, n_neighbors=n_neighbors)
