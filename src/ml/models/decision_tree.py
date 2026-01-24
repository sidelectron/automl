"""Decision Tree model implementation.

From ML text Page 79, 238: Decision Trees for classification and regression.
Provides interpretable rules and feature importance.
"""

from typing import Optional
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .base import BaseModel


class DecisionTreeModel(BaseModel):
    """Decision Tree model wrapper for classification and regression."""

    def __init__(self, **hyperparameters):
        """Initialize Decision Tree model.

        Args:
            task_type: 'binary_classification', 'multiclass_classification', or 'regression'
            max_depth: Maximum depth of tree
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples at a leaf node
            criterion: Split criterion ('gini', 'entropy' for classification; 'mse', 'mae' for regression)
            **hyperparameters: Additional model hyperparameters
        """
        super().__init__(**hyperparameters)
        self.task_type = hyperparameters.pop("task_type", "binary_classification")

        # Set default hyperparameters
        defaults = {
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
        }
        defaults.update(hyperparameters)

        if self.task_type == "regression":
            defaults.setdefault("criterion", "squared_error")
            self.model = DecisionTreeRegressor(**defaults)
        else:
            defaults.setdefault("criterion", "gini")
            self.model = DecisionTreeClassifier(**defaults)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple] = None
    ) -> "DecisionTreeModel":
        """Train Decision Tree model.

        Args:
            X: Training features
            y: Training target
            validation_data: Optional (X_val, y_val) tuple (not used)

        Returns:
            Self
        """
        self.model.fit(X, y)
        self.fitted = True
        self.feature_names_ = X.columns.tolist() if hasattr(X, 'columns') else None
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

    def get_feature_importance(self) -> dict:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        importances = self.model.feature_importances_

        if self.feature_names_:
            return dict(zip(self.feature_names_, importances.tolist()))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(importances)}

    def get_tree_depth(self) -> int:
        """Get the depth of the tree.

        Returns:
            Tree depth
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        return self.model.get_depth()
