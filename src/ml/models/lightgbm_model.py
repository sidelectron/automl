"""LightGBM model implementation."""

import re
from typing import Optional
import pandas as pd
import numpy as np

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from .base import BaseModel

# LightGBM rejects: , [ ] { } " : and non-ASCII (JSON / non-ASCII in feature names)
_LGB_FORBIDDEN = re.compile(r"[,\[\]{}\":]|[^\x00-\x7F]")


def _sanitize_feature_names_for_lgb(X: pd.DataFrame) -> pd.DataFrame:
    """Sanitize column names for LightGBM. Returns a copy of X with safe column names."""
    X = X.copy()
    raw = [str(c) for c in X.columns]
    sanitized = [_LGB_FORBIDDEN.sub("_", n) for n in raw]
    sanitized = [(s.strip("_") or "f") for s in sanitized]
    # Deduplicate: first keeps name, later duplicates get _1, _2, ...
    out = []
    cnt: dict[str, int] = {}
    for s in sanitized:
        if s not in cnt:
            cnt[s] = 0
        else:
            cnt[s] += 1
            s = f"{s}_{cnt[s]}"
        out.append(s)
    X.columns = out
    return X


class LightGBMModel(BaseModel):
    """LightGBM model wrapper."""

    def __init__(self, **hyperparameters):
        """Initialize LightGBM model.

        Args:
            **hyperparameters: LightGBM hyperparameters
        """
        if lgb is None:
            raise ImportError("lightgbm package not installed. Install with: pip install lightgbm")

        super().__init__(**hyperparameters)
        self.task_type = hyperparameters.pop("task_type", "binary_classification")

        # Set default hyperparameters
        defaults = {
            "n_estimators": 100,
            "max_depth": -1,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "random_state": 42,
            "verbose": -1,
        }
        defaults.update(hyperparameters)

        if self.task_type == "binary_classification":
            self.model = lgb.LGBMClassifier(**defaults)
        elif self.task_type == "multiclass_classification":
            self.model = lgb.LGBMClassifier(**defaults)
        else:  # regression
            self.model = lgb.LGBMRegressor(**defaults)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple] = None
    ) -> "LightGBMModel":
        """Train LightGBM model.

        Args:
            X: Training features
            y: Training target
            validation_data: Optional (X_val, y_val) tuple

        Returns:
            Self
        """
        X = _sanitize_feature_names_for_lgb(X)
        if validation_data:
            X_val, y_val = validation_data
            X_val = _sanitize_feature_names_for_lgb(X_val)
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
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

        X = _sanitize_feature_names_for_lgb(X)
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

        X = _sanitize_feature_names_for_lgb(X)
        return self.model.predict_proba(X)
