"""Base model class."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np


class BaseModel(ABC):
    """Abstract base class for ML models."""

    def __init__(self, **hyperparameters):
        """Initialize model.

        Args:
            **hyperparameters: Model-specific hyperparameters
        """
        self.hyperparameters = hyperparameters
        self.model = None
        self.fitted = False

    @abstractmethod
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple] = None
    ) -> "BaseModel":
        """Train the model.

        Args:
            X: Training features
            y: Training target
            validation_data: Optional (X_val, y_val) tuple

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.

        Args:
            X: Input features

        Returns:
            Predicted values
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities (for classification).

        Args:
            X: Input features

        Returns:
            Probability array
        """
        pass

    def save(self, filepath: str):
        """Save model to disk.

        Args:
            filepath: Path to save model
        """
        import pickle
        from pathlib import Path

        if not self.fitted:
            raise ValueError("Model must be fitted before saving")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "BaseModel":
        """Load model from disk.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded model
        """
        import pickle

        with open(filepath, "rb") as f:
            return pickle.load(f)
