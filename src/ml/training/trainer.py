"""Model training with cross-validation using sklearn methods."""

from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline

from ..models.base import BaseModel


class ModelTrainer:
    """Trainer for ML models with cross-validation."""

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        cv_folds: int = 5
    ):
        """Initialize trainer.

        Args:
            test_size: Proportion of data for test set
            random_state: Random seed
            cv_folds: Number of cross-validation folds
        """
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds

    def train_with_validation(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[BaseModel, Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
        """Train model with train/validation split.

        Args:
            model: Model instance to train
            X: Features
            y: Target

        Returns:
            Tuple of (trained_model, (X_train, y_train), (X_val, y_val))
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if model.task_type != "regression" else None
        )

        model.train(X_train, y_train, validation_data=(X_val, y_val))

        return model, (X_train, y_train), (X_val, y_val)

    def train_with_cv(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        scoring: Optional[str] = None,
        task_type: str = "binary_classification"
    ) -> Dict[str, Any]:
        """Train model with cross-validation and return detailed results.

        WARNING: This method assumes X is RAW (unprocessed) data, OR that
        preprocessing is handled inside the model. If X is already preprocessed,
        this will cause DATA LEAKAGE because the same preprocessing was applied
        to all folds. Use train_pipeline_with_cv() instead for proper CV with
        preprocessing.

        Args:
            model: Model instance to train
            X: Features (should be raw/unprocessed, or model should include preprocessing)
            y: Target
            scoring: Scoring metric (default: based on task type)
            task_type: Type of ML task

        Returns:
            Dictionary with mean score, std score, and all fold scores
        """
        # Set default scoring based on task type
        if scoring is None:
            if task_type == "regression":
                scoring = "neg_mean_squared_error"
            elif task_type == "binary_classification":
                scoring = "roc_auc"
            else:
                scoring = "accuracy"

        # Choose CV strategy
        if task_type == "regression":
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        # For sklearn-compatible models
        if hasattr(model.model, "fit"):
            scores = cross_val_score(
                model.model,
                X, y,
                cv=cv,
                scoring=scoring,
                n_jobs=-1
            )
            return {
                "mean_score": float(scores.mean()),
                "std_score": float(scores.std()),
                "scores": scores.tolist(),
                "scoring": scoring
            }

        # Fallback: train on full data
        model.train(X, y)
        return {
            "mean_score": 0.0,
            "std_score": 0.0,
            "scores": [],
            "scoring": scoring
        }
    
    def train_pipeline_with_cv(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        scoring: Optional[str] = None,
        task_type: str = "binary_classification"
    ) -> Dict[str, Any]:
        """Train sklearn Pipeline with cross-validation.

        Args:
            pipeline: sklearn Pipeline with preprocessing and model
            X: Features
            y: Target
            scoring: Scoring metric
            task_type: Type of ML task

        Returns:
            Dictionary with CV results
        """
        # Set default scoring
        if scoring is None:
            if task_type == "regression":
                scoring = "neg_mean_squared_error"
            elif task_type == "binary_classification":
                scoring = "roc_auc"
            else:
                scoring = "accuracy"

        # Choose CV strategy
        if task_type == "regression":
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        scores = cross_val_score(
            pipeline,
            X, y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        return {
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std()),
            "scores": scores.tolist(),
            "scoring": scoring
        }
