"""Hyperparameter tuning using GridSearchCV and RandomizedSearchCV."""

from typing import Dict, Any, Optional, List
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from ..models.base import BaseModel


class HyperparameterTuner:
    """Tune hyperparameters using GridSearchCV or RandomizedSearchCV."""

    def __init__(
        self,
        method: str = "grid",
        cv: int = 5,
        n_iter: int = 10,
        scoring: Optional[str] = None,
        n_jobs: int = -1,
        verbose: int = 1
    ):
        """Initialize hyperparameter tuner.

        Args:
            method: "grid" for GridSearchCV or "random" for RandomizedSearchCV
            cv: Number of cross-validation folds
            n_iter: Number of iterations for RandomizedSearchCV
            scoring: Scoring metric (default: based on task type)
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
        """
        self.method = method
        self.cv = cv
        self.n_iter = n_iter
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose

    def tune(
        self,
        pipeline: Pipeline,
        param_grid: Dict[str, List[Any]],
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str = "binary_classification"
    ) -> Dict[str, Any]:
        """Tune hyperparameters for a pipeline.

        Args:
            pipeline: sklearn Pipeline with preprocessing and model
            param_grid: Parameter grid for tuning
            X: Features
            y: Target
            task_type: Type of ML task

        Returns:
            Dictionary with best parameters, best score, and best estimator
        """
        # Set default scoring based on task type
        if self.scoring is None:
            if task_type == "regression":
                self.scoring = "neg_mean_squared_error"
            elif task_type == "binary_classification":
                self.scoring = "roc_auc"
            else:
                self.scoring = "accuracy"

        if self.method == "grid":
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )
        else:  # random
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_grid,
                n_iter=self.n_iter,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                random_state=42
            )

        search.fit(X, y)

        return {
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),
            "best_estimator": search.best_estimator_,
            "cv_results": search.cv_results_
        }

    @staticmethod
    def get_default_param_grids(model_name: str, task_type: str) -> Dict[str, List[Any]]:
        """Get default parameter grids for common models.

        Args:
            model_name: Name of the model
            task_type: Type of ML task

        Returns:
            Parameter grid dictionary
        """
        grids = {}

        if model_name.lower() == "random_forest":
            grids = {
                "model__n_estimators": [100, 200, 500],
                "model__max_depth": [10, 20, None],
                "model__min_samples_split": [2, 5, 10],
                "model__max_features": ["sqrt", "log2", None]
            }
        elif model_name.lower() == "xgboost":
            grids = {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.1, 0.3],
                "model__subsample": [0.8, 1.0]
            }
        elif model_name.lower() == "lightgbm":
            grids = {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.1, 0.3],
                "model__num_leaves": [31, 50, 100]
            }
        elif model_name.lower() == "logistic_regression":
            grids = {
                "model__C": [0.1, 1.0, 10.0],
                "model__penalty": ["l1", "l2"],
                "model__solver": ["liblinear", "saga"]
            }

        return grids
