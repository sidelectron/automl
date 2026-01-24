"""Model factory for creating model instances."""

from typing import Dict, Any, Optional

from .base import BaseModel
from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel
from .random_forest_model import RandomForestModel
from .logistic_regression import LogisticRegressionModel
from .linear_regression_model import LinearRegressionModel
from .decision_tree import DecisionTreeModel
from .knn_model import KNNModel
from .svm_model import SVMModel


class ModelFactory:
    """Factory for creating model instances.

    Supports models for classification and regression:
    - Ensemble: XGBoost, LightGBM, Random Forest
    - Linear: Logistic Regression, Linear Regression (with Ridge/Lasso/ElasticNet)
    - Tree-based: Decision Tree
    - Instance-based: KNN
    - Kernel-based: SVM
    """

    # Mapping of model names to classes
    MODEL_MAP = {
        # Ensemble models
        "xgboost": XGBoostModel,
        "lightgbm": LightGBMModel,
        "random_forest": RandomForestModel,
        # Linear models
        "logistic_regression": LogisticRegressionModel,
        "linear_regression": LinearRegressionModel,
        # Tree-based
        "decision_tree": DecisionTreeModel,
        # Instance-based
        "knn": KNNModel,
        # Kernel-based
        "svm": SVMModel,
    }

    @classmethod
    def create(
        cls,
        model_name: str,
        task_type: str = "binary_classification",
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> BaseModel:
        """Create a model instance.

        Args:
            model_name: Name of the model (e.g., "xgboost", "lightgbm")
            task_type: Type of task ("binary_classification", "multiclass_classification", "regression")
            hyperparameters: Optional model-specific hyperparameters

        Returns:
            Model instance

        Raises:
            ValueError: If model_name is not recognized
        """
        model_name_lower = model_name.lower().replace("-", "_")

        if model_name_lower not in cls.MODEL_MAP:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {list(cls.MODEL_MAP.keys())}"
            )

        model_class = cls.MODEL_MAP[model_name_lower]

        kwargs = {"task_type": task_type}
        if hyperparameters:
            kwargs.update(hyperparameters)

        return model_class(**kwargs)

    @classmethod
    def list_models(cls) -> list:
        """List available model names.

        Returns:
            List of model names
        """
        return list(cls.MODEL_MAP.keys())
