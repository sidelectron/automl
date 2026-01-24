"""Standard ML metrics calculation."""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix as sklearn_confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


class ConfusionMatrix:
    """Confusion matrix wrapper."""

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Initialize confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        self.matrix = sklearn_confusion_matrix(y_true, y_pred)
        if self.matrix.shape == (2, 2):
            self.true_negative = int(self.matrix[0, 0])
            self.false_positive = int(self.matrix[0, 1])
            self.false_negative = int(self.matrix[1, 0])
            self.true_positive = int(self.matrix[1, 1])
        else:
            # Multiclass: use sum of diagonal for TP, rest for others
            self.true_positive = int(np.trace(self.matrix))
            self.true_negative = None
            self.false_positive = None
            self.false_negative = None

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary.

        Returns:
            Dictionary with TP, TN, FP, FN
        """
        result = {
            "true_positive": self.true_positive,
        }
        if self.true_negative is not None:
            result.update({
                "true_negative": self.true_negative,
                "false_positive": self.false_positive,
                "false_negative": self.false_negative,
            })
        return result


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = "binary_classification"
) -> Dict[str, Any]:
    """Calculate standard ML metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        task_type: Type of task ("binary_classification", "multiclass_classification", "regression")

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    if task_type == "regression":
        metrics["mse"] = float(mean_squared_error(y_true, y_pred))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
    else:
        # Classification metrics
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

        if task_type == "binary_classification":
            metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
            metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
            metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

            # Confusion matrix
            cm = ConfusionMatrix(y_true, y_pred)
            metrics["confusion_matrix"] = cm.to_dict()
        else:
            # Multiclass: use average
            metrics["precision"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
            metrics["recall"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
            metrics["f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    return metrics
