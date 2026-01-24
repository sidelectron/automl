"""Threshold tuning for binary classification."""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from ..models.base import BaseModel


class ThresholdTuner:
    """Tune decision threshold for binary classification."""

    def __init__(
        self,
        thresholds: Optional[List[float]] = None,
        intent: Optional[Dict[str, Any]] = None
    ):
        """Initialize threshold tuner.

        Args:
            thresholds: List of thresholds to try (default: [0.3, 0.4, 0.5, 0.6, 0.7])
            intent: Optional parsed intent for business context
        """
        self.thresholds = thresholds or [0.3, 0.4, 0.5, 0.6, 0.7]
        self.intent = intent
        # Import here to avoid circular dependency
        if intent:
            from ..evaluation.business_translator import BusinessMetricCalculator
            self.business_calculator = BusinessMetricCalculator(intent)
        else:
            self.business_calculator = None

    def tune(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[Dict[str, Any]]:
        """Tune threshold and return results for each threshold.

        Args:
            model: Trained model
            X: Features
            y: True labels

        Returns:
            List of results dictionaries (one per threshold)
        """
        if model.task_type == "regression":
            # For regression, return single result with no threshold
            predictions = model.predict(X)
            metrics = calculate_metrics(y, predictions, task_type="regression")
            return [{
                "threshold": None,
                "metrics": metrics,
                "business_metrics": None
            }]

        # Get probability predictions
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            # Binary classification: use positive class probability
            proba_positive = proba[:, 1]
        else:
            # Multiclass: use max probability (not ideal for threshold tuning)
            proba_positive = proba.max(axis=1)

        results = []

        for threshold in self.thresholds:
            # Apply threshold
            predictions = (proba_positive >= threshold).astype(int)

            # Calculate metrics (import here to avoid circular dependency)
            from ..evaluation.metrics import calculate_metrics
            metrics = calculate_metrics(y, predictions, task_type="binary_classification")

            # Calculate business metrics if intent provided
            business_metrics = None
            if self.business_calculator:
                business_metrics = self.business_calculator.calculate(
                    metrics.get("confusion_matrix", {})
                )

            results.append({
                "threshold": threshold,
                "metrics": metrics,
                "business_metrics": business_metrics
            })

        return results

    def find_best_threshold(
        self,
        results: List[Dict[str, Any]],
        metric: str = "net_value"
    ) -> Dict[str, Any]:
        """Find best threshold based on metric.

        Args:
            results: List of threshold results from tune()
            metric: Metric to optimize for ("net_value", "roi", "f1", "recall", etc.)

        Returns:
            Best result dictionary
        """
        if not results:
            raise ValueError("No results provided")

        # Filter results with the metric
        valid_results = []
        for result in results:
            if metric in ["net_value", "roi"]:
                if result.get("business_metrics") and metric in result["business_metrics"]:
                    valid_results.append(result)
            else:
                if result.get("metrics") and metric in result["metrics"]:
                    valid_results.append(result)

        if not valid_results:
            # Fallback to first result
            return results[0]

        # Find best based on metric
        if metric in ["net_value", "roi"]:
            best = max(valid_results, key=lambda x: x["business_metrics"][metric])
        else:
            best = max(valid_results, key=lambda x: x["metrics"][metric])

        return best
