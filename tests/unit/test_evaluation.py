"""Unit tests for evaluation module."""

import pytest
import numpy as np

from src.ml.evaluation.metrics import calculate_metrics
from src.ml.evaluation.business_translator import BusinessMetricCalculator


def test_calculate_metrics_binary():
    """Test metrics calculation for binary classification."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])

    metrics = calculate_metrics(y_true, y_pred, task_type="binary_classification")

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "confusion_matrix" in metrics


def test_calculate_metrics_regression():
    """Test metrics calculation for regression."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])

    metrics = calculate_metrics(y_true, y_pred, task_type="regression")

    assert "mse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert "rmse" in metrics


def test_business_metric_calculator():
    """Test business metrics calculation."""
    intent = {
        "business_context": {
            "true_positive_value": 500.0,
            "false_positive_cost": 20.0,
            "cost_ratio": 0.04
        }
    }

    calculator = BusinessMetricCalculator(intent)

    confusion_matrix = {
        "true_positive": 100,
        "false_positive": 50,
        "true_negative": 200,
        "false_negative": 50
    }

    business_metrics = calculator.calculate(confusion_matrix)

    assert "net_value" in business_metrics
    assert "roi" in business_metrics
    assert "total_cost" in business_metrics
    assert "total_value" in business_metrics

    # Verify calculations
    expected_value = 100 * 500.0
    expected_cost = 150 * 20.0
    expected_net = expected_value - expected_cost

    assert abs(business_metrics["net_value"] - expected_net) < 0.01
