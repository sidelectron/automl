"""Unit tests for models module."""

import pytest
import pandas as pd
import numpy as np

from src.ml.models.factory import ModelFactory


def test_model_factory():
    """Test model factory."""
    # Test creating XGBoost model
    try:
        model = ModelFactory.create("xgboost", task_type="binary_classification")
        assert model is not None
        assert model.task_type == "binary_classification"
    except ImportError:
        pytest.skip("XGBoost not installed")

    # Test creating LightGBM model
    try:
        model = ModelFactory.create("lightgbm", task_type="binary_classification")
        assert model is not None
    except ImportError:
        pytest.skip("LightGBM not installed")

    # Test creating Random Forest model
    model = ModelFactory.create("random_forest", task_type="binary_classification")
    assert model is not None

    # Test invalid model name
    with pytest.raises(ValueError):
        ModelFactory.create("invalid_model")


def test_model_training():
    """Test model training."""
    # Create simple dataset
    X = pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100)
    })
    y = pd.Series(np.random.randint(0, 2, 100))

    # Test Random Forest (always available)
    model = ModelFactory.create("random_forest", task_type="binary_classification")
    model.train(X, y)

    # Test prediction
    predictions = model.predict(X)
    assert len(predictions) == len(y)

    # Test predict_proba
    proba = model.predict_proba(X)
    assert proba.shape[0] == len(y)
