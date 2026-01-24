"""Unit tests for preprocessing module."""

import pytest
import pandas as pd
import numpy as np

from src.ml.preprocessing import (
    MedianImputer,
    StandardScaler,
    OneHotEncoder,
    PreprocessingPipeline
)


def test_median_imputer():
    """Test median imputation."""
    df = pd.DataFrame({
        "col1": [1, 2, np.nan, 4, 5],
        "col2": [10, 20, 30, np.nan, 50]
    })

    imputer = MedianImputer()
    imputer.fit(df)
    result = imputer.transform(df)

    assert result["col1"].isna().sum() == 0
    assert result["col2"].isna().sum() == 0


def test_standard_scaler():
    """Test standard scaling."""
    df = pd.DataFrame({
        "col1": [1, 2, 3, 4, 5],
        "col2": [10, 20, 30, 40, 50]
    })

    scaler = StandardScaler()
    scaler.fit(df)
    result = scaler.transform(df)

    assert result["col1"].std() < 2  # Should be close to 1 after scaling
    assert result["col2"].std() < 2


def test_preprocessing_pipeline():
    """Test preprocessing pipeline."""
    df = pd.DataFrame({
        "numeric": [1, 2, np.nan, 4, 5],
        "categorical": ["A", "B", "A", "B", "A"]
    })

    strategy = {
        "name": "test_strategy",
        "preprocessing_steps": [
            {
                "step_type": "imputation",
                "method": "median",
                "parameters": {},
                "columns": ["numeric"]
            }
        ],
        "model_candidates": ["xgboost"]
    }

    pipeline = PreprocessingPipeline()
    pipeline.build_from_strategy(strategy)
    result = pipeline.fit_transform(df)

    assert result.shape[0] == df.shape[0]
