"""End-to-end integration test for full pipeline."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.llm.ollama_provider import OllamaProvider
from src.agents.orchestrator import Orchestrator
from src.version_store import VersionStore


@pytest.fixture
def mock_llm():
    """Create mock LLM provider."""
    llm = Mock(spec=OllamaProvider)
    
    # Mock intent parsing
    llm.generate_json = Mock(side_effect=[
        # Intent parsing response
        {
            "task_type": "binary_classification",
            "target_variable": "target",
            "business_context": {
                "priority_metric": "recall",
                "true_positive_value": 500.0,
                "false_positive_cost": 20.0,
                "cost_ratio": 0.04
            }
        },
        # Profile response
        {
            "intent_flags": [],
            "insights": ["Test insight"]
        },
        # EDA response
        {
            "visualizations": [],
            "insights": ["EDA insight"]
        },
        # Strategy response
        [
            {
                "name": "strategy_1",
                "preprocessing_steps": [
                    {
                        "step_type": "imputation",
                        "method": "median",
                        "parameters": {},
                        "columns": []
                    }
                ],
                "model_candidates": ["random_forest"]
            }
        ],
        # Comparison response
        {
            "comparison_text": "Test comparison"
        }
    ])
    return llm


@pytest.fixture
def test_dataset(tmp_path):
    """Create test dataset."""
    df = pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "target": np.random.randint(0, 2, 100)
    })
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_full_pipeline_skeleton(mock_llm, test_dataset):
    """Test full pipeline skeleton (without actual training)."""
    # This is a skeleton test - full implementation would require
    # actual model training which is slow
    
    version_store = VersionStore(db_path=":memory:")  # In-memory DB for testing
    orchestrator = Orchestrator(mock_llm, version_store)

    user_input = "Predict target. Catching positives is priority."
    
    # Test intent parsing
    intent = orchestrator.parse_intent(user_input, test_dataset)
    assert intent is not None
    assert "task_type" in intent

    # Test profiling
    profile = orchestrator.profile_data(test_dataset)
    assert profile is not None

    # Note: Full pipeline test would require actual model training
    # which is skipped here for speed
