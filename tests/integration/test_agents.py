"""Integration tests for agents."""

import pytest
from unittest.mock import Mock, MagicMock

from src.llm.ollama_provider import OllamaProvider
from src.agents.intent_agent import IntentAgent
from src.agents.profiler_agent import ProfilerAgent


@pytest.fixture
def mock_llm():
    """Create mock LLM provider."""
    llm = Mock(spec=OllamaProvider)
    llm.generate_json = Mock(return_value={
        "task_type": "binary_classification",
        "target_variable": "churn",
        "business_context": {
            "priority_metric": "recall",
            "true_positive_value": 500.0,
            "false_positive_cost": 20.0,
            "cost_ratio": 0.04
        }
    })
    return llm


def test_intent_agent(mock_llm):
    """Test Intent Agent."""
    agent = IntentAgent(mock_llm)

    user_input = "I want to predict customer churn. Catching churners is priority."
    intent = agent.parse(user_input)

    assert intent["task_type"] == "binary_classification"
    assert intent["target_variable"] == "churn"
    assert intent["business_context"]["priority_metric"] == "recall"


def test_profiler_agent(mock_llm, tmp_path):
    """Test Profiler Agent."""
    # Create test CSV
    import pandas as pd
    test_data = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": ["A", "B", "A", "B", "A"],
        "target": [0, 1, 0, 1, 0]
    })
    csv_path = tmp_path / "test.csv"
    test_data.to_csv(csv_path, index=False)

    agent = ProfilerAgent(mock_llm)

    intent = {
        "task_type": "binary_classification",
        "target_variable": "target",
        "business_context": {
            "priority_metric": "recall"
        }
    }

    profile = agent.profile(str(csv_path), intent)

    assert "data_types" in profile
    assert "statistics" in profile
    assert "missing_values" in profile
