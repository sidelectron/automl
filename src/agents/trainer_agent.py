"""Trainer Agent for coordinating model training."""

from typing import Dict, Any, List, Optional
import pandas as pd
from pathlib import Path

from ..llm.llm_interface import LLMInterface
from ..version_store import VersionStore
from ..engine import ExecutionEngine
from ..ml.preprocessing import PreprocessingPipeline
from ..ml.models.factory import ModelFactory
from ..ml.training import ModelTrainer, ThresholdTuner
from ..ml.evaluation import ThresholdOptimizer


class TrainerAgent:
    """Agent for coordinating training process."""

    def __init__(
        self,
        llm_provider: LLMInterface,
        execution_engine: ExecutionEngine,
        version_store: VersionStore
    ):
        """Initialize Trainer Agent.

        Args:
            llm_provider: LLM provider instance
            execution_engine: Execution engine instance
            version_store: Version store for saving results
        """
        self.llm = llm_provider
        self.execution_engine = execution_engine
        self.version_store = version_store
        self.agent_name = "trainer"

    def train(
        self,
        strategies: List[Dict[str, Any]],
        dataset_path: str,
        intent: Dict[str, Any],
        selected_strategies: Optional[List[str]] = None,
        experiment_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Train models for selected strategies.

        Args:
            strategies: List of strategy dictionaries
            dataset_path: Path to dataset file
            intent: Parsed intent
            selected_strategies: Optional list of strategy names to execute
            experiment_id: Optional experiment ID for saving results

        Returns:
            List of training results
        """
        # Filter strategies if selection provided
        if selected_strategies:
            strategies = [s for s in strategies if s.get("name") in selected_strategies]

        # Execute strategies (parallel execution handled by engine)
        all_results = self.execution_engine.execute_strategies_parallel(
            strategies,
            dataset_path,
            intent
        )

        # Flatten results (one list per strategy -> single list)
        flattened_results = []
        for strategy_idx, strategy_results in enumerate(all_results):
            strategy_name = strategies[strategy_idx].get("name", f"strategy_{strategy_idx}")

            # Save strategy to version store
            if experiment_id:
                strategy_id = self.version_store.save_strategy(
                    experiment_id,
                    strategy_name,
                    strategies[strategy_idx]
                )

                # Save each result
                for result in strategy_results:
                    result_id = self.version_store.save_result(
                        strategy_id,
                        result.get("model_name"),
                        result.get("threshold", 0.5),
                        result.get("metrics", {}),
                        result.get("business_metrics"),
                        result.get("model_path"),
                        result.get("preprocessing_path")
                    )
                    result["result_id"] = result_id
                    result["strategy_name"] = strategy_name

            flattened_results.extend(strategy_results)

        return flattened_results
