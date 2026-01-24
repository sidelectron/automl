"""Orchestrator for coordinating multi-agent workflow.

The orchestrator manages the state machine and coordinates all agents
to execute the full AutoML pipeline.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from pathlib import Path

from ..llm.llm_interface import LLMInterface
from ..version_store import VersionStore, QueryManager
from ..engine import ExecutionEngine
from ..project_generator import ProjectGenerator
from .intent_agent import IntentAgent
from .profiler_agent import ProfilerAgent
from .eda_agent import EDAAgent
from .strategy_agent import StrategyAgent
from .trainer_agent import TrainerAgent
from .comparator_agent import ComparatorAgent


class WorkflowState(Enum):
    """Workflow states."""

    INIT = "INIT"
    INTENT = "INTENT"
    PROFILE = "PROFILE"
    EDA = "EDA"
    STRATEGY = "STRATEGY"
    TRAIN = "TRAIN"
    COMPARE = "COMPARE"
    GENERATE = "GENERATE"
    END = "END"
    ERROR = "ERROR"


class Orchestrator:
    """Main orchestrator for coordinating agents."""

    def __init__(
        self,
        llm_provider: LLMInterface,
        version_store: Optional[VersionStore] = None
    ):
        """Initialize the orchestrator.

        Args:
            llm_provider: LLM provider instance
            version_store: Optional version store for experiment tracking
        """
        self.llm = llm_provider
        self.version_store = version_store or VersionStore()
        self.query_manager = QueryManager(self.version_store)
        self.execution_engine = ExecutionEngine()

        # State management
        self.state = WorkflowState.INIT
        self.experiment_id: Optional[str] = None

        # Workflow data
        self.intent: Optional[Dict[str, Any]] = None
        self.profile: Optional[Dict[str, Any]] = None
        self.eda_results: Optional[Dict[str, Any]] = None
        self.strategies: Optional[List[Dict[str, Any]]] = None
        self.training_results: Optional[List[Dict[str, Any]]] = None
        self.comparison: Optional[Dict[str, Any]] = None

        # Agent instances (will be initialized as agents are implemented)
        self.intent_agent = IntentAgent(self.llm)
        self.profiler_agent = ProfilerAgent(self.llm)
        self.eda_agent = EDAAgent(self.llm)
        self.strategy_agent = StrategyAgent(self.llm)
        self.trainer_agent = TrainerAgent(self.llm, self.execution_engine, self.version_store)
        self.comparator_agent = ComparatorAgent(self.llm)

    def parse_intent(self, user_input: str, dataset_path: Optional[str] = None) -> Dict[str, Any]:
        """Parse user intent from natural language.

        Args:
            user_input: Natural language description of the task
            dataset_path: Optional path to dataset file

        Returns:
            Parsed intent as dictionary

        Raises:
            NotImplementedError: Intent Agent not yet implemented
        """
        if self.intent_agent is None:
            raise NotImplementedError("Intent Agent not yet implemented")

        self.state = WorkflowState.INTENT
        self.intent = self.intent_agent.parse_with_validation(user_input, dataset_path)
        return self.intent

    def profile_data(self, dataset_path: str) -> Dict[str, Any]:
        """Profile the dataset with intent awareness.

        Args:
            dataset_path: Path to dataset file

        Returns:
            Data profile as dictionary

        Raises:
            NotImplementedError: Profiler Agent not yet implemented
        """
        if self.profiler_agent is None:
            raise NotImplementedError("Profiler Agent not yet implemented")

        self.state = WorkflowState.PROFILE
        self.profile = self.profiler_agent.profile(dataset_path, self.intent)
        return self.profile

    def generate_eda(self) -> Dict[str, Any]:
        """Generate target-focused EDA.

        Returns:
            EDA results as dictionary

        Raises:
            NotImplementedError: EDA Agent not yet implemented
        """
        if self.eda_agent is None:
            raise NotImplementedError("EDA Agent not yet implemented")

        self.state = WorkflowState.EDA
        dataset_path = self.intent.get("dataset_path") if self.intent else None
        self.eda_results = self.eda_agent.generate(self.profile, self.intent, dataset_path)
        return self.eda_results

    def propose_strategies(self) -> List[Dict[str, Any]]:
        """Propose preprocessing strategies.

        Returns:
            List of strategy dictionaries

        Raises:
            NotImplementedError: Strategy Agent not yet implemented
        """
        if self.strategy_agent is None:
            raise NotImplementedError("Strategy Agent not yet implemented")

        self.state = WorkflowState.STRATEGY
        self.strategies = self.strategy_agent.propose(self.profile, self.intent, self.eda_results)
        return self.strategies

    def train_models(self, selected_strategies: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Train models for selected strategies.

        Args:
            selected_strategies: Optional list of strategy names to execute.
                                If None, executes all strategies.

        Returns:
            List of training results

        Raises:
            NotImplementedError: Trainer Agent not yet implemented
        """
        if self.trainer_agent is None:
            raise NotImplementedError("Trainer Agent not yet implemented")

        self.state = WorkflowState.TRAIN
        dataset_path = self.intent.get("dataset_path") if self.intent else None
        if not dataset_path:
            raise ValueError("Dataset path not found in intent")

        self.training_results = self.trainer_agent.train(
            self.strategies,
            dataset_path,
            self.intent,
            selected_strategies,
            self.experiment_id
        )
        return self.training_results

    def compare_results(self) -> Dict[str, Any]:
        """Compare results and select winner.

        Returns:
            Comparison results with winner and business explanation

        Raises:
            NotImplementedError: Comparator Agent not yet implemented
        """
        if self.comparator_agent is None:
            raise NotImplementedError("Comparator Agent not yet implemented")

        self.state = WorkflowState.COMPARE
        self.comparison = self.comparator_agent.compare(self.training_results, self.intent)
        return self.comparison

    def generate_project(self, output_dir: str = "data/generated_projects") -> str:
        """Generate complete Python project.

        Args:
            output_dir: Directory to save generated project

        Returns:
            Path to generated project
        """
        if not self.comparison:
            raise ValueError("Comparison results not available. Run compare_results() first.")

        self.state = WorkflowState.GENERATE

        generator = ProjectGenerator(base_dir=output_dir)
        winner = self.comparison.get("winner", {})

        project_path = generator.generate(
            self.intent,
            winner,
            self.comparison,
            self.experiment_id
        )

        return project_path

    def run_full_pipeline(
        self,
        user_input: str,
        dataset_path: str,
        selected_strategies: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run the complete pipeline from intent to project generation.

        Args:
            user_input: Natural language description of the task
            dataset_path: Path to dataset file
            selected_strategies: Optional list of strategy names to execute

        Returns:
            Complete pipeline results
        """
        try:
            # Step 1: Parse intent
            self.intent = self.parse_intent(user_input, dataset_path)
            self.experiment_id = self.version_store.save_experiment(
                self.intent,
                dataset_path,
                user_input
            )

            # Step 2: Profile data
            self.profile = self.profile_data(dataset_path)

            # Step 3: Generate EDA
            self.eda_results = self.generate_eda()

            # Step 4: Propose strategies
            self.strategies = self.propose_strategies()

            # Step 5: Train models
            self.training_results = self.train_models(selected_strategies)

            # Step 6: Compare results
            self.comparison = self.compare_results()
            if self.comparison and self.experiment_id:
                self.version_store.save_comparison(
                    self.experiment_id,
                    self.comparison
                )

            # Step 7: Generate project
            try:
                project_path = self.generate_project()
            except Exception as e:
                print(f"Warning: Project generation failed: {e}")
                project_path = None

            self.state = WorkflowState.END

            return {
                "experiment_id": self.experiment_id,
                "intent": self.intent,
                "profile": self.profile,
                "eda": self.eda_results,
                "strategies": self.strategies,
                "results": self.training_results,
                "comparison": self.comparison,
                "project_path": project_path
            }

        except Exception as e:
            self.state = WorkflowState.ERROR
            raise
