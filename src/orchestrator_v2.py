"""Orchestrator v2 for fully dynamic AutoML version."""

from enum import Enum
from typing import Optional, Dict, Any, List
from pathlib import Path

from .llm.llm_interface import LLMInterface
from .version_store import VersionStore, QueryManager
from .engine_v2 import DynamicExecutor
from .engine_v2.code_validator import CodeValidator
from .agents.intent_agent import IntentAgent
from .agents.profiler_agent import ProfilerAgent
from .agents.eda_agent import EDAAgent
from .agents.comparator_agent import ComparatorAgent
from .agents_v2.code_generation_agent import CodeGenerationAgent
from .agents_v2.code_fixer_agent import CodeFixerAgent
from .agents_v2.strategy_agent_v2 import StrategyAgentV2
from .agents_v2.model_agent_v2 import ModelAgentV2


class WorkflowStateV2(Enum):
    """Workflow states for dynamic version."""

    INIT = "INIT"
    INTENT = "INTENT"
    PROFILE = "PROFILE"
    EDA = "EDA"
    PLAN = "PLAN"
    CODE_GENERATE = "CODE_GENERATE"
    EXECUTE = "EXECUTE"
    VERIFY = "VERIFY"
    COMPARE = "COMPARE"
    END = "END"
    ERROR = "ERROR"


class OrchestratorV2:
    """Orchestrator for fully dynamic AutoML version."""

    def __init__(
        self,
        llm_provider: LLMInterface,
        version_store: Optional[VersionStore] = None,
        work_dir: Optional[str] = None
    ):
        """Initialize orchestrator v2.

        Args:
            llm_provider: LLM provider instance
            version_store: Optional version store for experiment tracking
            work_dir: Working directory for code execution
        """
        self.llm = llm_provider
        self.version_store = version_store or VersionStore()
        self.query_manager = QueryManager(self.version_store)

        # State management
        self.state = WorkflowStateV2.INIT
        self.experiment_id: Optional[str] = None

        # Workflow data
        self.intent: Optional[Dict[str, Any]] = None
        self.profile: Optional[Dict[str, Any]] = None
        self.eda_results: Optional[Dict[str, Any]] = None
        self.preprocessing_plan: Optional[str] = None
        self.modeling_plan: Optional[str] = None
        self.generated_code: Dict[str, str] = {}  # code_type -> code_content
        self.execution_results: Dict[str, Any] = {}  # code_type -> ExecutionResult
        self.comparison: Optional[Dict[str, Any]] = None

        # Agent instances
        self.intent_agent = IntentAgent(self.llm)
        self.profiler_agent = ProfilerAgent(self.llm)
        self.eda_agent = EDAAgent(self.llm)
        self.strategy_agent_v2 = StrategyAgentV2(self.llm)
        self.model_agent_v2 = ModelAgentV2(self.llm)
        self.code_generation_agent = CodeGenerationAgent(self.llm)
        self.code_fixer_agent = CodeFixerAgent(self.llm)
        self.comparator_agent = ComparatorAgent(self.llm)

        # Execution engine (code_fixer will be passed to execute_with_retry)
        self.executor = DynamicExecutor(work_dir=work_dir)
        self.validator = CodeValidator()

    def parse_intent(self, user_input: str, dataset_path: Optional[str] = None) -> Dict[str, Any]:
        """Parse user intent from natural language.

        Args:
            user_input: Natural language description of the task
            dataset_path: Optional path to dataset file

        Returns:
            Parsed intent as dictionary
        """
        self.state = WorkflowStateV2.INTENT
        self.intent = self.intent_agent.parse_with_validation(user_input, dataset_path)
        return self.intent

    def profile_data(self, dataset_path: str) -> Dict[str, Any]:
        """Profile the dataset with intent awareness.

        Args:
            dataset_path: Path to dataset file

        Returns:
            Data profile dictionary
        """
        self.state = WorkflowStateV2.PROFILE
        self.profile = self.profiler_agent.profile(dataset_path, self.intent)
        return self.profile

    def generate_eda(self) -> Dict[str, Any]:
        """Generate target-focused EDA.

        Returns:
            EDA results dictionary
        """
        if not self.profile or not self.intent:
            raise ValueError("Profile and intent must be available before generating EDA")

        self.state = WorkflowStateV2.EDA
        dataset_path = self.intent.get("dataset_path")
        if not dataset_path:
            raise ValueError("Dataset path not found in intent")

        self.eda_results = self.eda_agent.generate(
            self.profile,
            self.intent,
            dataset_path
        )
        return self.eda_results

    def generate_plans(self) -> Dict[str, str]:
        """Generate text-based plans (preprocessing + modeling).

        Returns:
            Dictionary with 'preprocessing_plan' and 'modeling_plan'
        """
        if not self.profile or not self.intent:
            raise ValueError("Profile and intent must be available before generating plans")

        self.state = WorkflowStateV2.PLAN

        # Generate preprocessing plan
        self.preprocessing_plan = self.strategy_agent_v2.generate_preprocessing_plan(
            self.profile,
            self.intent,
            self.eda_results
        )

        # Generate modeling plan
        self.modeling_plan = self.model_agent_v2.generate_modeling_plan(
            self.preprocessing_plan,
            self.intent
        )

        return {
            "preprocessing_plan": self.preprocessing_plan,
            "modeling_plan": self.modeling_plan
        }

    def generate_code(self) -> Dict[str, str]:
        """Generate Python code from plans.

        Returns:
            Dictionary with generated code (preprocessing, training, prediction)
        """
        if not self.preprocessing_plan or not self.modeling_plan:
            raise ValueError("Plans must be generated before code generation")

        self.state = WorkflowStateV2.CODE_GENERATE

        # Generate preprocessing code
        preprocessing_code = self.code_generation_agent.generate_preprocessing_code(
            self.preprocessing_plan,
            self.profile,
            self.intent
        )
        self.generated_code["preprocessing"] = preprocessing_code

        # Generate training code
        training_code = self.code_generation_agent.generate_training_code(
            preprocessing_code,
            self.modeling_plan,
            self.intent
        )
        self.generated_code["training"] = training_code

        # Generate prediction code
        prediction_code = self.code_generation_agent.generate_prediction_code(
            training_code,
            self.intent
        )
        self.generated_code["prediction"] = prediction_code

        # Save generated code to version store
        if self.experiment_id:
            for code_type, code_content in self.generated_code.items():
                self.version_store.save_generated_code(
                    self.experiment_id,
                    code_type,
                    code_content
                )

        return self.generated_code

    def execute_code(
        self,
        code_type: str = "preprocessing",
        max_attempts: int = 5
    ) -> Dict[str, Any]:
        """Execute generated code with retry logic.

        Args:
            code_type: Type of code to execute ('preprocessing', 'training', 'prediction')
            max_attempts: Maximum retry attempts

        Returns:
            Execution result dictionary
        """
        if code_type not in self.generated_code:
            raise ValueError(f"Code for {code_type} not generated yet")

        self.state = WorkflowStateV2.EXECUTE

        code = self.generated_code[code_type]
        script_name = f"{code_type}.py"

        # Validate code before execution
        is_valid, errors = self.validator.validate(code)
        if not is_valid:
            for error in errors:
                if "dangerous operation" in error.lower():
                    raise ValueError(
                        f"Validation failed (security): {error}. "
                        "Generated code uses disallowed operations; cannot auto-fix."
                    )
            for error in errors:
                err_lower = error.lower()
                if "import" in err_lower:
                    code = self.code_fixer_agent.fix_import_error(code, error)
                elif "syntax" in err_lower or "parse" in err_lower:
                    code = self.code_fixer_agent.fix_syntax_error(code, error)
            self.generated_code[code_type] = code

        # Execute with retry (longer timeout for training)
        exec_timeout = 120 if code_type == "training" else 30
        def fix_callback(code, error, error_type, execution_log):
            return self.code_fixer_agent.fix_code(code, error, error_type, execution_log)

        result = self.executor.execute_with_retry(
            code,
            script_name,
            max_attempts=max_attempts,
            fix_code_callback=fix_callback,
            timeout=exec_timeout
        )

        self.execution_results[code_type] = result

        # Save execution record
        if self.experiment_id:
            code_record = self.version_store.get_generated_code(self.experiment_id, code_type)
            if code_record:
                self.version_store.save_execution(
                    code_record["code_id"],
                    attempt_number=1,  # Would track actual attempts in full implementation
                    success=result.success,
                    execution_log=result.stdout + "\n" + result.stderr,
                    error_type=result.error_type
                )

        return {
            "code_type": code_type,
            "success": result.success,
            "return_code": result.return_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "execution_time": result.execution_time,
            "error_type": result.error_type
        }

    def verify_execution(self) -> bool:
        """Verify that code execution was successful.

        Returns:
            True if verification passes, False otherwise
        """
        self.state = WorkflowStateV2.VERIFY

        # Check if all required code executed successfully
        required_types = ["preprocessing", "training"]
        for code_type in required_types:
            if code_type not in self.execution_results:
                return False
            if not self.execution_results[code_type].success:
                return False

        return True

    def compare_results(self) -> Dict[str, Any]:
        """Compare results and select winner.

        Returns:
            Comparison results with winner and business explanation
        """
        if not self.execution_results:
            raise ValueError("No execution results available for comparison")

        self.state = WorkflowStateV2.COMPARE

        training_exec = self.execution_results.get("training")
        success = training_exec.success if hasattr(training_exec, "success") else (training_exec or {}).get("success")
        if not training_exec or not success:
            self.comparison = {
                "winner": None,
                "error": "Training execution failed",
                "comparison_text": "Unable to compare results due to execution failure."
            }
            if self.comparison and self.experiment_id:
                self.version_store.save_comparison(self.experiment_id, self.comparison)
            return self.comparison

        import json
        import re

        results_path = self.executor.work_dir / "results.json"
        training_results: List[Dict[str, Any]] = []

        def parse_results_list(raw: Any) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            if not isinstance(raw, list):
                return out
            for r in raw:
                if isinstance(r, dict) and ("metrics" in r or "business_metrics" in r):
                    out.append(r)
            return out

        if results_path.exists():
            try:
                raw = json.loads(results_path.read_text(encoding="utf-8"))
                training_results = parse_results_list(raw)
            except Exception:
                pass

        if not training_results and training_exec:
            stdout = getattr(training_exec, "stdout", "") or ""
            stderr = getattr(training_exec, "stderr", "") or ""
            combined = stdout + "\n" + stderr
            for blob in re.findall(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", combined):
                try:
                    raw = json.loads(blob)
                    training_results = parse_results_list(raw)
                    if training_results:
                        results_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
                        break
                except Exception:
                    continue

        if training_results and self.intent:
            try:
                self.comparison = self.comparator_agent.compare(training_results, self.intent)
            except Exception:
                self.comparison = {
                    "winner": {"code_type": "training", "success": True},
                    "business_impact": {"explanation": "Training completed; comparison parse failed."},
                    "comparison_text": "Training completed successfully. Structured comparison unavailable."
                }
        else:
            self.comparison = {
                "winner": {"code_type": "training", "success": True},
                "business_impact": {"explanation": "Results extracted from executed training code"},
                "comparison_text": "Training completed successfully. Results extracted from execution output."
            }

        if self.comparison and self.experiment_id:
            self.version_store.save_comparison(self.experiment_id, self.comparison)
        return self.comparison

    def run_full_pipeline(
        self,
        user_input: str,
        dataset_path: str,
        work_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run the complete dynamic pipeline.

        Args:
            user_input: Natural language description of the task
            dataset_path: Path to dataset file
            work_dir: Optional working directory for code execution (default: project root derived from dataset path)

        Returns:
            Complete pipeline results
        """
        try:
            # Resolve dataset path to absolute; store for codegen and execution
            dataset_path_resolved = str(Path(dataset_path).resolve())

            # Step 1: Parse intent
            self.intent = self.parse_intent(user_input, dataset_path_resolved)
            self.intent["dataset_path"] = dataset_path_resolved

            self.experiment_id = self.version_store.save_experiment(
                self.intent,
                dataset_path_resolved,
                user_input
            )

            # Derive work_dir if not provided: directory containing the dataset (e.g. project/archive/file -> project/archive).
            # This works for archive/, data/, or project-root datasets. Prefer passing work_dir explicitly (e.g. project root).
            if work_dir is None:
                work_dir = str(Path(dataset_path_resolved).resolve().parent)
            self.executor = DynamicExecutor(work_dir=work_dir)

            # Step 2: Profile data
            self.profile = self.profile_data(dataset_path_resolved)

            # Step 3: Generate EDA
            self.eda_results = self.generate_eda()

            # Step 4: Generate plans
            plans = self.generate_plans()

            # Step 5: Generate code
            code = self.generate_code()

            # Step 6: Execute code
            execution_results = {}
            for code_type in ["preprocessing", "training"]:
                result = self.execute_code(code_type)
                execution_results[code_type] = result

            # Step 7: Verify execution
            verification_passed = self.verify_execution()

            # Step 8: Compare results
            if verification_passed:
                self.comparison = self.compare_results()
            else:
                self.comparison = {
                    "error": "Execution verification failed",
                    "execution_results": execution_results
                }

            self.state = WorkflowStateV2.END

            return {
                "experiment_id": self.experiment_id,
                "intent": self.intent,
                "profile": self.profile,
                "eda_results": self.eda_results,
                "plans": plans,
                "generated_code": code,
                "execution_results": execution_results,
                "verification_passed": verification_passed,
                "comparison": self.comparison
            }

        except Exception as e:
            self.state = WorkflowStateV2.ERROR
            return {
                "error": str(e),
                "state": self.state.value
            }
