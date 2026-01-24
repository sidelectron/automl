"""Strategy Agent for proposing preprocessing strategies."""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..llm.llm_interface import LLMInterface
from ..contracts.models import StrategySchema


class StrategyAgent:
    """Agent for generating preprocessing strategies."""

    def __init__(self, llm_provider: LLMInterface):
        """Initialize the Strategy Agent.

        Args:
            llm_provider: LLM provider instance
        """
        self.llm = llm_provider
        self.agent_name = "strategy"

    def _load_prompt_template(self) -> str:
        """Load the strategy generation prompt template.

        Returns:
            Prompt template string
        """
        prompt_path = Path(__file__).parent.parent / "llm" / "prompts" / "strategy_generation.txt"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return "Generate preprocessing strategies based on data profile and intent."

    def propose(
        self,
        profile: Dict[str, Any],
        intent: Dict[str, Any],
        eda_results: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Propose preprocessing strategies.

        Args:
            profile: Data profile from Profiler Agent
            intent: Parsed intent from Intent Agent
            eda_results: Optional EDA results from EDA Agent

        Returns:
            List of strategy dictionaries matching StrategySchema
        """
        system_prompt = self._load_prompt_template()

        # Build context
        context = {
            "profile": profile,
            "intent": intent
        }

        if eda_results:
            context["eda"] = eda_results

        user_prompt = f"""Generate preprocessing strategies based on the following context.

# Context #
```json
{json.dumps(context, indent=2)}
```

Generate 3-5 diverse preprocessing strategies that consider:
1. The data profile (types, missing values, class distribution)
2. User intent (priority metric, business context)
3. EDA insights (if available)

Each strategy should be self-contained and executable.
"""

        try:
            response = self.llm.generate_json(
                prompt=system_prompt + "\n\n" + user_prompt,
                temperature=0.3,  # Some creativity for diverse strategies
                agent=self.agent_name
            )

            # Ensure response is a list
            if isinstance(response, dict):
                # Sometimes LLM returns single strategy wrapped in dict
                if "strategies" in response:
                    strategies = response["strategies"]
                else:
                    strategies = [response]
            else:
                strategies = response

            # Normalize LLM output: convert 'type' to 'step_type' in preprocessing_steps
            # (LLM sometimes uses 'type' instead of 'step_type')
            for strategy in strategies:
                if "preprocessing_steps" in strategy:
                    for step in strategy["preprocessing_steps"]:
                        if "type" in step and "step_type" not in step:
                            step["step_type"] = step.pop("type")

            # Validate each strategy
            validated_strategies = []
            for strategy in strategies:
                try:
                    validated = StrategySchema(**strategy)
                    validated_strategies.append(validated.model_dump())
                except Exception as e:
                    # Skip invalid strategies but log error
                    print(f"Warning: Invalid strategy skipped: {e}")
                    continue

            # If no valid strategies from LLM, use fallback
            if not validated_strategies:
                print("Info: No valid LLM strategies, using fallback strategy")
                return self._generate_fallback_strategy(profile, intent)

            return validated_strategies

        except Exception as e:
            # Fallback: generate basic strategy
            print(f"Info: LLM strategy generation failed ({e}), using fallback")
            return self._generate_fallback_strategy(profile, intent)

    def _generate_fallback_strategy(
        self,
        profile: Dict[str, Any],
        intent: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate a basic fallback strategy if LLM fails.

        Args:
            profile: Data profile (can be None)
            intent: Parsed intent

        Returns:
            List with one basic strategy
        """
        task_type = intent.get("task_type", "binary_classification")
        target_variable = intent.get("target_variable", "target")
        
        # Handle None profile gracefully
        if profile is None:
            profile = {}
        
        data_types = profile.get("data_types", {})
        missing_values = profile.get("missing_values", {})

        # Basic preprocessing steps
        steps = []

        # Exclude target variable from feature columns
        numeric_cols = [col for col, dtype in data_types.items()
                       if dtype == "numeric" and col != target_variable]
        categorical_cols = [col for col, dtype in data_types.items()
                          if dtype == "categorical" and col != target_variable]

        if any(missing_values.values()):
            if numeric_cols:
                steps.append({
                    "step_type": "imputation",
                    "method": "median",
                    "parameters": {},
                    "columns": numeric_cols
                })
            if categorical_cols:
                steps.append({
                    "step_type": "imputation",
                    "method": "mode",
                    "parameters": {},
                    "columns": categorical_cols
                })

        # Scaling for numeric features
        if numeric_cols:
            steps.append({
                "step_type": "scaling",
                "method": "standard_scaler",
                "parameters": {},
                "columns": numeric_cols
            })

        # Encoding for categorical features
        if categorical_cols:
            steps.append({
                "step_type": "encoding",
                "method": "one_hot",
                "parameters": {},
                "columns": categorical_cols
            })

        # Handle imbalance for classification
        if task_type in ["binary_classification", "multiclass_classification"]:
            priority_metric = intent.get("business_context", {}).get("priority_metric", "f1")
            if priority_metric == "recall":
                steps.append({
                    "step_type": "handling_imbalance",
                    "method": "smote",
                    "parameters": {},
                    "columns": []
                })

        # Default model candidates
        if task_type == "regression":
            model_candidates = ["xgboost", "lightgbm", "random_forest"]
        else:
            model_candidates = ["xgboost", "lightgbm", "random_forest", "logistic_regression"]

        return [{
            "name": "basic_strategy",
            "preprocessing_steps": steps,
            "model_candidates": model_candidates,
            "rationale": "Basic preprocessing strategy with standard imputation, scaling, and encoding.",
            "expected_outcomes": {}
        }]
