"""Intent Agent for parsing business goals from natural language."""

import json
from typing import Dict, Any, Optional
from pathlib import Path

from ..llm.llm_interface import LLMInterface
from ..contracts.models import IntentSchema, BusinessContext


class IntentAgent:
    """Agent for parsing user intent with business context."""

    def __init__(self, llm_provider: LLMInterface):
        """Initialize the Intent Agent.

        Args:
            llm_provider: LLM provider instance
        """
        self.llm = llm_provider
        self.agent_name = "intent"

    def _load_prompt_template(self) -> str:
        """Load the intent parsing prompt template.

        Returns:
            Prompt template string
        """
        prompt_path = Path(__file__).parent.parent / "llm" / "prompts" / "intent_parsing.txt"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return "Parse the user's requirement into JSON format with business context."

    def parse(self, user_input: str, dataset_path: Optional[str] = None) -> Dict[str, Any]:
        """Parse user intent from natural language.

        Args:
            user_input: Natural language description of the task
            dataset_path: Optional path to dataset file

        Returns:
            Parsed intent as dictionary matching IntentSchema

        Raises:
            ValueError: If parsing fails or validation fails
        """
        import pandas as pd

        # Load prompt template
        system_prompt = self._load_prompt_template()

        # Read dataset columns if path provided
        dataset_info = ""
        column_names = []
        if dataset_path:
            try:
                df = pd.read_csv(dataset_path, nrows=5)
                column_names = df.columns.tolist()
                dataset_info = f"""
# Dataset Information #
Columns available: {column_names}
Sample data (first 5 rows):
{df.to_string()}

IMPORTANT: The target_variable MUST be one of the actual column names: {column_names}
"""
            except Exception as e:
                dataset_info = f"\n# Dataset Path #\n{dataset_path}\n"

        # Construct user prompt
        user_prompt = f"""Please carefully parse the following user instruction.

# Instruction #
{user_input}
{dataset_info}
# Valid JSON Response #
"""

        # Generate JSON response using LLM
        try:
            response = self.llm.generate_json(
                prompt=system_prompt + "\n\n" + user_prompt,
                temperature=0.01,  # Low temperature for deterministic parsing
                agent=self.agent_name,
                timeout=180
            )
        except Exception as e:
            raise ValueError(f"Failed to parse intent: {e}") from e

        # Calculate cost_ratio if both values are provided
        if "business_context" in response:
            bc = response["business_context"]
            tp_value = bc.get("true_positive_value", 0.0)
            fp_cost = bc.get("false_positive_cost", 0.0)

            if tp_value > 0:
                bc["cost_ratio"] = fp_cost / tp_value
            else:
                bc["cost_ratio"] = 0.0

        # Add dataset_path if provided
        if dataset_path:
            response["dataset_path"] = dataset_path

        # Add description
        response["description"] = user_input

        # Validate against Pydantic model
        try:
            validated = IntentSchema(**response)
            return validated.model_dump()
        except Exception as e:
            raise ValueError(f"Intent validation failed: {e}") from e

    def parse_with_validation(
        self,
        user_input: str,
        dataset_path: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Parse intent with retry logic.

        Args:
            user_input: Natural language description
            dataset_path: Optional dataset path
            max_retries: Maximum retry attempts

        Returns:
            Validated intent dictionary

        Raises:
            ValueError: If all retries fail
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                return self.parse(user_input, dataset_path)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Update validation status in logger
                    if hasattr(self.llm, 'validate_and_relog'):
                        self.llm.validate_and_relog(
                            prompt=user_input,
                            response="",
                            agent=self.agent_name,
                            is_valid=False
                        )
                continue

        raise ValueError(f"Failed to parse intent after {max_retries} attempts: {last_error}") from last_error
