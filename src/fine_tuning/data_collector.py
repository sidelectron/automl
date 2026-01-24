"""Data collector for fine-tuning dataset preparation."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class DataCollector:
    """Collect LLM interactions for fine-tuning dataset."""

    def __init__(
        self,
        log_dir: str = "data/fine_tuning/logs"
    ):
        """Initialize data collector.

        Args:
            log_dir: Directory containing fine-tuning logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def collect_from_logs(
        self,
        agent_name: str,
        filter_valid: bool = True
    ) -> List[Dict[str, Any]]:
        """Collect interactions from fine-tuning logs.

        Args:
            agent_name: Name of agent to collect from (e.g., 'code_generation')
            filter_valid: Only collect validated interactions

        Returns:
            List of interaction dictionaries
        """
        log_file = self.log_dir / f"{agent_name}_interactions.jsonl"
        if not log_file.exists():
            return []

        interactions = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    interaction = json.loads(line.strip())
                    if not filter_valid or interaction.get("valid", False):
                        interactions.append(interaction)
                except json.JSONDecodeError:
                    continue

        return interactions

    def prepare_code_generation_dataset(
        self,
        output_file: str = "data/fine_tuning/datasets/code_generation.jsonl"
    ) -> str:
        """Prepare fine-tuning dataset for code generation.

        Args:
            output_file: Output file path

        Returns:
            Path to created dataset file
        """
        # Collect from code generation agent
        interactions = self.collect_from_logs("code_generation", filter_valid=True)

        dataset = []
        for interaction in interactions:
            # Extract instruction, input, and output
            prompt = interaction.get("prompt", "")
            response = interaction.get("response", "")

            # Parse prompt to extract context
            # This is a simplified version - in practice, you'd parse more carefully
            if "Generate" in prompt and "preprocessing" in prompt.lower():
                instruction = "Generate preprocessing pipeline in Python using sklearn"
                input_text = prompt  # Full prompt as input
                output_text = response  # Generated code

                dataset.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text
                })

        # Write dataset
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return str(output_path)

    def prepare_code_fixing_dataset(
        self,
        output_file: str = "data/fine_tuning/datasets/code_fixing.jsonl"
    ) -> str:
        """Prepare fine-tuning dataset for code fixing.

        Args:
            output_file: Output file path

        Returns:
            Path to created dataset file
        """
        # Collect from code fixer agent
        interactions = self.collect_from_logs("code_fixer", filter_valid=True)

        dataset = []
        for interaction in interactions:
            prompt = interaction.get("prompt", "")
            response = interaction.get("response", "")

            if "error" in prompt.lower() and "fix" in prompt.lower():
                instruction = "Fix the error in the Python code"
                input_text = prompt
                output_text = response

                dataset.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text
                })

        # Write dataset
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return str(output_path)
