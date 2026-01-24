"""Fine-tuning data logger for collecting LLM interactions.

This module logs all (prompt, response) pairs to build a fine-tuning dataset.
All interactions are saved in JSONL format compatible with LLaMA-Factory.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class FineTuningLogger:
    """Logger for collecting LLM interaction data for future fine-tuning.

    Logs every LLM call with:
    - Prompt and response content
    - Agent that made the call
    - Validation status (whether output was valid)
    - Timestamp for tracking

    Output format is JSONL, compatible with LLaMA-Factory and similar tools.
    """

    def __init__(self, output_dir: str = "data/fine_tuning"):
        """Initialize the fine-tuning logger.

        Args:
            output_dir: Directory to save fine-tuning data (default: data/fine_tuning)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_interaction(
        self,
        prompt: str,
        response: str,
        agent: str,
        valid: bool,
        metadata: Optional[dict] = None
    ) -> None:
        """Log a single LLM interaction.

        Args:
            prompt: The input prompt sent to the LLM
            response: The LLM's response
            agent: Name of the agent that made the call (e.g., "intent", "profiler")
            valid: Whether the response passed validation
            metadata: Optional additional metadata to store
        """
        timestamp = datetime.now().isoformat()

        # Create the interaction record in LLaMA-Factory format
        interaction = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ],
            "metadata": {
                "agent": agent,
                "valid": valid,
                "timestamp": timestamp,
                **(metadata or {})
            }
        }

        # Save to agent-specific file
        output_file = self.output_dir / f"{agent}_interactions.jsonl"

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(interaction, ensure_ascii=False) + "\n")

    def get_stats(self) -> dict:
        """Get statistics about logged interactions.

        Returns:
            Dictionary with counts per agent and overall statistics
        """
        stats = {
            "total_interactions": 0,
            "valid_interactions": 0,
            "invalid_interactions": 0,
            "by_agent": {}
        }

        for jsonl_file in self.output_dir.glob("*_interactions.jsonl"):
            agent_name = jsonl_file.stem.replace("_interactions", "")
            agent_stats = {"total": 0, "valid": 0, "invalid": 0}

            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    interaction = json.loads(line)
                    agent_stats["total"] += 1
                    stats["total_interactions"] += 1

                    if interaction["metadata"]["valid"]:
                        agent_stats["valid"] += 1
                        stats["valid_interactions"] += 1
                    else:
                        agent_stats["invalid"] += 1
                        stats["invalid_interactions"] += 1

            stats["by_agent"][agent_name] = agent_stats

        return stats

    def export_for_training(self, output_file: str, agent: Optional[str] = None) -> None:
        """Export interactions in format ready for fine-tuning.

        Args:
            output_file: Path to output file
            agent: If specified, only export interactions from this agent
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as out_f:
            if agent:
                # Export specific agent
                input_file = self.output_dir / f"{agent}_interactions.jsonl"
                if input_file.exists():
                    with open(input_file, "r", encoding="utf-8") as in_f:
                        for line in in_f:
                            interaction = json.loads(line)
                            # Only export valid interactions for training
                            if interaction["metadata"]["valid"]:
                                # Remove metadata for cleaner training data
                                training_sample = {
                                    "messages": interaction["messages"]
                                }
                                out_f.write(json.dumps(training_sample, ensure_ascii=False) + "\n")
            else:
                # Export all agents
                for jsonl_file in self.output_dir.glob("*_interactions.jsonl"):
                    with open(jsonl_file, "r", encoding="utf-8") as in_f:
                        for line in in_f:
                            interaction = json.loads(line)
                            # Only export valid interactions for training
                            if interaction["metadata"]["valid"]:
                                training_sample = {
                                    "messages": interaction["messages"]
                                }
                                out_f.write(json.dumps(training_sample, ensure_ascii=False) + "\n")
