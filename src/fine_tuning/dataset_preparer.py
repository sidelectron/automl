"""Dataset preparer for fine-tuning format conversion."""

import json
from pathlib import Path
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split


class DatasetPreparer:
    """Prepare datasets for fine-tuning."""

    def __init__(self):
        """Initialize dataset preparer."""
        pass

    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Load JSONL file.

        Args:
            file_path: Path to JSONL file

        Returns:
            List of dictionaries
        """
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def save_jsonl(self, data: List[Dict[str, Any]], file_path: str):
        """Save data to JSONL file.

        Args:
            data: List of dictionaries
            file_path: Output file path
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def split_dataset(
        self,
        dataset_path: str,
        train_ratio: float = 0.8,
        output_dir: str = "data/fine_tuning/datasets"
    ) -> Dict[str, str]:
        """Split dataset into train and validation sets.

        Args:
            dataset_path: Path to dataset JSONL file
            train_ratio: Ratio of training data (default: 0.8)
            output_dir: Output directory

        Returns:
            Dictionary with 'train' and 'val' file paths
        """
        data = self.load_jsonl(dataset_path)

        # Split data
        train_data, val_data = train_test_split(
            data,
            train_size=train_ratio,
            random_state=42
        )

        # Save splits
        base_name = Path(dataset_path).stem
        train_path = Path(output_dir) / f"{base_name}_train.jsonl"
        val_path = Path(output_dir) / f"{base_name}_val.jsonl"

        self.save_jsonl(train_data, str(train_path))
        self.save_jsonl(val_data, str(val_path))

        return {
            "train": str(train_path),
            "val": str(val_path)
        }

    def format_for_qlora(
        self,
        dataset_path: str,
        output_path: str
    ) -> str:
        """Format dataset for QLoRA fine-tuning.

        Args:
            dataset_path: Path to dataset JSONL file
            output_path: Output file path

        Returns:
            Path to formatted dataset
        """
        data = self.load_jsonl(dataset_path)

        # Format for QLoRA (instruction-following format)
        formatted_data = []
        for item in data:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output_text = item.get("output", "")

            # Create prompt in instruction-following format
            if input_text:
                prompt = f"{instruction}\n\nInput:\n{input_text}\n\nOutput:\n"
            else:
                prompt = f"{instruction}\n\nOutput:\n"

            formatted_item = {
                "instruction": prompt,
                "output": output_text
            }
            formatted_data.append(formatted_item)

        # Save formatted dataset
        self.save_jsonl(formatted_data, output_path)

        return output_path
