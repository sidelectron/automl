"""Code generator for project files."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import shutil


class CodeGenerator:
    """Generate Python code files for the project."""

    def __init__(self, project_dir: Path):
        """Initialize code generator.

        Args:
            project_dir: Path to project directory
        """
        self.project_dir = project_dir

    def generate_preprocessing(
        self,
        strategy: Dict[str, Any],
        preprocessing_path: Optional[str] = None
    ):
        """Generate preprocessing.py from strategy.

        Args:
            strategy: Winning strategy dictionary
            preprocessing_path: Optional path to saved preprocessing pipeline
        """
        preprocessing_steps = strategy.get("preprocessing_steps", [])

        code = '''"""Data preprocessing pipeline.

This module implements the preprocessing steps for the ML pipeline.
"""

import pandas as pd
import pickle
from pathlib import Path


def load_preprocessing_pipeline(pipeline_path: str):
    """Load fitted preprocessing pipeline.

    Args:
        pipeline_path: Path to saved pipeline

    Returns:
        Loaded preprocessing pipeline
    """
    with open(pipeline_path, "rb") as f:
        return pickle.load(f)


def preprocess_data(data_path: str, pipeline_path: str) -> pd.DataFrame:
    """Preprocess data using the fitted pipeline.

    Args:
        data_path: Path to raw data file
        pipeline_path: Path to saved preprocessing pipeline

    Returns:
        Preprocessed DataFrame
    """
    # Load data
    df = pd.read_csv(data_path)

    # Load pipeline
    pipeline = load_preprocessing_pipeline(pipeline_path)

    # Transform data
    df_processed = pipeline.transform(df)

    return df_processed


if __name__ == "__main__":
    # Example usage
    data_path = "data/raw/data.csv"
    pipeline_path = "models/preprocessing.pkl"
    processed_data = preprocess_data(data_path, pipeline_path)
    print(f"Processed data shape: {processed_data.shape}")
'''

        # Add strategy details as comments
        strategy_comment = f"""
# Preprocessing Strategy: {strategy.get('name', 'unknown')}
# Steps:
"""
        for step in preprocessing_steps:
            strategy_comment += f"# - {step.get('step_type')}: {step.get('method')}\n"

        code = strategy_comment + code

        # Write file
        output_path = self.project_dir / "src" / "preprocessing.py"
        output_path.write_text(code, encoding="utf-8")

        # Copy preprocessing pipeline if provided
        if preprocessing_path:
            src = Path(preprocessing_path).resolve()
            if src.exists():
                (self.project_dir / "models").mkdir(parents=True, exist_ok=True)
                dest_path = self.project_dir / "models" / "preprocessing.pkl"
                shutil.copy(src, dest_path)

    def generate_train(
        self,
        winner: Dict[str, Any],
        model_path: Optional[str] = None
    ):
        """Generate train.py from winner configuration.

        Args:
            winner: Winner dictionary with model and strategy info
            model_path: Optional path to saved model
        """
        model_name = winner.get("model_name", "xgboost")
        strategy_name = winner.get("strategy_name", "unknown")

        code = f'''"""Model training script.

This module trains the {model_name} model using the {strategy_name} preprocessing strategy.
"""

import pandas as pd
import pickle
from pathlib import Path
from src.preprocessing import load_preprocessing_pipeline


def load_model(model_path: str):
    """Load trained model.

    Args:
        model_path: Path to saved model

    Returns:
        Loaded model
    """
    with open(model_path, "rb") as f:
        return pickle.load(f)


def train_model(data_path: str, target_variable: str):
    """Train the model.

    Args:
        data_path: Path to training data
        target_variable: Name of target variable

    Returns:
        Trained model
    """
    # Load data
    df = pd.read_csv(data_path)

    # Split features and target
    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    # Load preprocessing pipeline
    pipeline_path = "models/preprocessing.pkl"
    pipeline = load_preprocessing_pipeline(pipeline_path)

    # Preprocess data
    X_processed = pipeline.transform(X)

    # Load model (already trained)
    model_path = "models/model.pkl"
    model = load_model(model_path)

    return model


if __name__ == "__main__":
    # Example usage
    data_path = "data/raw/train.csv"
    target_variable = "target"  # Update with actual target variable
    model = train_model(data_path, target_variable)
    print("Model loaded successfully")
'''

        # Write file
        output_path = self.project_dir / "src" / "train.py"
        output_path.write_text(code, encoding="utf-8")

        # Copy model if provided
        if model_path:
            src = Path(model_path).resolve()
            if src.exists():
                (self.project_dir / "models").mkdir(parents=True, exist_ok=True)
                dest_path = self.project_dir / "models" / "model.pkl"
                shutil.copy(src, dest_path)

    def generate_predict(
        self,
        winner: Dict[str, Any],
        intent: Dict[str, Any]
    ):
        """Generate predict.py for inference.

        Args:
            winner: Winner dictionary
            intent: Parsed intent with target variable
        """
        model_name = winner.get("model_name", "xgboost")
        threshold = winner.get("threshold", 0.5)
        target_variable = intent.get("target_variable", "target")
        task_type = intent.get("task_type", "binary_classification")

        code = f'''"""Prediction script for new data.

This module loads the trained model and makes predictions on new data.
"""

import pandas as pd
import pickle
from pathlib import Path
from src.preprocessing import load_preprocessing_pipeline


def load_model(model_path: str = "models/model.pkl"):
    """Load trained model.

    Args:
        model_path: Path to saved model

    Returns:
        Loaded model
    """
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict(data_path: str, threshold: float = {threshold}) -> pd.DataFrame:
    """Make predictions on new data.

    Args:
        data_path: Path to new data file
        threshold: Decision threshold (for classification)

    Returns:
        DataFrame with predictions
    """
    # Load data
    df = pd.read_csv(data_path)

    # Load preprocessing pipeline
    pipeline_path = "models/preprocessing.pkl"
    pipeline = load_preprocessing_pipeline(pipeline_path)

    # Preprocess data
    df_processed = pipeline.transform(df)

    # Load model
    model = load_model()

    # Make predictions
    if "{task_type}" in ["binary_classification", "multiclass_classification"]:
        # Get probabilities
        proba = model.predict_proba(df_processed)
        if proba.shape[1] == 2:
            # Binary classification
            proba_positive = proba[:, 1]
            predictions = (proba_positive >= threshold).astype(int)
        else:
            # Multiclass: use max probability
            predictions = proba.argmax(axis=1)
    else:
        # Regression
        predictions = model.predict(df_processed)

    # Add predictions to dataframe
    df["prediction"] = predictions

    return df


if __name__ == "__main__":
    # Example usage
    new_data_path = "data/raw/new_data.csv"
    results = predict(new_data_path)
    results.to_csv("data/processed/predictions.csv", index=False)
    print(f"Predictions saved. Shape: {{results.shape}}")
'''

        # Write file
        output_path = self.project_dir / "src" / "predict.py"
        output_path.write_text(code, encoding="utf-8")

    def generate_requirements(self):
        """Generate requirements.txt file."""
        requirements = """pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.4.0
xgboost>=2.0.0
lightgbm>=4.3.0
"""

        output_path = self.project_dir / "requirements.txt"
        output_path.write_text(requirements, encoding="utf-8")
