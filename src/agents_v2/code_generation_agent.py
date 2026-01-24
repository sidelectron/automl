"""Code generation agent for creating executable Python code."""

import re
import json
from typing import Dict, Any, Optional
from pathlib import Path

from ..llm.llm_interface import LLMInterface


class CodeGenerationAgent:
    """Agent for generating executable Python code from text instructions."""

    def __init__(self, llm_provider: LLMInterface):
        """Initialize code generation agent.

        Args:
            llm_provider: LLM provider instance
        """
        self.llm = llm_provider
        self.agent_name = "code_generation"

    def _load_prompt_template(self) -> str:
        """Load code generation prompt template.

        Returns:
            Prompt template string
        """
        prompt_path = Path(__file__).parent.parent / "llm" / "prompts" / "code_generation.txt"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return "Generate complete, executable Python code for machine learning pipelines."

    def generate_preprocessing_code(
        self,
        strategy_instructions: str,
        data_profile: Dict[str, Any],
        intent: Dict[str, Any]
    ) -> str:
        """Generate preprocessing Python code.

        Args:
            strategy_instructions: Text instructions for preprocessing
            data_profile: Data profile dictionary
            intent: Parsed intent dictionary

        Returns:
            Generated Python code as string
        """
        system_prompt = self._load_prompt_template()

        # Format data profile
        numeric_cols = [
            col for col, dtype in data_profile.get("data_types", {}).items()
            if dtype == "numeric"
        ]
        categorical_cols = [
            col for col, dtype in data_profile.get("data_types", {}).items()
            if dtype == "categorical"
        ]
        missing_values = data_profile.get("missing_values", {})
        dataset_path = intent.get("dataset_path", "")

        user_prompt = f"""Generate a complete preprocessing pipeline in Python using sklearn.

# Data Profile
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}
- Missing values: {json.dumps(missing_values, indent=2)}
- Class distribution: {data_profile.get('class_distribution', {})}

# User Intent
- Task type: {intent.get('task_type', 'binary_classification')}
- Target variable: {intent.get('target_variable', 'target')}
- Priority metric: {intent.get('business_context', {}).get('priority_metric', 'f1')}
- Business context: {json.dumps(intent.get('business_context', {}), indent=2)}
"""
        if dataset_path:
            user_prompt += f"""
# Dataset
- CSV path (use exactly this when loading data): {dataset_path}
- Load the CSV from the path above. Do not use a different path.
"""
        user_prompt += f"""
# Instructions
{strategy_instructions}

Generate complete, executable Python code that:
1. Handles missing values appropriately:
   - Use SimpleImputer(strategy='median', add_indicator=True) for numeric columns
   - Use SimpleImputer(strategy='most_frequent') for categorical columns
   - Consider IterativeImputer for multivariate imputation when features are correlated
2. Handles outliers if detected (use IQR or Z-score capping):
   - IQR method: clip values to [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
3. Encodes categorical variables:
   - OneHotEncoder for low-cardinality categoricals
   - TargetEncoder (mean encoding) for high-cardinality categoricals
4. Scales numeric features (use StandardScaler, MinMaxScaler, or RobustScaler)
5. Applies PCA if many correlated features exist
6. Handles class imbalance if needed (use SMOTE when priority_metric is 'recall')
7. Uses sklearn Pipeline and ColumnTransformer
8. Includes proper error handling
9. Saves the fitted pipeline using pickle

CRITICAL - Data Leakage Prevention:
- The pipeline must have separate fit() and transform() methods
- NEVER call fit_transform() on the full dataset
- The pipeline will be fit ONLY on training data after train-test split
- Validation/test data must only be transformed (not fit)

The code should be a complete Python module with functions that can be imported and used.
"""

        try:
            response = self.llm.generate(
                prompt=system_prompt + "\n\n" + user_prompt,
                temperature=0.2,  # Lower temperature for more deterministic code
                max_tokens=4096,
                agent=self.agent_name,
                timeout=180
            )

            # Extract code from markdown blocks
            code = self._extract_code_from_response(response)
            return code

        except Exception as e:
            raise Exception(f"Failed to generate preprocessing code: {e}") from e

    def generate_training_code(
        self,
        preprocessing_code: str,
        model_instructions: str,
        intent: Dict[str, Any]
    ) -> str:
        """Generate training Python code.

        Args:
            preprocessing_code: Generated preprocessing code
            model_instructions: Text instructions for modeling
            intent: Parsed intent dictionary

        Returns:
            Generated Python code as string
        """
        system_prompt = self._load_prompt_template()

        task_type = intent.get("task_type", "binary_classification")
        target_variable = intent.get("target_variable", "target")
        business_context = intent.get("business_context", {})
        dataset_path = intent.get("dataset_path", "")

        user_prompt = f"""Generate a complete training script in Python.

# Task Information
- Task type: {task_type}
- Target variable: {target_variable}
- Priority metric: {business_context.get('priority_metric', 'f1')}
- Business context: {json.dumps(business_context, indent=2)}
"""
        if dataset_path:
            user_prompt += f"""
# Dataset
- CSV path (use exactly this): {dataset_path}
- Load the CSV from the path above. Do not use a different path.
"""
        user_prompt += f"""
# Preprocessing Code
```python
{preprocessing_code}
```

# Modeling Instructions
{model_instructions}

Generate complete, executable Python code that:
1. Loads data from CSV using the dataset path above
2. FIRST splits data into train/validation sets (with stratification for classification)
3. THEN fits preprocessing pipeline ONLY on training data
4. Transforms both train and validation sets (validation is transform only, NOT fit)
5. Trains multiple models on preprocessed training data. Available models:
   - Ensemble: XGBoost, LightGBM, Random Forest
   - Linear: Logistic Regression, Linear Regression (Ridge, Lasso, ElasticNet)
   - Tree-based: Decision Tree
   - Instance-based: KNN (K-Nearest Neighbors)
   - Kernel-based: SVM (Support Vector Machine)
6. For classification: Evaluates at thresholds [0.3, 0.4, 0.5, 0.6, 0.7]
7. Calculates business metrics (net_value, ROI) for each threshold
8. Saves trained models and results
9. Handles class imbalance if priority_metric is 'recall'
10. CRITICAL - Writes results to 'results.json' in the current directory (required for comparison).
    You MUST write this file. Valid JSON: a list of objects. Each object MUST have:
    - model_name (str), strategy_name (str), threshold (float or null)
    - metrics: accuracy, f1, recall, precision, confusion_matrix (true_positive, true_negative, false_positive, false_negative)
    - business_metrics: net_value, roi, total_value, total_cost
    Example: [{{"model_name":"xgboost","strategy_name":"basic","threshold":0.5,"metrics":{{...}},"business_metrics":{{...}}}}]
    Use: with open("results.json","w") as f: json.dump(results_list, f, indent=2)

CRITICAL - Data Leakage Prevention:
- Split BEFORE any preprocessing to avoid data leakage
- Fit preprocessing (scaler, encoder, imputer) ONLY on X_train
- Use transform() (not fit_transform()) on X_validation
- Example correct order:
  X_train, X_val, y_train, y_val = train_test_split(X, y, ...)
  X_train_processed = pipeline.fit_transform(X_train)
  X_val_processed = pipeline.transform(X_val)  # NO fit here!

The code should be a complete Python script that can be run directly.
"""

        try:
            response = self.llm.generate(
                prompt=system_prompt + "\n\n" + user_prompt,
                temperature=0.2,
                max_tokens=4096,
                agent=self.agent_name,
                timeout=180
            )

            code = self._extract_code_from_response(response)
            return code

        except Exception as e:
            raise Exception(f"Failed to generate training code: {e}") from e

    def generate_prediction_code(
        self,
        training_code: str,
        intent: Dict[str, Any]
    ) -> str:
        """Generate prediction Python code.

        Args:
            training_code: Generated training code
            intent: Parsed intent dictionary

        Returns:
            Generated Python code as string
        """
        system_prompt = self._load_prompt_template()

        task_type = intent.get("task_type", "binary_classification")
        target_variable = intent.get("target_variable", "target")
        threshold = intent.get("recommended_threshold", 0.5)

        user_prompt = f"""Generate a complete prediction script in Python.

# Task Information
- Task type: {task_type}
- Target variable: {target_variable}
- Recommended threshold: {threshold}

# Training Code Reference
```python
{training_code[:500]}  # First 500 chars for context
```

Generate complete, executable Python code that:
1. Loads new data from CSV
2. Loads the saved preprocessing pipeline
3. Loads the saved trained model
4. Preprocesses the new data
5. Makes predictions
6. For classification: Applies the threshold to probabilities
7. Saves predictions to CSV

The code should be a complete Python script that can be run directly.
"""

        try:
            response = self.llm.generate(
                prompt=system_prompt + "\n\n" + user_prompt,
                temperature=0.2,
                max_tokens=2048,
                agent=self.agent_name,
                timeout=180
            )

            code = self._extract_code_from_response(response)
            return code

        except Exception as e:
            raise Exception(f"Failed to generate prediction code: {e}") from e

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response.

        Args:
            response: LLM response (may contain markdown code blocks)

        Returns:
            Extracted Python code
        """
        # Try to extract from ```python blocks
        pattern = r"```python\s*(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            # Return the first (and usually only) code block
            code = matches[0].strip()
            return code

        # If no code blocks, check if entire response is code
        # (some models don't use markdown)
        if response.strip().startswith("import ") or response.strip().startswith("from "):
            return response.strip()

        # If no code found, return response as-is (caller will handle validation)
        return response.strip()
