"""Strategy Agent v2 for generating text-based preprocessing plans."""

import json
from typing import Dict, Any, Optional
from pathlib import Path

from ..llm.llm_interface import LLMInterface


class StrategyAgentV2:
    """Agent for generating text-based preprocessing plans (not JSON)."""

    def __init__(self, llm_provider: LLMInterface):
        """Initialize Strategy Agent v2.

        Args:
            llm_provider: LLM provider instance
        """
        self.llm = llm_provider
        self.agent_name = "strategy_v2"

    def _load_prompt_template(self) -> str:
        """Load prompt template for text-based plan generation.

        Returns:
            Prompt template string
        """
        # Use a similar template but request text instructions instead of JSON
        return """You are an experienced data scientist in the AutoML development team.
Your task is to generate detailed text-based preprocessing plans that can be used by a code generation agent.

# Your Responsibilities #
Generate a comprehensive preprocessing plan as detailed text instructions (not JSON).
The plan should be self-contained and enable another agent to write complete Python code.

# Preprocessing Techniques #
1. Imputation: Handle missing values
   - Simple: median, mean, mode (sklearn SimpleImputer)
   - Advanced: KNN imputation (sklearn KNNImputer)
   - Multivariate: Iterative/MICE imputation (sklearn IterativeImputer) - captures feature relationships

2. Outlier Handling (from ML text Pages 127-141):
   - IQR method: Cap/remove values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
   - Z-score method: Cap/remove values with |z| > 3
   - Percentile method: Cap values outside [1st, 99th] percentiles

3. Scaling: Normalize/standardize features
   - StandardScaler: Zero mean, unit variance (best for most cases)
   - MinMaxScaler: Scale to [0, 1] range
   - RobustScaler: Uses median/IQR, robust to outliers

4. Encoding: Encode categorical variables
   - OneHotEncoder: For low-cardinality categoricals
   - OrdinalEncoder/LabelEncoder: For ordinal categoricals
   - TargetEncoder: For high-cardinality categoricals (uses mean of target)

5. Feature Engineering:
   - Interaction features: Product of numeric pairs (x1 * x2)
   - Polynomial features: Squared terms (x^2)
   - Log transformations: For skewed distributions (log1p)
   - DateTime extraction: year, month, day, dayofweek, is_weekend
   - Binning: Discretize continuous variables

6. Dimensionality Reduction:
   - PCA: When many correlated features exist

7. Class Imbalance: Handle imbalanced data
   - SMOTE, undersampling, oversampling, class weights

# Plan Format #
Your plan should be detailed text instructions that specify:
- Which columns need which preprocessing
- Specific methods and parameters to use
- Order of operations (imputation -> outliers -> encoding -> scaling -> feature engineering)
- Rationale for each step
- Expected outcomes

Do NOT output JSON. Output plain text instructions.
"""

    def generate_preprocessing_plan(
        self,
        profile: Dict[str, Any],
        intent: Dict[str, Any],
        eda_results: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate text-based preprocessing plan.

        Args:
            profile: Data profile from Profiler Agent
            intent: Parsed intent from Intent Agent
            eda_results: Optional EDA results from EDA Agent

        Returns:
            Text-based preprocessing plan
        """
        system_prompt = self._load_prompt_template()

        # Build context
        context = {
            "profile": profile,
            "intent": intent
        }

        if eda_results:
            context["eda"] = eda_results

        user_prompt = f"""Generate a detailed preprocessing plan based on the following context.

# Context #
```json
{json.dumps(context, indent=2)}
```

Generate a comprehensive text-based preprocessing plan that includes:
1. Specific steps for handling missing values (which columns, which method, why)
2. Scaling strategy (which scaler, which columns, why)
3. Encoding strategy (which encoder, which columns, why)
4. Feature engineering steps (if needed)
5. Class imbalance handling (if needed, based on priority_metric)
6. Order of operations
7. Expected outcomes

The plan should be detailed enough for a code generation agent to write complete Python code.
Output only the plan text, no JSON, no code blocks.
"""

        try:
            response = self.llm.generate(
                prompt=system_prompt + "\n\n" + user_prompt,
                temperature=0.3,  # Some creativity for diverse plans
                max_tokens=2048,
                agent=self.agent_name,
                timeout=180
            )

            return response.strip()

        except Exception as e:
            # Fallback: generate basic plan
            return self._generate_fallback_plan(profile, intent)

    def _generate_fallback_plan(
        self,
        profile: Dict[str, Any],
        intent: Dict[str, Any]
    ) -> str:
        """Generate basic fallback plan.

        Args:
            profile: Data profile
            intent: Parsed intent

        Returns:
            Basic text plan
        """
        data_types = profile.get("data_types", {})
        missing_values = profile.get("missing_values", {})
        task_type = intent.get("task_type", "binary_classification")
        priority_metric = intent.get("business_context", {}).get("priority_metric", "f1")

        plan = "Preprocessing Plan:\n"
        plan += "1. Handle missing values in numeric columns using median imputation with missing indicators\n"
        plan += "2. Handle missing values in categorical columns using mode imputation\n"
        plan += "3. Apply standard scaling to all numeric features\n"
        plan += "4. Apply one-hot encoding to categorical features with handle_unknown='ignore'\n"

        if task_type in ["binary_classification", "multiclass_classification"]:
            if priority_metric == "recall":
                plan += "5. Use SMOTE to handle class imbalance (priority is recall)\n"

        plan += "6. Combine all steps using sklearn Pipeline and ColumnTransformer\n"

        return plan
