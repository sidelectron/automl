"""Model Agent v2 for generating text-based modeling plans."""

import json
from typing import Dict, Any, Optional
from pathlib import Path

from ..llm.llm_interface import LLMInterface


class ModelAgentV2:
    """Agent for generating text-based modeling plans (not JSON)."""

    def __init__(self, llm_provider: LLMInterface):
        """Initialize Model Agent v2.

        Args:
            llm_provider: LLM provider instance
        """
        self.llm = llm_provider
        self.agent_name = "model_v2"

    def _load_prompt_template(self) -> str:
        """Load prompt template for text-based plan generation.

        Returns:
            Prompt template string
        """
        return """You are an experienced machine learning research engineer in the AutoML development team.
Your task is to generate detailed text-based modeling plans that can be used by a code generation agent.

# Your Responsibilities #
Generate a comprehensive modeling plan as detailed text instructions (not JSON).
The plan should specify models, hyperparameters, threshold tuning, and business metrics calculation.

# Available Models #

## Ensemble Models (High Performance)
- XGBoost: Gradient boosting, good for most tasks
  - Key params: n_estimators, max_depth, learning_rate, scale_pos_weight
- LightGBM: Fast gradient boosting, good for large datasets
  - Key params: n_estimators, num_leaves, learning_rate, class_weight
- Random Forest: Robust ensemble, good for feature importance
  - Key params: n_estimators, max_depth, class_weight

## Linear Models (Interpretable)
- Logistic Regression: Classification baseline, highly interpretable
  - Key params: C, penalty ('l1', 'l2', 'elasticnet'), class_weight
- Linear Regression: Regression baseline
  - Variants: Ridge (L2), Lasso (L1), ElasticNet

## Tree-Based Models
- Decision Tree: Simple, highly interpretable, prone to overfitting
  - Key params: max_depth, min_samples_split, min_samples_leaf

## Instance-Based Models
- KNN (K-Nearest Neighbors): Distance-based, good for small datasets
  - Key params: n_neighbors, weights ('uniform', 'distance'), metric

## Kernel-Based Models
- SVM (Support Vector Machine): Good for high-dimensional data
  - Key params: C, kernel ('linear', 'rbf', 'poly'), gamma

# Model Selection Guidelines #
- Start with ensemble models (XGBoost, LightGBM) for best performance
- Include Linear/Logistic Regression as interpretable baseline
- Use Decision Tree when interpretability is critical
- Use KNN for small datasets with clear cluster structure
- Use SVM for high-dimensional sparse data

# Plan Format #
Your plan should be detailed text instructions that specify:
- Which models to train (select 3-5 appropriate models)
- Hyperparameters for each model
- Threshold tuning strategy (for classification)
- Business metrics calculation
- Model selection criteria

Do NOT output JSON. Output plain text instructions.
"""

    def generate_modeling_plan(
        self,
        data_plan: str,
        intent: Dict[str, Any]
    ) -> str:
        """Generate text-based modeling plan.

        Args:
            data_plan: Preprocessing plan from StrategyAgentV2
            intent: Parsed intent from Intent Agent

        Returns:
            Text-based modeling plan
        """
        system_prompt = self._load_prompt_template()

        task_type = intent.get("task_type", "binary_classification")
        business_context = intent.get("business_context", {})
        priority_metric = business_context.get("priority_metric", "f1")

        user_prompt = f"""Generate a detailed modeling plan based on the following context.

# Preprocessing Plan #
{data_plan}

# User Intent #
- Task type: {task_type}
- Priority metric: {priority_metric}
- Business context: {json.dumps(business_context, indent=2)}

Generate a comprehensive text-based modeling plan that includes:
1. Which models to train (XGBoost, LightGBM, Random Forest, Logistic Regression)
2. Hyperparameters for each model (specific values, not ranges)
3. Threshold tuning strategy (evaluate at [0.3, 0.4, 0.5, 0.6, 0.7] for classification)
4. Business metrics calculation (net_value, ROI based on true_positive_value and false_positive_cost)
5. Model selection criteria (based on priority_metric or business metric)
6. Class imbalance handling (scale_pos_weight or class_weight if priority_metric is 'recall')

The plan should be detailed enough for a code generation agent to write complete Python training code.
Output only the plan text, no JSON, no code blocks.
"""

        try:
            response = self.llm.generate(
                prompt=system_prompt + "\n\n" + user_prompt,
                temperature=0.3,
                max_tokens=2048,
                agent=self.agent_name,
                timeout=180
            )

            return response.strip()

        except Exception as e:
            # Fallback: generate basic plan
            return self._generate_fallback_plan(intent)

    def _generate_fallback_plan(
        self,
        intent: Dict[str, Any]
    ) -> str:
        """Generate basic fallback plan.

        Args:
            intent: Parsed intent

        Returns:
            Basic text plan
        """
        task_type = intent.get("task_type", "binary_classification")
        priority_metric = intent.get("business_context", {}).get("priority_metric", "f1")

        plan = "Modeling Plan:\n"
        plan += "1. Train XGBoost with n_estimators=200, max_depth=5, learning_rate=0.1\n"
        plan += "2. Train LightGBM with n_estimators=200, num_leaves=50, learning_rate=0.1\n"
        plan += "3. Train Random Forest with n_estimators=500, max_depth=10\n"

        if task_type in ["binary_classification", "multiclass_classification"]:
            if priority_metric == "recall":
                plan += "4. For each model, use scale_pos_weight=9 to handle class imbalance\n"
            plan += "5. Evaluate each model at thresholds [0.3, 0.4, 0.5, 0.6, 0.7]\n"
            plan += "6. Calculate business metrics (net_value, ROI) for each threshold\n"
        else:
            plan += "4. For regression, evaluate using RMSE and R2 score\n"

        plan += "7. Select best model-threshold combination based on priority_metric or net_value\n"

        return plan
