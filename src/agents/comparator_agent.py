"""Comparator Agent for comparing results and explaining in business terms."""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..llm.llm_interface import LLMInterface
from ..contracts.models import ComparisonResultSchema
from ..ml.evaluation import ThresholdOptimizer


class ComparatorAgent:
    """Agent for comparing results and generating business explanations."""

    def __init__(self, llm_provider: LLMInterface):
        """Initialize Comparator Agent.

        Args:
            llm_provider: LLM provider instance
        """
        self.llm = llm_provider
        self.agent_name = "comparator"

    def _load_prompt_template(self) -> str:
        """Load the comparison prompt template.

        Returns:
            Prompt template string
        """
        prompt_path = Path(__file__).parent.parent / "llm" / "prompts" / "comparison.txt"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return "Compare model results and explain winner in business terms."

    def compare(
        self,
        training_results: List[Dict[str, Any]],
        intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare results and select winner.

        Args:
            training_results: List of training results from Trainer Agent
            intent: Parsed intent for business context

        Returns:
            Comparison results dictionary matching ComparisonResultSchema
        """
        if not training_results:
            raise ValueError("No training results provided for comparison")

        # Find winner based on business metric
        priority_metric = intent.get("business_context", {}).get("priority_metric", "net_value")

        # Filter results with business metrics
        results_with_business = [
            r for r in training_results
            if r.get("business_metrics") and priority_metric in r["business_metrics"]
        ]

        if results_with_business:
            # Select winner by business metric
            winner = max(
                results_with_business,
                key=lambda x: x["business_metrics"][priority_metric]
            )
        else:
            # Fall back to standard metrics
            if priority_metric in ["recall", "precision", "f1", "accuracy"]:
                results_with_metric = [
                    r for r in training_results
                    if r.get("metrics") and priority_metric in r["metrics"]
                ]
                if results_with_metric:
                    winner = max(
                        results_with_metric,
                        key=lambda x: x["metrics"][priority_metric]
                    )
                else:
                    winner = training_results[0]
            else:
                winner = training_results[0]

        # Get top alternatives (excluding winner)
        alternatives = [
            r for r in training_results
            if r != winner
        ]
        alternatives.sort(
            key=lambda x: (
                x.get("business_metrics", {}).get(priority_metric, 0)
                if x.get("business_metrics")
                else x.get("metrics", {}).get(priority_metric, 0)
            ),
            reverse=True
        )
        alternatives = alternatives[:3]

        # Calculate business impact
        confusion_matrix = winner.get("metrics", {}).get("confusion_matrix", {})
        business_metrics = winner.get("business_metrics", {})

        business_impact = {
            "true_positives": confusion_matrix.get("true_positive", 0),
            "false_positives": confusion_matrix.get("false_positive", 0),
            "false_negatives": confusion_matrix.get("false_negative", 0),
            "total_actions": (
                confusion_matrix.get("true_positive", 0) +
                confusion_matrix.get("false_positive", 0)
            ),
            "explanation": ""
        }

        financial_impact = {
            "potential_value": business_metrics.get("total_value", 0.0),
            "total_cost": business_metrics.get("total_cost", 0.0),
            "net_value": business_metrics.get("net_value", 0.0),
            "roi": business_metrics.get("roi", 0.0)
        }

        # Generate business explanation using LLM
        system_prompt = self._load_prompt_template()

        user_prompt = f"""Compare the following model results and explain the winner in business terms.

# Training Results #
```json
{json.dumps(training_results, indent=2)}
```

# User Intent #
```json
{json.dumps(intent, indent=2)}
```

# Winner #
```json
{json.dumps(winner, indent=2)}
```

Generate a business explanation that:
1. Explains why this winner is best (in business terms, not technical)
2. Describes the business impact (e.g., "Catches 1,540 of 1,900 churners")
3. Describes the financial impact (e.g., "Net value: $723,500, ROI: 16.6x")
4. Recommends the optimal threshold
5. Compares with alternatives
"""

        try:
            llm_response = self.llm.generate_json(
                prompt=system_prompt + "\n\n" + user_prompt,
                temperature=0.3,
                agent=self.agent_name,
                timeout=180
            )

            # Merge LLM response with calculated values
            comparison = {
                "winner": {
                    "strategy_name": winner.get("strategy_name"),
                    "model_name": winner.get("model_name"),
                    "threshold": winner.get("threshold"),
                    "metrics": winner.get("metrics", {}),
                    "business_metrics": winner.get("business_metrics", {}),
                    "model_path": winner.get("model_path"),
                    "preprocessing_path": winner.get("preprocessing_path")
                },
                "business_impact": llm_response.get("business_impact", business_impact),
                "financial_impact": llm_response.get("financial_impact", financial_impact),
                "recommended_threshold": winner.get("threshold", 0.5),
                "alternatives": [
                    {
                        "strategy_name": alt.get("strategy_name"),
                        "model_name": alt.get("model_name"),
                        "threshold": alt.get("threshold"),
                        "business_metrics": alt.get("business_metrics", {})
                    }
                    for alt in alternatives
                ],
                "comparison_text": llm_response.get("comparison_text", "Comparison complete.")
            }

            # Update business_impact with calculated values
            if "explanation" not in comparison["business_impact"] or not comparison["business_impact"]["explanation"]:
                comparison["business_impact"]["explanation"] = comparison["comparison_text"]

        except Exception as e:
            # Fallback: generate basic comparison
            comparison = {
                "winner": {
                    "strategy_name": winner.get("strategy_name"),
                    "model_name": winner.get("model_name"),
                    "threshold": winner.get("threshold"),
                    "metrics": winner.get("metrics", {}),
                    "business_metrics": winner.get("business_metrics", {}),
                    "model_path": winner.get("model_path"),
                    "preprocessing_path": winner.get("preprocessing_path")
                },
                "business_impact": business_impact,
                "financial_impact": financial_impact,
                "recommended_threshold": winner.get("threshold", 0.5),
                "alternatives": [
                    {
                        "strategy_name": alt.get("strategy_name"),
                        "model_name": alt.get("model_name"),
                        "threshold": alt.get("threshold"),
                        "business_metrics": alt.get("business_metrics", {})
                    }
                    for alt in alternatives
                ],
                "comparison_text": f"Winner: {winner.get('model_name')} with threshold {winner.get('threshold')}. "
                                 f"Net value: ${business_metrics.get('net_value', 0):,.0f}, "
                                 f"ROI: {business_metrics.get('roi', 0):.1f}x"
            }

        # Validate against schema
        try:
            validated = ComparisonResultSchema(**comparison)
            return validated.model_dump()
        except Exception as e:
            # Return unvalidated if validation fails
            return comparison
