"""Optimal threshold finder based on business metrics."""

from typing import List, Dict, Any, Optional


class ThresholdOptimizer:
    """Find optimal threshold for business goals."""

    def __init__(self, intent: Optional[Dict[str, Any]] = None):
        """Initialize optimizer.

        Args:
            intent: Parsed intent with priority_metric
        """
        self.intent = intent or {}
        self.priority_metric = self.intent.get("business_context", {}).get("priority_metric", "f1")

    def find_optimal(
        self,
        threshold_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Find optimal threshold based on priority metric.

        Args:
            threshold_results: List of results from threshold tuning

        Returns:
            Best result dictionary
        """
        if not threshold_results:
            raise ValueError("No threshold results provided")

        # Determine metric to optimize
        if self.priority_metric in ["net_value", "roi"]:
            # Optimize business metric
            valid_results = [
                r for r in threshold_results
                if r.get("business_metrics") and self.priority_metric in r["business_metrics"]
            ]
            if valid_results:
                return max(valid_results, key=lambda x: x["business_metrics"][self.priority_metric])

        # Fall back to standard metrics
        metric_map = {
            "recall": "recall",
            "precision": "precision",
            "f1": "f1",
            "accuracy": "accuracy",
        }

        target_metric = metric_map.get(self.priority_metric, "f1")

        valid_results = [
            r for r in threshold_results
            if r.get("metrics") and target_metric in r["metrics"]
        ]

        if valid_results:
            return max(valid_results, key=lambda x: x["metrics"][target_metric])

        # Last resort: return first result
        return threshold_results[0]
