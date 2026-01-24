"""Business metrics calculation from confusion matrix."""

from typing import Dict, Any, Optional


class BusinessMetricCalculator:
    """Calculate business impact metrics from confusion matrix."""

    def __init__(self, intent: Optional[Dict[str, Any]] = None):
        """Initialize calculator.

        Args:
            intent: Parsed intent with business context
        """
        self.intent = intent or {}
        self.business_context = self.intent.get("business_context", {})

        self.true_positive_value = self.business_context.get("true_positive_value", 0.0)
        self.false_positive_cost = self.business_context.get("false_positive_cost", 0.0)
        self.cost_ratio = self.business_context.get("cost_ratio", 0.0)

    def calculate(self, confusion_matrix: Dict[str, int]) -> Dict[str, float]:
        """Calculate business metrics from confusion matrix.

        Args:
            confusion_matrix: Dictionary with TP, TN, FP, FN

        Returns:
            Dictionary of business metrics
        """
        tp = confusion_matrix.get("true_positive", 0)
        fp = confusion_matrix.get("false_positive", 0)
        tn = confusion_matrix.get("true_negative", 0)
        fn = confusion_matrix.get("false_negative", 0)

        # Calculate business value
        total_value = tp * self.true_positive_value
        total_cost = (tp + fp) * self.false_positive_cost
        net_value = total_value - total_cost

        # Calculate ROI
        if total_cost > 0:
            roi = net_value / total_cost
        else:
            roi = float("inf") if net_value > 0 else 0.0

        return {
            "net_value": float(net_value),
            "roi": float(roi),
            "total_cost": float(total_cost),
            "total_value": float(total_value),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        }
