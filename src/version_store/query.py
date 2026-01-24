"""Query utilities for version store."""

import json
from typing import Optional, Dict, Any, List
from .store import VersionStore


class QueryManager:
    """Query manager for experiment data."""

    def __init__(self, version_store: VersionStore):
        """Initialize query manager.

        Args:
            version_store: VersionStore instance
        """
        self.store = version_store

    def get_experiment_results(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get all results for an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            List of results with strategy and model information
        """
        import sqlite3

        conn = sqlite3.connect(self.store.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT r.*, s.strategy_name, s.strategy_config
            FROM results r
            JOIN strategies s ON r.strategy_id = s.strategy_id
            WHERE s.experiment_id = ?
            ORDER BY r.threshold, s.strategy_name, r.model_name
        """, (experiment_id,))

        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            results.append({
                "result_id": row["result_id"],
                "strategy_name": row["strategy_name"],
                "model_name": row["model_name"],
                "threshold": row["threshold"],
                "metrics": json.loads(row["metrics_json"]),
                "business_metrics": json.loads(row["business_metrics_json"]) if row["business_metrics_json"] else None,
                "model_path": row["model_path"],
                "preprocessing_path": row["preprocessing_path"]
            })

        return results

    def get_best_result(
        self,
        experiment_id: str,
        metric: str = "net_value"
    ) -> Optional[Dict[str, Any]]:
        """Get best result by business metric.

        Args:
            experiment_id: Experiment ID
            metric: Metric to optimize for (net_value, roi, f1, etc.)

        Returns:
            Best result or None if no results found
        """
        results = self.get_experiment_results(experiment_id)

        if not results:
            return None

        # Filter results with business metrics
        results_with_business = [
            r for r in results
            if r["business_metrics"] and metric in r["business_metrics"]
        ]

        if not results_with_business:
            # Fall back to standard metrics
            if metric in ["f1", "precision", "recall", "accuracy"]:
                results_with_metric = [
                    r for r in results
                    if r["metrics"] and metric in r["metrics"]
                ]
                if results_with_metric:
                    return max(
                        results_with_metric,
                        key=lambda x: x["metrics"][metric]
                    )
            return None

        return max(
            results_with_business,
            key=lambda x: x["business_metrics"][metric]
        )

    def get_comparison(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get comparison results for an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Comparison results or None if not found
        """
        import sqlite3

        conn = sqlite3.connect(self.store.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT comparison_json FROM comparisons
            WHERE experiment_id = ?
            ORDER BY comparison_id DESC
            LIMIT 1
        """, (experiment_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return json.loads(row["comparison_json"])
        return None

    def compare_experiments(
        self,
        experiment_ids: List[str]
    ) -> Dict[str, Any]:
        """Compare multiple experiments.

        Args:
            experiment_ids: List of experiment IDs to compare

        Returns:
            Comparison summary
        """
        comparisons = []
        for exp_id in experiment_ids:
            comparison = self.get_comparison(exp_id)
            if comparison:
                comparisons.append({
                    "experiment_id": exp_id,
                    "comparison": comparison
                })

        return {
            "experiments": comparisons,
            "count": len(comparisons)
        }
