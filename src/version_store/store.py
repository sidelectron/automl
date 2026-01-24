"""Experiment storage using SQLite database."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from uuid import uuid4


class VersionStore:
    """Store and manage experiment versions."""

    def __init__(self, db_path: str = "data/experiments/version_store.db"):
        """Initialize the version store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                intent_json TEXT NOT NULL,
                dataset_path TEXT,
                description TEXT
            )
        """)

        # Strategies table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                strategy_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                strategy_config TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)

        # Results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                result_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                threshold REAL NOT NULL,
                metrics_json TEXT NOT NULL,
                business_metrics_json TEXT,
                model_path TEXT,
                preprocessing_path TEXT,
                FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
            )
        """)

        # Comparison results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS comparisons (
                comparison_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                comparison_json TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)

        conn.commit()
        conn.close()

    def save_experiment(
        self,
        intent_json: Dict[str, Any],
        dataset_path: Optional[str] = None,
        description: Optional[str] = None
    ) -> str:
        """Save a new experiment.

        Args:
            intent_json: Parsed intent as dictionary
            dataset_path: Path to dataset file
            description: Experiment description

        Returns:
            Experiment ID
        """
        experiment_id = str(uuid4())
        timestamp = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO experiments (experiment_id, timestamp, intent_json, dataset_path, description)
            VALUES (?, ?, ?, ?, ?)
        """, (
            experiment_id,
            timestamp,
            json.dumps(intent_json),
            dataset_path,
            description
        ))

        conn.commit()
        conn.close()

        return experiment_id

    def save_strategy(
        self,
        experiment_id: str,
        strategy_name: str,
        strategy_config: Dict[str, Any]
    ) -> str:
        """Save a strategy for an experiment.

        Args:
            experiment_id: Experiment ID
            strategy_name: Name of the strategy
            strategy_config: Strategy configuration as dictionary

        Returns:
            Strategy ID
        """
        strategy_id = str(uuid4())

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO strategies (strategy_id, experiment_id, strategy_name, strategy_config)
            VALUES (?, ?, ?, ?)
        """, (
            strategy_id,
            experiment_id,
            strategy_name,
            json.dumps(strategy_config)
        ))

        conn.commit()
        conn.close()

        return strategy_id

    def save_result(
        self,
        strategy_id: str,
        model_name: str,
        threshold: float,
        metrics: Dict[str, Any],
        business_metrics: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        preprocessing_path: Optional[str] = None
    ) -> str:
        """Save a training result.

        Args:
            strategy_id: Strategy ID
            model_name: Name of the model
            threshold: Decision threshold used
            metrics: Standard ML metrics
            business_metrics: Business impact metrics
            model_path: Path to saved model
            preprocessing_path: Path to saved preprocessing pipeline

        Returns:
            Result ID
        """
        result_id = str(uuid4())

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO results (
                result_id, strategy_id, model_name, threshold,
                metrics_json, business_metrics_json, model_path, preprocessing_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result_id,
            strategy_id,
            model_name,
            threshold,
            json.dumps(metrics),
            json.dumps(business_metrics) if business_metrics else None,
            model_path,
            preprocessing_path
        ))

        conn.commit()
        conn.close()

        return result_id

    def save_comparison(
        self,
        experiment_id: str,
        comparison_json: Dict[str, Any]
    ) -> str:
        """Save comparison results.

        Args:
            experiment_id: Experiment ID
            comparison_json: Comparison results as dictionary

        Returns:
            Comparison ID
        """
        comparison_id = str(uuid4())

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO comparisons (comparison_id, experiment_id, comparison_json)
            VALUES (?, ?, ?)
        """, (
            comparison_id,
            experiment_id,
            json.dumps(comparison_json)
        ))

        conn.commit()
        conn.close()

        return comparison_id

    def save_generated_code(
        self,
        experiment_id: str,
        code_type: str,
        code_content: str,
        execution_result: Optional[str] = None
    ) -> str:
        """Save generated code for an experiment.

        Args:
            experiment_id: Experiment ID
            code_type: Type of code ('preprocessing', 'training', 'prediction')
            code_content: Generated code content
            execution_result: Optional execution result

        Returns:
            Code ID
        """
        from uuid import uuid4
        code_id = str(uuid4())

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS generated_code (
                code_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                code_type TEXT NOT NULL,
                code_content TEXT NOT NULL,
                execution_result TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)

        timestamp = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO generated_code (
                code_id, experiment_id, code_type, code_content, execution_result, timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            code_id,
            experiment_id,
            code_type,
            code_content,
            execution_result,
            timestamp
        ))

        conn.commit()
        conn.close()

        return code_id

    def get_generated_code(
        self,
        experiment_id: str,
        code_type: str
    ) -> Optional[Dict[str, Any]]:
        """Get generated code for an experiment.

        Args:
            experiment_id: Experiment ID
            code_type: Type of code ('preprocessing', 'training', 'prediction')

        Returns:
            Code dictionary or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM generated_code
            WHERE experiment_id = ? AND code_type = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (experiment_id, code_type))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "code_id": row["code_id"],
                "experiment_id": row["experiment_id"],
                "code_type": row["code_type"],
                "code_content": row["code_content"],
                "execution_result": row["execution_result"],
                "timestamp": row["timestamp"]
            }
        return None

    def save_execution(
        self,
        code_id: str,
        attempt_number: int,
        success: bool,
        execution_log: str,
        error_type: Optional[str] = None
    ) -> str:
        """Save code execution record.

        Args:
            code_id: Code ID
            attempt_number: Attempt number (1-based)
            success: Whether execution succeeded
            execution_log: Execution log output
            error_type: Optional error type

        Returns:
            Execution ID
        """
        from uuid import uuid4
        execution_id = str(uuid4())

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS code_executions (
                execution_id TEXT PRIMARY KEY,
                code_id TEXT NOT NULL,
                attempt_number INTEGER NOT NULL,
                success BOOLEAN NOT NULL,
                execution_log TEXT,
                error_type TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (code_id) REFERENCES generated_code(code_id)
            )
        """)

        timestamp = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO code_executions (
                execution_id, code_id, attempt_number, success, execution_log, error_type, timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            execution_id,
            code_id,
            attempt_number,
            success,
            execution_log,
            error_type,
            timestamp
        ))

        conn.commit()
        conn.close()

        return execution_id

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment by ID.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment data or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM experiments WHERE experiment_id = ?
        """, (experiment_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "experiment_id": row["experiment_id"],
                "timestamp": row["timestamp"],
                "intent_json": json.loads(row["intent_json"]),
                "dataset_path": row["dataset_path"],
                "description": row["description"]
            }
        return None

    def list_experiments(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all experiments.

        Args:
            limit: Maximum number of experiments to return

        Returns:
            List of experiment summaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT experiment_id, timestamp, description
            FROM experiments
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "experiment_id": row["experiment_id"],
                "timestamp": row["timestamp"],
                "description": row["description"]
            }
            for row in rows
        ]
