"""Execution engine for running preprocessing strategies and training models."""

from typing import Dict, Any, List, Optional
from multiprocessing import Pool
import traceback
import pandas as pd
from pathlib import Path

from ..ml.preprocessing import PreprocessingPipeline
from ..ml.models.factory import ModelFactory
from ..ml.training import ModelTrainer, ThresholdTuner
from ..ml.evaluation import ThresholdOptimizer


class ExecutionEngine:
    """Engine for executing preprocessing strategies and training models."""

    def __init__(self):
        """Initialize the execution engine."""
        # These will be set when ML modules are implemented
        self.preprocessing_module = None
        self.models_module = None
        self.training_module = None
        self.evaluation_module = None

    def execute_strategy(
        self,
        strategy: Dict[str, Any],
        dataset_path: str,
        intent: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute a single strategy.

        Args:
            strategy: Strategy dictionary from Strategy Agent
            dataset_path: Path to dataset file
            intent: Parsed intent for context

        Returns:
            List of training results (one per model-threshold combination)
        """
        # Load dataset
        df = pd.read_csv(dataset_path)
        target_variable = intent.get("target_variable")
        task_type = intent.get("task_type", "binary_classification")

        if target_variable not in df.columns:
            raise ValueError(f"Target variable '{target_variable}' not found in dataset")

        # Split features and target
        X = df.drop(columns=[target_variable])
        y = df[target_variable]

        # IMPORTANT: Split BEFORE preprocessing to avoid data leakage
        # The preprocessing pipeline must be fit ONLY on training data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y if task_type != "regression" else None
        )

        # Build and execute preprocessing pipeline using sklearn Pipeline
        # Fit on training data only, then transform both train and validation
        pipeline = PreprocessingPipeline()
        pipeline.build_from_strategy(strategy, X=X_train)
        X_train_processed = pipeline.fit_transform(X_train, y_train)
        X_val_processed = pipeline.transform(X_val)  # Transform only, no fitting!

        # Train each model candidate
        results = []
        model_candidates = strategy.get("model_candidates", [])
        strategy_name = strategy.get("name", "unknown")

        for model_name in model_candidates:
            try:
                # Create model
                model = ModelFactory.create(model_name, task_type=task_type)

                # Train model on preprocessed training data
                # Use our pre-split validation set (X_val_processed) for evaluation
                # This ensures no data leakage: preprocessing was fit ONLY on training data
                model.train(X_train_processed, y_train)

                # Tune thresholds using the held-out validation set (for classification)
                # X_val_processed was transformed (not fit) so no leakage
                if task_type in ["binary_classification", "multiclass_classification"]:
                    threshold_tuner = ThresholdTuner(intent=intent)
                    threshold_results = threshold_tuner.tune(model, X_val_processed, y_val)

                    # Find best threshold
                    optimizer = ThresholdOptimizer(intent=intent)
                    best_result = optimizer.find_optimal(threshold_results)

                    # Save model and pipeline
                    model_path = f"data/models/{strategy_name}_{model_name}_model.pkl"
                    preprocessing_path = f"data/models/{strategy_name}_preprocessing.pkl"
                    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                    model.save(model_path)
                    pipeline.save(preprocessing_path)

                    results.append({
                        "strategy_name": strategy_name,
                        "model_name": model_name,
                        "threshold": best_result.get("threshold", 0.5),
                        "metrics": best_result.get("metrics", {}),
                        "business_metrics": best_result.get("business_metrics"),
                        "model_path": model_path,
                        "preprocessing_path": preprocessing_path
                    })
                else:
                    # Regression: no threshold tuning
                    # Use held-out validation set for evaluation
                    from ..ml.evaluation.metrics import calculate_metrics
                    predictions = model.predict(X_val_processed)
                    metrics = calculate_metrics(y_val, predictions, task_type="regression")

                    model_path = f"data/models/{strategy_name}_{model_name}_model.pkl"
                    preprocessing_path = f"data/models/{strategy_name}_preprocessing.pkl"
                    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                    model.save(model_path)
                    pipeline.save(preprocessing_path)

                    results.append({
                        "strategy_name": strategy_name,
                        "model_name": model_name,
                        "threshold": None,
                        "metrics": metrics,
                        "business_metrics": None,
                        "model_path": model_path,
                        "preprocessing_path": preprocessing_path
                    })

            except Exception as e:
                print(f"Error training {model_name} for strategy {strategy_name}: {e}")
                continue

        return results

    def execute_strategies_parallel(
        self,
        strategies: List[Dict[str, Any]],
        dataset_path: str,
        intent: Dict[str, Any],
        n_workers: Optional[int] = None
    ) -> List[List[Dict[str, Any]]]:
        """Execute multiple strategies in parallel.

        Args:
            strategies: List of strategy dictionaries
            dataset_path: Path to dataset file
            intent: Parsed intent for context
            n_workers: Number of parallel workers (default: number of strategies)

        Returns:
            List of result lists (one per strategy)
        """
        if n_workers is None:
            n_workers = len(strategies)

        # Prepare arguments for execution
        args = [(strategy, dataset_path, intent) for strategy in strategies]

        # Use sequential execution when n_workers==1 to avoid Windows multiprocessing
        # spawn issues (Pool requires if __name__ == '__main__' guard in entry script)
        if n_workers == 1:
            return [self._execute_strategy_safe(*arg) for arg in args]

        # Execute in parallel
        try:
            with Pool(n_workers) as pool:
                results = pool.starmap(self._execute_strategy_safe, args)
            return results
        except Exception as e:
            # Fallback to sequential execution on error
            print(f"Parallel execution failed, falling back to sequential: {e}")
            return [self._execute_strategy_safe(*arg) for arg in args]

    def _execute_strategy_safe(
        self,
        strategy: Dict[str, Any],
        dataset_path: str,
        intent: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Safely execute a strategy with error handling.

        Args:
            strategy: Strategy dictionary
            dataset_path: Path to dataset
            intent: Parsed intent

        Returns:
            List of results (empty list on error)
        """
        try:
            return self.execute_strategy(strategy, dataset_path, intent)
        except Exception as e:
            print(f"Error executing strategy '{strategy.get('name', 'unknown')}': {e}")
            print(traceback.format_exc())
            return []

    def set_ml_modules(
        self,
        preprocessing_module=None,
        models_module=None,
        training_module=None,
        evaluation_module=None
    ):
        """Set ML module dependencies.

        Args:
            preprocessing_module: Preprocessing module instance
            models_module: Models module instance
            training_module: Training module instance
            evaluation_module: Evaluation module instance
        """
        self.preprocessing_module = preprocessing_module
        self.models_module = models_module
        self.training_module = training_module
        self.evaluation_module = evaluation_module
