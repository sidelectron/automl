"""Preprocessing pipeline builder and executor using sklearn Pipeline and ColumnTransformer."""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler as SklearnStandardScaler,
    MinMaxScaler as SklearnMinMaxScaler,
    RobustScaler as SklearnRobustScaler,
    OneHotEncoder as SklearnOneHotEncoder,
    LabelEncoder as SklearnLabelEncoder,
    OrdinalEncoder,
    PowerTransformer,
    FunctionTransformer
)
from sklearn.feature_selection import SelectKBest, chi2, f_regression

from .base import BaseTransformer
from .imputation import MedianImputer, ModeImputer, KNNImputer, IterativeImputer
from .scaling import StandardScaler, MinMaxScaler, RobustScaler
from .encoding import OneHotEncoder, LabelEncoder, TargetEncoder
from .feature_engineering import FeatureEngineer
from .outlier import IQROutlierHandler, ZScoreOutlierHandler, PercentileOutlierHandler
from .dimensionality import PCATransformer


class PreprocessingPipeline:
    """Pipeline for executing preprocessing steps from strategy JSON using sklearn Pipeline."""

    # Mapping of method names to sklearn transformers
    TRANSFORMER_MAP = {
        "imputation": {
            "mean": lambda **kwargs: SimpleImputer(strategy="mean", add_indicator=True, **kwargs),
            "median": lambda **kwargs: SimpleImputer(strategy="median", add_indicator=True, **kwargs),
            "mode": lambda **kwargs: SimpleImputer(strategy="most_frequent", add_indicator=True, **kwargs),
            "knn": lambda **kwargs: KNNImputer(**kwargs),
            # NEW: Iterative imputation (MICE) for multivariate imputation
            "iterative": lambda **kwargs: IterativeImputer(**kwargs),
            "mice": lambda **kwargs: IterativeImputer(**kwargs),
        },
        "scaling": {
            "standard_scaler": lambda **kwargs: SklearnStandardScaler(**kwargs),
            "min_max_scaler": lambda **kwargs: SklearnMinMaxScaler(**kwargs),
            "robust_scaler": lambda **kwargs: SklearnRobustScaler(**kwargs),
        },
        "encoding": {
            # From ML PDF: OneHotEncoder should use drop='first' to prevent Dummy Variable Trap
            # sparse_output=False returns standard array (not sparse matrix)
            # handle_unknown='ignore' handles new categories in test data
            "one_hot": lambda **kwargs: SklearnOneHotEncoder(
                sparse_output=False, 
                handle_unknown='ignore',
                drop='first',  # Prevents multicollinearity (Dummy Variable Trap)
                **kwargs
            ),
            "label": lambda **kwargs: OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, **kwargs),
            # NEW: Target encoding for high-cardinality categoricals
            "target": lambda **kwargs: TargetEncoder(**kwargs),
            "mean_target": lambda **kwargs: TargetEncoder(**kwargs),
        },
        "transformation": {
            "log": lambda **kwargs: FunctionTransformer(func=np.log1p, inverse_func=np.expm1, **kwargs),
            "sqrt": lambda **kwargs: FunctionTransformer(func=np.sqrt, **kwargs),
            "power": lambda **kwargs: PowerTransformer(**kwargs),
        },
        # NEW: Outlier handling (from ML text Pages 127-141)
        "outlier": {
            "iqr_cap": lambda **kwargs: IQROutlierHandler(action="cap", **kwargs),
            "iqr_remove": lambda **kwargs: IQROutlierHandler(action="remove", **kwargs),
            "iqr_flag": lambda **kwargs: IQROutlierHandler(action="flag", **kwargs),
            "zscore_cap": lambda **kwargs: ZScoreOutlierHandler(action="cap", **kwargs),
            "zscore_remove": lambda **kwargs: ZScoreOutlierHandler(action="remove", **kwargs),
            "percentile_cap": lambda **kwargs: PercentileOutlierHandler(action="cap", **kwargs),
        },
        # NEW: Dimensionality reduction
        "dimensionality": {
            "pca": lambda **kwargs: PCATransformer(**kwargs),
        },
    }

    def __init__(self):
        """Initialize preprocessing pipeline."""
        self.pipeline: Optional[Pipeline] = None
        self.column_transformer: Optional[ColumnTransformer] = None
        self.fitted = False
        self.column_names: Optional[List[str]] = None

    def build_from_strategy(self, strategy: Dict[str, Any], X: Optional[pd.DataFrame] = None) -> "PreprocessingPipeline":
        """Build sklearn Pipeline from strategy JSON.

        Args:
            strategy: Strategy dictionary with preprocessing_steps
            X: Optional DataFrame to infer column types

        Returns:
            Self for method chaining
        """
        preprocessing_steps = strategy.get("preprocessing_steps", [])
        
        if not preprocessing_steps:
            # Empty pipeline - passthrough
            self.pipeline = Pipeline([("passthrough", FunctionTransformer())])
            return self

        # Separate steps by type for ColumnTransformer
        transformers = []
        pipeline_steps = []

        # Group steps that can be combined in ColumnTransformer
        imputation_steps = []
        encoding_steps = []
        scaling_steps = []
        transformation_steps = []
        dimensionality_steps = []
        outlier_steps = []
        feature_selection = None

        numeric_columns = []
        categorical_columns = []

        if X is not None:
            numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()

        for step_config in preprocessing_steps:
            step_type = step_config.get("step_type")
            method = step_config.get("method")
            parameters = step_config.get("parameters", {})
            columns = step_config.get("columns", [])

            if step_type == "imputation":
                if method in self.TRANSFORMER_MAP.get("imputation", {}):
                    transformer_func = self.TRANSFORMER_MAP["imputation"][method]
                    transformer = transformer_func(**parameters)
                    target_cols = columns if columns else (numeric_columns if method in ["mean", "median"] else categorical_columns)
                    if target_cols:
                        imputation_steps.append((f"impute_{method}", transformer, target_cols))
            
            elif step_type == "encoding":
                if method in self.TRANSFORMER_MAP.get("encoding", {}):
                    transformer_func = self.TRANSFORMER_MAP["encoding"][method]
                    transformer = transformer_func(**parameters)
                    target_cols = columns if columns else categorical_columns
                    if target_cols:
                        encoding_steps.append((f"encode_{method}", transformer, target_cols))
            
            elif step_type == "scaling":
                if method in self.TRANSFORMER_MAP.get("scaling", {}):
                    transformer_func = self.TRANSFORMER_MAP["scaling"][method]
                    transformer = transformer_func(**parameters)
                    target_cols = columns if columns else numeric_columns
                    if target_cols:
                        scaling_steps.append((f"scale_{method}", transformer, target_cols))
            
            elif step_type == "transformation":
                if method in self.TRANSFORMER_MAP.get("transformation", {}):
                    transformer_func = self.TRANSFORMER_MAP["transformation"][method]
                    transformer = transformer_func(**parameters)
                    target_cols = columns if columns else numeric_columns
                    if target_cols:
                        transformation_steps.append((f"transform_{method}", transformer, target_cols))
            
            elif step_type == "outlier":
                if method in self.TRANSFORMER_MAP.get("outlier", {}):
                    transformer_func = self.TRANSFORMER_MAP["outlier"][method]
                    transformer = transformer_func(**parameters)
                    target_cols = columns if columns else numeric_columns
                    if target_cols:
                        outlier_steps.append((f"outlier_{method}", transformer, target_cols))

            elif step_type == "dimensionality":
                if method in self.TRANSFORMER_MAP.get("dimensionality", {}):
                    transformer_func = self.TRANSFORMER_MAP["dimensionality"][method]
                    transformer = transformer_func(**parameters)
                    target_cols = columns if columns else numeric_columns
                    if target_cols:
                        dimensionality_steps.append((f"dim_{method}", transformer, target_cols))

            elif step_type == "feature_selection":
                k = parameters.get("k", 10)
                score_func = chi2 if strategy.get("task_type") == "classification" else f_regression
                feature_selection = SelectKBest(score_func=score_func, k=k)

        # Build ColumnTransformer with proper chaining
        # CRITICAL: ColumnTransformer applies transformers in PARALLEL, not sequentially!
        # So if we have both impute_cat and encode_cat on same columns, we get BOTH outputs concatenated.
        # Solution: Chain operations using nested Pipelines
        
        column_transformers = []
        
        # Build numeric pipeline: impute -> outlier -> scale -> transform
        numeric_pipeline_steps = []
        numeric_cols_for_pipeline = set()
        
        # Collect numeric imputation steps
        for name, trans, cols in imputation_steps:
            cols_list = cols if isinstance(cols, list) else ([cols] if cols else [])
            # Check if this is for numeric columns (either explicitly or by default)
            if cols_list:
                if all(c in numeric_columns for c in cols_list):
                    numeric_cols_for_pipeline.update(cols_list)
                    if not any(s[0] == "impute" for s in numeric_pipeline_steps):
                        numeric_pipeline_steps.append(("impute", trans))
            elif not cols_list and numeric_columns:
                # Empty columns means "all applicable" - check method to determine type
                method_name = name.split("_")[-1] if "_" in name else name
                if method_name in ["mean", "median"]:
                    numeric_cols_for_pipeline.update(numeric_columns)
                    if not any(s[0] == "impute" for s in numeric_pipeline_steps):
                        numeric_pipeline_steps.append(("impute", trans))
        
        # Collect other numeric steps
        for name, trans, cols in outlier_steps:
            cols_list = cols if isinstance(cols, list) else ([cols] if cols else [])
            if cols_list and all(c in numeric_columns for c in cols_list):
                numeric_pipeline_steps.append((name, trans))
            elif not cols_list and numeric_columns:
                numeric_pipeline_steps.append((name, trans))
        
        for name, trans, cols in scaling_steps:
            cols_list = cols if isinstance(cols, list) else ([cols] if cols else [])
            if cols_list and all(c in numeric_columns for c in cols_list):
                numeric_pipeline_steps.append((name, trans))
            elif not cols_list and numeric_columns:
                numeric_pipeline_steps.append((name, trans))
        
        for name, trans, cols in transformation_steps:
            cols_list = cols if isinstance(cols, list) else ([cols] if cols else [])
            if cols_list and all(c in numeric_columns for c in cols_list):
                numeric_pipeline_steps.append((name, trans))
            elif not cols_list and numeric_columns:
                numeric_pipeline_steps.append((name, trans))
        
        if numeric_pipeline_steps:
            numeric_pipeline = Pipeline(numeric_pipeline_steps)
            numeric_cols_list = list(numeric_cols_for_pipeline) if numeric_cols_for_pipeline else numeric_columns
            if numeric_cols_list:
                column_transformers.append(("numeric_pipeline", numeric_pipeline, numeric_cols_list))
        
        # Build categorical pipeline: impute -> encode
        categorical_pipeline_steps = []
        categorical_cols_for_pipeline = set()
        
        # Collect categorical imputation steps
        for name, trans, cols in imputation_steps:
            cols_list = cols if isinstance(cols, list) else ([cols] if cols else [])
            if cols_list:
                if all(c in categorical_columns for c in cols_list):
                    categorical_cols_for_pipeline.update(cols_list)
                    if not any(s[0] == "impute" for s in categorical_pipeline_steps):
                        categorical_pipeline_steps.append(("impute", trans))
            elif not cols_list and categorical_columns:
                method_name = name.split("_")[-1] if "_" in name else name
                if method_name in ["mode", "most_frequent"]:
                    categorical_cols_for_pipeline.update(categorical_columns)
                    if not any(s[0] == "impute" for s in categorical_pipeline_steps):
                        categorical_pipeline_steps.append(("impute", trans))
        
        # Collect encoding steps
        for name, trans, cols in encoding_steps:
            cols_list = cols if isinstance(cols, list) else ([cols] if cols else [])
            if cols_list:
                if all(c in categorical_columns for c in cols_list):
                    categorical_pipeline_steps.append((name, trans))
            elif not cols_list and categorical_columns:
                categorical_pipeline_steps.append((name, trans))
        
        if categorical_pipeline_steps:
            categorical_pipeline = Pipeline(categorical_pipeline_steps)
            categorical_cols_list = list(categorical_cols_for_pipeline) if categorical_cols_for_pipeline else categorical_columns
            if categorical_cols_list:
                column_transformers.append(("categorical_pipeline", categorical_pipeline, categorical_cols_list))
        
        # Handle unencoded categoricals (if any)
        if X is not None and categorical_columns:
            handled_categoricals = categorical_cols_for_pipeline
            unencoded_categoricals = [col for col in categorical_columns if col not in handled_categoricals]
            if unencoded_categoricals:
                # Add default encoding pipeline for unencoded categoricals
                # From ML PDF: Use drop='first' to prevent Dummy Variable Trap
                default_pipeline = Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent", add_indicator=True)),
                    ("encode", SklearnOneHotEncoder(
                        sparse_output=False, 
                        handle_unknown='ignore',
                        drop='first'  # Prevents multicollinearity
                    ))
                ])
                column_transformers.append(("categorical_default", default_pipeline, unencoded_categoricals))
        
        # Add dimensionality reduction steps (will be added to main pipeline after ColumnTransformer)
        dimensionality_pipeline_steps = []
        if dimensionality_steps:
            for name, trans, cols in dimensionality_steps:
                dimensionality_pipeline_steps.append((name, trans))


        # Build pipeline
        pipeline_steps = []
        
        if column_transformers:
            # Always drop unhandled columns to prevent string->float conversion errors
            # The ColumnTransformer will output only the transformed columns
            remainder_handling = 'drop'
            
            self.column_transformer = ColumnTransformer(
                transformers=column_transformers,
                remainder=remainder_handling,
                verbose_feature_names_out=False
            )
            pipeline_steps.append(("preprocessing", self.column_transformer))
        
        if feature_selection:
            pipeline_steps.append(("feature_selection", feature_selection))

        if pipeline_steps:
            self.pipeline = Pipeline(pipeline_steps)
        else:
            self.pipeline = Pipeline([("passthrough", FunctionTransformer())])

        return self

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "PreprocessingPipeline":
        """Fit sklearn pipeline.

        Args:
            X: Input DataFrame
            y: Optional target Series

        Returns:
            Self for method chaining
        """
        if self.pipeline is None:
            raise ValueError("Pipeline must be built before fitting")
        
        self.column_names = X.columns.tolist()
        self.pipeline.fit(X, y)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data through sklearn pipeline.

        Args:
            X: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        if self.pipeline is None:
            raise ValueError("Pipeline must be built before transform")

        X_transformed = self.pipeline.transform(X)
        
        # Convert to DataFrame if possible
        if isinstance(X_transformed, np.ndarray):
            # Try to get feature names from pipeline
            try:
                feature_names = self.pipeline.get_feature_names_out()
                return pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
            except:
                # Fallback: use generic column names
                return pd.DataFrame(
                    X_transformed,
                    columns=[f"feature_{i}" for i in range(X_transformed.shape[1])],
                    index=X.index
                )
        
        return X_transformed

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit and transform in one step.

        WARNING - Data Leakage Prevention:
            Only use fit_transform() on TRAINING data after train-test split.
            For validation/test data, use transform() only (not fit_transform).

            Correct usage:
                X_train_processed = pipeline.fit_transform(X_train, y_train)
                X_val_processed = pipeline.transform(X_val)  # No fit here!

        Args:
            X: Input DataFrame (should be training data only!)
            y: Optional target Series

        Returns:
            Transformed DataFrame
        """
        if self.pipeline is None:
            raise ValueError("Pipeline must be built before fit_transform")

        self.column_names = X.columns.tolist()
        X_transformed = self.pipeline.fit_transform(X, y)
        self.fitted = True

        # Convert to DataFrame with proper numeric dtypes
        if isinstance(X_transformed, np.ndarray):
            # Ensure numeric dtype (handles any object arrays from transformers)
            # First check if we have string/object data that needs encoding
            if X_transformed.dtype == object or not np.issubdtype(X_transformed.dtype, np.number):
                # Try to convert to numeric, coercing errors to NaN
                try:
                    df_temp = pd.DataFrame(X_transformed)
                    # Check which columns have non-numeric data
                    non_numeric_cols = []
                    for col_idx in range(df_temp.shape[1]):
                        try:
                            pd.to_numeric(df_temp.iloc[:, col_idx], errors='raise')
                        except (ValueError, TypeError):
                            non_numeric_cols.append(col_idx)
                    
                    if non_numeric_cols:
                        # Try to convert with coercion
                        X_transformed = df_temp.apply(pd.to_numeric, errors='coerce').values
                        # Check if we still have non-numeric data
                        if pd.isna(X_transformed).any():
                            raise ValueError(
                                f"Pipeline output contains non-numeric data in columns {non_numeric_cols}. "
                                "All categorical columns must be encoded. "
                                "The pipeline should automatically encode unencoded categoricals, "
                                "but some columns may have been passed through incorrectly."
                            )
                    else:
                        X_transformed = df_temp.values
                except Exception as e:
                    # If that fails, the pipeline didn't encode categoricals properly
                    raise ValueError(
                        f"Pipeline output contains non-numeric data that cannot be converted. "
                        f"Error: {str(e)}. "
                        "Ensure all categorical columns are encoded before conversion to float."
                    ) from e
            
            # Now convert to float64
            try:
                X_transformed = X_transformed.astype(np.float64)
            except (ValueError, TypeError) as e:
                # If direct conversion fails, try element-wise with error handling
                try:
                    X_transformed = pd.DataFrame(X_transformed).apply(pd.to_numeric, errors='coerce').values.astype(np.float64)
                except Exception as e2:
                    raise ValueError(
                        f"Failed to convert pipeline output to numeric: {str(e2)}. "
                        "This usually means categorical columns were not properly encoded."
                    ) from e2

            try:
                feature_names = self.pipeline.get_feature_names_out()
                return pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
            except:
                return pd.DataFrame(
                    X_transformed,
                    columns=[f"feature_{i}" for i in range(X_transformed.shape[1])],
                    index=X.index
                )

        # If DataFrame, ensure numeric columns
        if isinstance(X_transformed, pd.DataFrame):
            for col in X_transformed.columns:
                if X_transformed[col].dtype == 'object':
                    X_transformed[col] = pd.to_numeric(X_transformed[col], errors='coerce')

        return X_transformed

    def save(self, filepath: str):
        """Save fitted pipeline to disk.

        Args:
            filepath: Path to save pipeline
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before saving")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "PreprocessingPipeline":
        """Load fitted pipeline from disk.

        Args:
            filepath: Path to saved pipeline

        Returns:
            Loaded pipeline
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)
    
    def get_feature_names_out(self) -> List[str]:
        """Get output feature names from pipeline.
        
        Returns:
            List of feature names
        """
        if self.pipeline is None:
            return []
        try:
            return self.pipeline.get_feature_names_out().tolist()
        except:
            return []