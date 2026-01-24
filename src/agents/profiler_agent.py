"""Profiler Agent for intent-aware data analysis."""

import json
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

from ..llm.llm_interface import LLMInterface
from ..contracts.models import ProfileSchema, IntentSchema
from ..utils.data_loader import load_csv_robust, DataLoadResult


class ProfilerAgent:
    """Agent for profiling data with intent awareness."""

    def __init__(self, llm_provider: LLMInterface):
        """Initialize the Profiler Agent.

        Args:
            llm_provider: LLM provider instance
        """
        self.llm = llm_provider
        self.agent_name = "profiler"

    def _load_prompt_template(self) -> str:
        """Load the profiling prompt template.

        Returns:
            Prompt template string
        """
        prompt_path = Path(__file__).parent.parent / "llm" / "prompts" / "profiling.txt"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return "Analyze the dataset and generate a profile with intent-aware insights."

    def _detect_data_type(self, series: pd.Series) -> str:
        """Detect data type of a pandas Series.

        Args:
            series: Pandas Series

        Returns:
            Data type: "numeric", "categorical", "datetime", or "text"
        """
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        elif pd.api.types.is_numeric_dtype(series):
            return "numeric"
        elif series.dtype == "object":
            # Check if it's text (long strings) or categorical (few unique values)
            unique_ratio = series.nunique() / len(series)
            avg_length = series.astype(str).str.len().mean()
            if unique_ratio > 0.5 and avg_length > 20:
                return "text"
            return "categorical"
        return "categorical"

    def _calculate_statistics(self, df: pd.DataFrame, data_types: Dict[str, str]) -> Dict[str, Any]:
        """Calculate statistical summary including skewness and kurtosis.

        Args:
            df: DataFrame
            data_types: Dictionary mapping column names to types

        Returns:
            Statistics dictionary
        """
        stats = {
            "shape": list(df.shape),
            "numeric_summary": {},
            "categorical_summary": {}
        }

        # Numeric summary with skewness and kurtosis
        numeric_cols = [col for col, dtype in data_types.items() if dtype == "numeric"]
        if numeric_cols:
            numeric_df = df[numeric_cols]
            stats["numeric_summary"] = {
                col: {
                    "mean": float(numeric_df[col].mean()) if not numeric_df[col].isna().all() else None,
                    "median": float(numeric_df[col].median()) if not numeric_df[col].isna().all() else None,
                    "std": float(numeric_df[col].std()) if not numeric_df[col].isna().all() else None,
                    "min": float(numeric_df[col].min()) if not numeric_df[col].isna().all() else None,
                    "max": float(numeric_df[col].max()) if not numeric_df[col].isna().all() else None,
                    # NEW: Skewness and Kurtosis (from ML text Page 48)
                    "skewness": float(numeric_df[col].skew()) if not numeric_df[col].isna().all() else None,
                    "kurtosis": float(numeric_df[col].kurtosis()) if not numeric_df[col].isna().all() else None,
                    # NEW: Quartiles for IQR calculation
                    "q1": float(numeric_df[col].quantile(0.25)) if not numeric_df[col].isna().all() else None,
                    "q3": float(numeric_df[col].quantile(0.75)) if not numeric_df[col].isna().all() else None,
                }
                for col in numeric_cols
            }

        # Categorical summary
        categorical_cols = [col for col, dtype in data_types.items() if dtype == "categorical"]
        if categorical_cols:
            stats["categorical_summary"] = {
                col: df[col].value_counts().to_dict()
                for col in categorical_cols
            }

        return stats

    def _detect_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect duplicate rows in the DataFrame.

        From ML text Page 43: "Duplicate rows can mislead your model"

        Args:
            df: DataFrame

        Returns:
            Dictionary with duplicate information
        """
        duplicate_count = int(df.duplicated().sum())
        total_rows = len(df)

        duplicates_info = {
            "count": duplicate_count,
            "percentage": round(duplicate_count / total_rows * 100, 2) if total_rows > 0 else 0,
            "has_duplicates": duplicate_count > 0
        }

        # Get sample of duplicate rows (first 5)
        if duplicate_count > 0:
            duplicate_rows = df[df.duplicated(keep='first')]
            duplicates_info["sample_indices"] = duplicate_rows.head(5).index.tolist()

        return duplicates_info

    def _detect_outliers_iqr(
        self,
        df: pd.DataFrame,
        data_types: Dict[str, str],
        threshold: float = 1.5
    ) -> Dict[str, Any]:
        """Detect outliers using IQR method.

        From ML text Pages 127-134: IQR Method for outlier detection
        Formula: Lower = Q1 - 1.5*IQR, Upper = Q3 + 1.5*IQR

        Args:
            df: DataFrame
            data_types: Dictionary mapping column names to types
            threshold: IQR multiplier (default 1.5)

        Returns:
            Dictionary with outlier information per column
        """
        outliers = {}
        numeric_cols = [col for col, dtype in data_types.items() if dtype == "numeric"]

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_count = int(outlier_mask.sum())

            if outlier_count > 0 or True:  # Always include for transparency
                outliers[col] = {
                    "count": outlier_count,
                    "percentage": round(outlier_count / len(series) * 100, 2),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "iqr": float(iqr),
                    "has_outliers": outlier_count > 0
                }

        return outliers

    def _calculate_correlations(
        self,
        df: pd.DataFrame,
        data_types: Dict[str, str],
        target_variable: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculate correlations between numeric features and with target.

        From ML text Page 56: Correlation analysis

        Args:
            df: DataFrame
            data_types: Dictionary mapping column names to types
            target_variable: Optional target column name

        Returns:
            Dictionary with correlation information
        """
        numeric_cols = [col for col, dtype in data_types.items() if dtype == "numeric"]

        if len(numeric_cols) < 2:
            return {"message": "Not enough numeric columns for correlation"}

        correlations = {
            "method": "pearson",
            "feature_correlations": {},
            "target_correlations": {}
        }

        # Calculate full correlation matrix
        try:
            corr_matrix = df[numeric_cols].corr()

            # Find highly correlated feature pairs (|r| > 0.8)
            high_corr_pairs = []
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) > 0.8:
                        high_corr_pairs.append({
                            "feature1": col1,
                            "feature2": col2,
                            "correlation": round(float(corr_val), 4)
                        })

            correlations["high_correlation_pairs"] = high_corr_pairs

            # Correlations with target variable
            if target_variable and target_variable in numeric_cols:
                target_corr = corr_matrix[target_variable].drop(target_variable)
                correlations["target_correlations"] = {
                    col: round(float(val), 4)
                    for col, val in target_corr.sort_values(ascending=False).items()
                }

        except Exception as e:
            correlations["error"] = str(e)

        return correlations

    def _analyze_class_distribution(
        self,
        df: pd.DataFrame,
        target_variable: str
    ) -> Optional[Dict[str, float]]:
        """Analyze class distribution for classification tasks.

        Args:
            df: DataFrame
            target_variable: Name of target variable

        Returns:
            Class distribution dictionary or None
        """
        if target_variable not in df.columns:
            return None

        value_counts = df[target_variable].value_counts()
        total = len(df)
        return {str(k): float(v / total) for k, v in value_counts.items()}

    def profile(
        self,
        dataset_path: str,
        intent: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Profile the dataset with intent awareness.

        Implements comprehensive data profiling from ML text:
        - Data types detection
        - Missing values analysis
        - Duplicate detection (Page 43)
        - Skewness/Kurtosis (Page 48)
        - Outlier detection using IQR (Pages 127-134)
        - Correlation analysis (Page 56)

        Args:
            dataset_path: Path to dataset file (CSV)
            intent: Optional parsed intent for intent-aware analysis

        Returns:
            Profile dictionary matching ProfileSchema
        """
        # Load dataset using robust loader with encoding detection
        load_result = load_csv_robust(dataset_path)

        if not load_result.load_successful:
            return {
                "error": load_result.error_message,
                "warnings": load_result.warnings,
                "load_successful": False
            }

        df = load_result.df

        # Detect data types
        data_types = {col: self._detect_data_type(df[col]) for col in df.columns}

        # Calculate statistics (now includes skewness, kurtosis, quartiles)
        statistics = self._calculate_statistics(df, data_types)

        # Missing values
        missing_values = {col: int(df[col].isna().sum()) for col in df.columns}

        # NEW: Duplicate detection (ML text Page 43)
        duplicates = self._detect_duplicates(df)

        # NEW: Outlier detection using IQR (ML text Pages 127-134)
        outliers = self._detect_outliers_iqr(df, data_types)

        # NEW: Correlation analysis (ML text Page 56)
        target_variable = intent.get("target_variable") if intent else None
        correlations = self._calculate_correlations(df, data_types, target_variable)

        # Class distribution (if classification task and target specified)
        class_distribution = None
        if intent and intent.get("task_type") in ["binary_classification", "multiclass_classification"]:
            target_variable = intent.get("target_variable")
            if target_variable:
                class_distribution = self._analyze_class_distribution(df, target_variable)

        # Build context for LLM
        profile_context = {
            "data_types": data_types,
            "statistics": statistics,
            "missing_values": missing_values,
            "class_distribution": class_distribution,
            "duplicates": duplicates,
            "outliers": outliers,
            "correlations": correlations
        }

        # Generate intent-aware flags and insights using LLM
        system_prompt = self._load_prompt_template()

        user_prompt = f"""Analyze the following dataset profile and generate intent-aware flags and insights.

# Dataset Profile #
```json
{json.dumps(profile_context, indent=2)}
```

# User Intent #
```json
{json.dumps(intent, indent=2) if intent else "{}"}
```

Generate intent-aware flags and insights based on the profile and intent.
"""

        try:
            llm_response = self.llm.generate_json(
                prompt=system_prompt + "\n\n" + user_prompt,
                temperature=0.1,
                agent=self.agent_name,
                timeout=180
            )
        except Exception as e:
            # Fallback: generate basic flags without LLM
            llm_response = {
                "intent_flags": [],
                "insights": ["Dataset loaded successfully. Basic analysis complete."]
            }

        # Merge LLM response with calculated profile
        # Ensure intent_flags is always a list
        intent_flags = llm_response.get("intent_flags", [])
        if not isinstance(intent_flags, list):
            # If LLM returned a dict or other type, convert to list
            intent_flags = []
        
        profile = {
            "data_types": data_types,
            "statistics": statistics,
            "missing_values": missing_values,
            "duplicates": duplicates,
            "outliers": outliers,
            "correlations": correlations,
            "intent_flags": intent_flags,
            "insights": llm_response.get("insights", []),
            # Include data loading metadata
            "load_metadata": {
                "encoding": load_result.encoding,
                "delimiter": load_result.delimiter,
                "file_size_mb": load_result.file_size_mb,
                "warnings": load_result.warnings
            }
        }

        if class_distribution:
            profile["class_distribution"] = class_distribution

        # Add intent-aware flags based on analysis
        if intent:
            priority_metric = intent.get("business_context", {}).get("priority_metric")
            cost_ratio = intent.get("business_context", {}).get("cost_ratio", 0.0)

            # Check for low positive class rate (for recall optimization)
            if priority_metric == "recall" and class_distribution:
                positive_class_rate = min(class_distribution.values())
                if positive_class_rate < 0.1:
                    profile["intent_flags"].append({
                        "type": "warning",
                        "message": f"Low positive class rate detected ({positive_class_rate:.1%}). For recall optimization, consider handling class imbalance.",
                        "severity": "high"
                    })

            # Check for high cost ratio (for precision optimization)
            if priority_metric == "precision" and cost_ratio > 0.1:
                profile["intent_flags"].append({
                    "type": "warning",
                    "message": f"High cost ratio detected ({cost_ratio:.2f}). Precision optimization recommended to minimize false positives.",
                    "severity": "medium"
                })

            # Check for high missing values
            total_rows = statistics["shape"][0]
            for col, missing_count in missing_values.items():
                missing_ratio = missing_count / total_rows if total_rows > 0 else 0
                if missing_ratio > 0.2:
                    profile["intent_flags"].append({
                        "type": "warning",
                        "message": f"High missing value rate in '{col}' ({missing_ratio:.1%}). Consider imputation strategy.",
                        "severity": "medium"
                    })

            # NEW: Check for duplicates
            if duplicates.get("has_duplicates"):
                profile["intent_flags"].append({
                    "type": "warning",
                    "message": f"Found {duplicates['count']} duplicate rows ({duplicates['percentage']:.1f}%). Consider removing duplicates.",
                    "severity": "medium"
                })

            # NEW: Check for significant outliers
            for col, outlier_info in outliers.items():
                if outlier_info.get("has_outliers") and outlier_info.get("percentage", 0) > 5:
                    profile["intent_flags"].append({
                        "type": "warning",
                        "message": f"Column '{col}' has {outlier_info['percentage']:.1f}% outliers. Consider capping or removal.",
                        "severity": "medium"
                    })

            # NEW: Check for highly skewed distributions
            for col, col_stats in statistics.get("numeric_summary", {}).items():
                skewness = col_stats.get("skewness")
                if skewness is not None and abs(skewness) > 2:
                    profile["intent_flags"].append({
                        "type": "info",
                        "message": f"Column '{col}' is highly skewed (skewness={skewness:.2f}). Consider log/power transformation.",
                        "severity": "low"
                    })

            # NEW: Check for highly correlated features
            high_corr = correlations.get("high_correlation_pairs", [])
            if high_corr:
                for pair in high_corr[:3]:  # Limit to top 3
                    profile["intent_flags"].append({
                        "type": "info",
                        "message": f"High correlation ({pair['correlation']:.2f}) between '{pair['feature1']}' and '{pair['feature2']}'. Consider removing one.",
                        "severity": "low"
                    })

        # Validate against Pydantic model
        try:
            validated = ProfileSchema(**profile)
            return validated.model_dump()
        except Exception as e:
            # Return unvalidated if validation fails (for debugging)
            return profile
