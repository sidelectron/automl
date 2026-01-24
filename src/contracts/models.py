"""Pydantic models for contract validation.

These models provide type-safe validation for all agent communications.
"""

from typing import Optional, Dict, Any, List, Union, Literal
from pydantic import BaseModel, Field


class BusinessContext(BaseModel):
    """Business context for intent-driven optimization."""

    priority_metric: Literal["recall", "precision", "f1", "accuracy", "net_value", "roi"] = Field(
        ...,
        description="Primary metric to optimize for"
    )
    true_positive_value: float = Field(
        default=0.0,
        description="Value of a true positive (e.g., saved customer worth $500)"
    )
    false_positive_cost: float = Field(
        default=0.0,
        description="Cost of a false positive (e.g., retention call costs $20)"
    )
    cost_ratio: float = Field(
        default=0.0,
        description="Ratio of false_positive_cost to true_positive_value"
    )
    value_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional value metrics for business context"
    )


class IntentSchema(BaseModel):
    """User intent with business context."""

    task_type: Literal["binary_classification", "multiclass_classification", "regression"] = Field(
        ...,
        description="Type of machine learning task"
    )
    target_variable: str = Field(..., description="Name of the target variable to predict")
    business_context: BusinessContext = Field(..., description="Business context for optimization")
    dataset_path: Optional[str] = Field(None, description="Path to the dataset file")
    description: Optional[str] = Field(None, description="Natural language description of the task")


class IntentFlag(BaseModel):
    """Intent-aware warning or recommendation."""

    type: Literal["warning", "info", "recommendation"] = Field(...)
    message: str = Field(..., description="Flag message")
    severity: Literal["low", "medium", "high"] = Field(...)


class ProfileSchema(BaseModel):
    """Data profile with statistics and intent-aware flags."""

    data_types: Dict[str, str] = Field(
        ...,
        description="Data types for each column",
        additional_properties=True
    )
    statistics: Dict[str, Any] = Field(
        ...,
        description="Statistical summary of the data"
    )
    missing_values: Dict[str, int] = Field(
        ...,
        description="Missing value counts per column"
    )
    intent_flags: List[IntentFlag] = Field(
        default_factory=list,
        description="Intent-aware warnings and recommendations"
    )
    class_distribution: Optional[Dict[str, float]] = Field(
        None,
        description="Class distribution for classification tasks"
    )
    insights: List[str] = Field(
        default_factory=list,
        description="LLM-generated insights about the data"
    )


class PreprocessingStep(BaseModel):
    """A single preprocessing step."""

    step_type: Literal["imputation", "scaling", "encoding", "feature_engineering", "handling_imbalance"] = Field(...)
    method: str = Field(..., description="Specific method (e.g., 'median', 'standard_scaler')")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Method-specific parameters"
    )
    columns: List[str] = Field(
        default_factory=list,
        description="Columns to apply this step to (empty = all applicable)"
    )


class StrategySchema(BaseModel):
    """A preprocessing strategy."""

    name: str = Field(..., description="Strategy name/identifier")
    preprocessing_steps: List[PreprocessingStep] = Field(
        ...,
        description="Ordered list of preprocessing steps"
    )
    model_candidates: List[Literal["xgboost", "lightgbm", "random_forest", "logistic_regression", "svm"]] = Field(
        ...,
        description="List of model types to try"
    )
    rationale: Optional[str] = Field(None, description="Explanation of why this strategy is suitable")
    expected_outcomes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Expected results from this strategy"
    )


class ConfusionMatrix(BaseModel):
    """Confusion matrix components."""

    true_positive: int = Field(..., description="True positives")
    true_negative: int = Field(..., description="True negatives")
    false_positive: int = Field(..., description="False positives")
    false_negative: int = Field(..., description="False negatives")


class Metrics(BaseModel):
    """Standard ML metrics."""

    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    confusion_matrix: Optional[ConfusionMatrix] = None


class BusinessMetrics(BaseModel):
    """Business impact metrics."""

    net_value: float = Field(..., description="Net business value: (TP × value) - (FP × cost)")
    roi: float = Field(..., description="Return on investment: net_value / total_cost")
    total_cost: float = Field(..., description="Total cost: (TP + FP) × cost_per_action")
    total_value: float = Field(..., description="Total value: TP × value_per_positive")


class TrainingResultSchema(BaseModel):
    """Training results for a strategy-model-threshold combination."""

    strategy_name: str = Field(..., description="Name of the preprocessing strategy used")
    model_name: str = Field(..., description="Name of the model")
    threshold: float = Field(..., description="Decision threshold used (0.0-1.0)")
    metrics: Metrics = Field(..., description="Standard ML metrics")
    business_metrics: Optional[BusinessMetrics] = Field(
        None,
        description="Business impact metrics"
    )
    model_path: Optional[str] = Field(None, description="Path to saved model file")
    preprocessing_path: Optional[str] = Field(None, description="Path to saved preprocessing pipeline")


class Winner(BaseModel):
    """Best performing combination."""

    strategy_name: str
    model_name: str
    threshold: float
    metrics: Dict[str, Any]
    business_metrics: Dict[str, Any]
    model_path: Optional[str] = Field(None, description="Path to saved model file")
    preprocessing_path: Optional[str] = Field(None, description="Path to preprocessing pipeline")


class BusinessImpact(BaseModel):
    """Business impact breakdown."""

    true_positives: int = Field(..., description="Number of true positives")
    false_positives: int = Field(..., description="Number of false positives")
    false_negatives: int = Field(..., description="Number of false negatives")
    total_actions: int = Field(..., description="Total actions taken (TP + FP)")
    explanation: str = Field(..., description="Natural language explanation")


class FinancialImpact(BaseModel):
    """Financial metrics."""

    potential_value: float = Field(..., description="Potential value if all positives were true")
    total_cost: float = Field(..., description="Total cost of actions")
    net_value: float = Field(..., description="Net value: potential_value - total_cost")
    roi: float = Field(..., description="Return on investment ratio")


class Alternative(BaseModel):
    """Alternative option."""

    strategy_name: str
    model_name: str
    threshold: float
    business_metrics: Dict[str, Any]


class ComparisonResultSchema(BaseModel):
    """Comparison results with winner and business explanation."""

    winner: Winner = Field(..., description="Best performing combination")
    business_impact: BusinessImpact = Field(..., description="Business impact breakdown")
    financial_impact: FinancialImpact = Field(..., description="Financial metrics")
    recommended_threshold: float = Field(..., description="Optimal threshold for business goals")
    alternatives: List[Alternative] = Field(
        default_factory=list,
        description="Top alternative options"
    )
    comparison_text: str = Field(..., description="LLM-generated comparison explanation")
