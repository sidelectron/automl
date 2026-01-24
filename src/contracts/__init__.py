"""Contracts and schemas for agent communication.

This package defines JSON schemas and Pydantic models for structured
communication between agents in the AutoML platform.
"""

from .models import (
    IntentSchema,
    ProfileSchema,
    StrategySchema,
    TrainingResultSchema,
    ComparisonResultSchema,
)

__all__ = [
    "IntentSchema",
    "ProfileSchema",
    "StrategySchema",
    "TrainingResultSchema",
    "ComparisonResultSchema",
]
