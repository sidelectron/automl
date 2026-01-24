"""Multi-agent system for Intent-Driven AutoML.

This package contains specialized agents that coordinate to build
optimal ML solutions with business context awareness.
"""

from .orchestrator import Orchestrator

__all__ = ["Orchestrator"]
