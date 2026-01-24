"""Abstract interface for LLM providers.

This module defines the contract that all LLM providers must implement,
allowing for easy swapping between different backends (Ollama, OpenAI, etc.).
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class LLMInterface(ABC):
    """Abstract base class for LLM providers.

    All LLM providers (Ollama, OpenAI, Claude, etc.) must implement this interface
    to ensure consistent behavior across the platform.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        format: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum number of tokens to generate
            format: Output format hint (e.g., "json")
            **kwargs: Additional provider-specific parameters

        Returns:
            The generated response as a string

        Raises:
            Exception: If the LLM call fails after retries
        """
        pass

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a JSON response from the LLM.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            Parsed JSON response as a dictionary

        Raises:
            Exception: If the LLM call fails or response is not valid JSON
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary with model information (name, version, etc.)
        """
        pass
