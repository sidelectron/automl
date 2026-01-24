"""Ollama LLM provider with fine-tuning data collection.

This module provides integration with Ollama for local LLM inference,
including automatic logging of all interactions for fine-tuning.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional, Dict, Any

try:
    import ollama
except ImportError:
    ollama = None

from .llm_interface import LLMInterface
from ..utils.fine_tuning_logger import FineTuningLogger


class OllamaProvider(LLMInterface):
    """Ollama LLM provider with integrated fine-tuning logging.

    Features:
    - Local LLM inference via Ollama
    - JSON mode support for structured outputs
    - Automatic retry with validation feedback
    - Fine-tuning data collection (all interactions logged)
    - Configurable temperature and token limits
    """

    def __init__(
        self,
        model: str = "qwen2.5-coder:3b",
        log_interactions: bool = True,
        fine_tuning_logger: Optional[FineTuningLogger] = None
    ):
        """Initialize the Ollama provider.

        Args:
            model: Ollama model name (default: qwen2.5-coder:3b)
            log_interactions: Whether to log interactions for fine-tuning
            fine_tuning_logger: Custom logger instance (if None, creates default)

        Raises:
            ImportError: If ollama package is not installed
        """
        if ollama is None:
            raise ImportError(
                "ollama package not installed. Install with: pip install ollama"
            )

        self.model = model
        self.log_interactions = log_interactions
        self.logger = fine_tuning_logger or FineTuningLogger()

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        format: Optional[str] = None,
        agent: str = "unknown",
        max_retries: int = 3,
        timeout: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate a response from Ollama.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            format: Output format ("json" for JSON mode)
            agent: Name of the calling agent (for logging)
            max_retries: Maximum retry attempts on failure
            timeout: Optional timeout in seconds (raises if exceeded)
            **kwargs: Additional Ollama-specific options

        Returns:
            The generated response as a string

        Raises:
            Exception: If all retries fail or timeout is exceeded
        """
        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
            **kwargs
        }

        def _chat():
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format=format,
                options=options
            )
            return response["message"]["content"]

        retry = 0
        last_error = None

        while retry < max_retries:
            try:
                if timeout is not None and timeout > 0:
                    with ThreadPoolExecutor(max_workers=1) as ex:
                        future = ex.submit(_chat)
                        response_text = future.result(timeout=timeout)
                else:
                    response_text = _chat()

                if self.log_interactions:
                    self.logger.log_interaction(
                        prompt=prompt,
                        response=response_text,
                        agent=agent,
                        valid=True,
                        metadata={
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "format": format,
                            "retry": retry
                        }
                    )
                return response_text

            except FuturesTimeoutError:
                err = Exception(f"Ollama generation timed out after {timeout} seconds")
                last_error = err
                if self.log_interactions:
                    self.logger.log_interaction(
                        prompt=prompt,
                        response=f"ERROR: {str(err)}",
                        agent=agent,
                        valid=False,
                        metadata={"error": str(err), "retry": retry, "temperature": temperature}
                    )
                retry += 1
                if retry >= max_retries:
                    raise err
            except Exception as e:
                last_error = e
                retry += 1
                if self.log_interactions:
                    self.logger.log_interaction(
                        prompt=prompt,
                        response=f"ERROR: {str(e)}",
                        agent=agent,
                        valid=False,
                        metadata={
                            "error": str(e),
                            "retry": retry,
                            "temperature": temperature
                        }
                    )
                if retry >= max_retries:
                    raise Exception(
                        f"Ollama generation failed after {max_retries} retries: {last_error}"
                    )

        raise Exception(f"Ollama generation failed: {last_error}")

    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        agent: str = "unknown",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a JSON response from Ollama.

        Uses Ollama's native JSON mode for structured outputs.
        Includes fallback parsing for responses wrapped in markdown.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            agent: Name of the calling agent (for logging)
            **kwargs: Additional Ollama options

        Returns:
            Parsed JSON response as a dictionary

        Raises:
            Exception: If response is not valid JSON after parsing attempts
        """
        # Request JSON format from Ollama
        response_text = self.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            format="json",  # Enable JSON mode
            agent=agent,
            **kwargs
        )

        # Try direct JSON parsing first
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback: extract JSON from markdown code blocks
            # Pattern matches: ```json\n{...}\n``` or ```\n{...}\n```
            pattern = r"^```(?:json)?\s*\n(.*?)(?=^```)```"
            matches = re.findall(pattern, response_text, re.DOTALL | re.MULTILINE)

            if matches:
                try:
                    return json.loads(matches[0].strip())
                except json.JSONDecodeError:
                    pass

            # Last resort: look for JSON object in response
            # Find first { and last }
            start = response_text.find("{")
            end = response_text.rfind("}")

            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(response_text[start:end + 1])
                except json.JSONDecodeError:
                    pass

            # All parsing failed
            raise Exception(
                f"Failed to parse JSON from Ollama response. Response: {response_text[:200]}..."
            )

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current Ollama model.

        Returns:
            Dictionary with model information
        """
        try:
            # Get model details from Ollama
            response = ollama.show(self.model)
            return {
                "model": self.model,
                "details": response.get("details", {}),
                "modelfile": response.get("modelfile", ""),
                "parameters": response.get("parameters", ""),
            }
        except Exception as e:
            return {
                "model": self.model,
                "error": str(e)
            }

    def validate_and_relog(
        self,
        prompt: str,
        response: str,
        agent: str,
        is_valid: bool
    ) -> None:
        """Update validation status of a logged interaction.

        Call this after validating an LLM response to update the log
        with the correct validation status.

        Args:
            prompt: The original prompt
            response: The LLM response
            agent: Agent name
            is_valid: Whether the response passed validation
        """
        if self.log_interactions:
            # Re-log with correct validation status
            # The logger will append a new entry with the validation result
            self.logger.log_interaction(
                prompt=prompt,
                response=response,
                agent=agent,
                valid=is_valid,
                metadata={"validation_update": True}
            )
