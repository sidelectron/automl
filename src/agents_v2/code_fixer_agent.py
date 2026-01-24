"""Code fixer agent for automatically fixing errors in generated code."""

import re
from typing import Dict, Any, Optional
from pathlib import Path

from ..llm.llm_interface import LLMInterface


class CodeFixerAgent:
    """Agent for fixing errors in generated code."""

    def __init__(self, llm_provider: LLMInterface):
        """Initialize code fixer agent.

        Args:
            llm_provider: LLM provider instance
        """
        self.llm = llm_provider
        self.agent_name = "code_fixer"

    def fix_syntax_error(
        self,
        code: str,
        error: str
    ) -> str:
        """Fix syntax errors in code.

        Args:
            code: Code with syntax error
            error: Error message

        Returns:
            Fixed code
        """
        prompt = f"""The following Python code has a syntax error. Fix it.

# Code with Error
```python
{code}
```

# Error Message
{error}

Fix the syntax error and return the corrected code. Start with ```python and end with ```.
"""

        try:
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.1,  # Very low temperature for error fixing
                max_tokens=4096,
                agent=self.agent_name,
                timeout=180
            )

            # Extract code from response
            fixed_code = self._extract_code_from_response(response)
            return fixed_code if fixed_code else code

        except Exception as e:
            # If fixing fails, return original code
            return code

    def fix_import_error(
        self,
        code: str,
        error: str
    ) -> str:
        """Fix import errors in code.

        Args:
            code: Code with import error
            error: Error message

        Returns:
            Fixed code with correct imports
        """
        prompt = f"""The following Python code has an import error. Fix it by adding the missing import.

# Code with Error
```python
{code}
```

# Error Message
{error}

Fix the import error by adding the necessary import statement. Only use allowed imports:
sklearn, pandas, numpy, xgboost, lightgbm, imblearn, pathlib, pickle, json

Start with ```python and end with ```.
"""

        try:
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=4096,
                agent=self.agent_name,
                timeout=180
            )

            fixed_code = self._extract_code_from_response(response)
            return fixed_code if fixed_code else code

        except Exception as e:
            return code

    def fix_runtime_error(
        self,
        code: str,
        error: str,
        execution_log: str
    ) -> str:
        """Fix runtime errors in code.

        Args:
            code: Code with runtime error
            error: Error message
            execution_log: Full execution log

        Returns:
            Fixed code
        """
        prompt = f"""The following Python code has a runtime error. Fix it.

# Code with Error
```python
{code}
```

# Error Message
{error}

# Execution Log
{execution_log}

Fix the runtime error and return the corrected code. Ensure:
1. All variables are defined
2. All function calls are valid
3. All imports are correct
4. Code logic is correct

Start with ```python and end with ```.
"""

        try:
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.2,  # Slightly higher for logic fixes
                max_tokens=4096,
                agent=self.agent_name,
                timeout=180
            )

            fixed_code = self._extract_code_from_response(response)
            return fixed_code if fixed_code else code

        except Exception as e:
            return code

    def fix_code(
        self,
        code: str,
        error: str,
        error_type: str,
        execution_log: Optional[str] = None
    ) -> str:
        """Unified method to fix code based on error type.

        Args:
            code: Code with error
            error: Error message
            error_type: Type of error ('syntax', 'import', 'runtime')
            execution_log: Optional full execution log

        Returns:
            Fixed code
        """
        if error_type == "syntax":
            return self.fix_syntax_error(code, error)
        elif error_type == "import":
            return self.fix_import_error(code, error)
        else:
            return self.fix_runtime_error(code, error, execution_log or "")

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response.

        Args:
            response: LLM response

        Returns:
            Extracted code
        """
        # Try to extract from ```python blocks
        pattern = r"```python\s*(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # If no code blocks, return response as-is
        return response.strip()
