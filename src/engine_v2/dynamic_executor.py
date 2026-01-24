"""Dynamic executor for running LLM-generated Python code safely."""

import subprocess
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import tempfile
import os


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    return_code: int
    stdout: str
    stderr: str
    execution_time: float
    error_type: Optional[str] = None


class DynamicExecutor:
    """Execute generated Python code safely in isolated environment."""

    def __init__(
        self,
        work_dir: Optional[str] = None,
        timeout: int = 30,
        device: str = "cpu",
        code_fixer: Optional[Callable] = None
    ):
        """Initialize dynamic executor.

        Args:
            work_dir: Working directory for code execution (default: temp directory)
            timeout: Execution timeout in seconds (default: 30)
            device: Device identifier (for compatibility, not used in CPU-only mode)
            code_fixer: Optional function to fix code errors (CodeFixerAgent method)
        """
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp())
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.device = device
        self.code_fixer = code_fixer

    def execute_code(
        self,
        code: str,
        script_name: str = "generated_code.py",
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> ExecutionResult:
        """Execute Python code safely.

        Args:
            code: Python code to execute
            script_name: Name of the script file
            context: Optional context dictionary (not used in subprocess execution)
            timeout: Optional timeout in seconds (override; default uses self.timeout)

        Returns:
            ExecutionResult with success status and output
        """
        script_path = self.work_dir / script_name
        script_path.parent.mkdir(parents=True, exist_ok=True)

        # Write code to file
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        # Execute code
        start_time = time.time()
        result = self._execute_script(str(script_path), timeout=timeout)
        execution_time = time.time() - start_time

        # Determine error type
        error_type = None
        if not result.success:
            if "SyntaxError" in result.stderr or "IndentationError" in result.stderr:
                error_type = "syntax"
            elif "ModuleNotFoundError" in result.stderr or "ImportError" in result.stderr:
                error_type = "import"
            else:
                error_type = "runtime"

            return ExecutionResult(
                success=result.success,
                return_code=result.return_code,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=execution_time,
                error_type=error_type
            )

        return ExecutionResult(
            success=True,
            return_code=result.return_code,
            stdout=result.stdout,
            stderr=result.stderr,
            execution_time=execution_time,
            error_type=None
        )

    def execute_with_retry(
        self,
        code: str,
        script_name: str = "generated_code.py",
        max_attempts: int = 5,
        context: Optional[Dict[str, Any]] = None,
        fix_code_callback: Optional[Callable] = None,
        timeout: Optional[int] = None
    ) -> ExecutionResult:
        """Execute code with automatic error recovery.

        Args:
            code: Python code to execute
            script_name: Name of the script file
            max_attempts: Maximum retry attempts (default: 5)
            context: Optional context dictionary
            fix_code_callback: Optional callback to fix code (signature: fix_code(code, error, error_type, execution_log) -> str)
            timeout: Optional timeout in seconds (override per call)

        Returns:
            ExecutionResult (from last attempt)
        """
        current_code = code
        last_result = None

        for attempt in range(max_attempts):
            # Execute code
            result = self.execute_code(current_code, script_name, context, timeout=timeout)
            last_result = result

            if result.success:
                return result

            # If we have a code fixer and this isn't the last attempt, try to fix
            if fix_code_callback and attempt < max_attempts - 1:
                error_msg = result.stderr or result.stdout
                error_type = result.error_type or "runtime"
                execution_log = result.stdout + "\n" + result.stderr

                try:
                    fixed_code = fix_code_callback(
                        current_code,
                        error_msg,
                        error_type,
                        execution_log
                    )
                    if fixed_code and fixed_code != current_code:
                        current_code = fixed_code
                except Exception as e:
                    # If fixing fails, continue with original code
                    pass

                # Wait before retry (exponential backoff)
                time.sleep(min(2 ** attempt, 10))  # Cap at 10 seconds

        # Return last result (failed)
        return last_result

    def _execute_script(
        self,
        script_path: str,
        timeout: Optional[int] = None
    ) -> ExecutionResult:
        """Execute script using subprocess.

        Args:
            script_path: Path to Python script
            timeout: Optional timeout in seconds (override; default uses self.timeout)

        Returns:
            ExecutionResult
        """
        if not os.path.exists(script_path):
            return ExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"Script file does not exist: {script_path}",
                execution_time=0.0,
                error_type="file_not_found"
            )

        cmd = f"python -u {script_path}"
        use_timeout = timeout if timeout is not None else self.timeout

        try:
            completed = subprocess.run(
                cmd,
                shell=True,
                cwd=str(self.work_dir),
                timeout=use_timeout,
                capture_output=True,
                text=True
            )
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            success = completed.returncode == 0
            if success and not stdout and stderr:
                stdout = stderr
                stderr = ""
            return ExecutionResult(
                success=success,
                return_code=completed.returncode,
                stdout=stdout,
                stderr=stderr,
                execution_time=0.0,
                error_type=None
            )
        except subprocess.TimeoutExpired as e:
            if e.process:
                e.process.kill()
            return ExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"Execution timeout after {use_timeout} seconds",
                execution_time=float(use_timeout),
                error_type="timeout"
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"Execution error: {str(e)}",
                execution_time=0.0,
                error_type="execution_error"
            )

    def validate_syntax(self, code: str) -> tuple[bool, list[str]]:
        """Validate Python code syntax.

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        import ast
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as e:
            return False, [f"Syntax error at line {e.lineno}: {e.msg}"]
        except Exception as e:
            return False, [f"Validation error: {str(e)}"]
