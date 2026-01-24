"""Code validator for security and import checks."""

import ast
import re
from typing import List, Tuple, Set


class CodeValidator:
    """Validate generated code for security and correctness."""

    # Allowed imports
    ALLOWED_IMPORTS = {
        "sklearn", "pandas", "numpy", "xgboost", "lightgbm",
        "imblearn", "pathlib", "pickle", "json", "os", "sys",
        "warnings", "math", "collections", "datetime", "typing"
    }

    # Dangerous operations to detect (open excluded: pickle save/load requires it)
    DANGEROUS_OPERATIONS = {
        "eval", "exec", "compile", "__import__",
        "file", "input", "raw_input"
    }

    # Dangerous modules
    DANGEROUS_MODULES = {
        "subprocess", "os.system", "os.popen", "shutil",
        "socket", "urllib", "requests", "http"
    }

    def __init__(self):
        """Initialize code validator."""
        pass

    def validate(
        self,
        code: str
    ) -> Tuple[bool, List[str]]:
        """Validate code for syntax, imports, and security.

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # 1. Syntax validation
        syntax_valid, syntax_errors = self.validate_syntax(code)
        if not syntax_valid:
            errors.extend(syntax_errors)
            return False, errors

        # 2. Import validation
        import_valid, import_errors = self.validate_imports(code)
        if not import_valid:
            errors.extend(import_errors)

        # 3. Security validation
        security_valid, security_errors = self.validate_security(code)
        if not security_valid:
            errors.extend(security_errors)

        is_valid = len(errors) == 0
        return is_valid, errors

    def validate_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """Validate Python syntax.

        Args:
            code: Python code

        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as e:
            return False, [f"Syntax error at line {e.lineno}: {e.msg}"]
        except Exception as e:
            return False, [f"Parse error: {str(e)}"]

    def validate_imports(self, code: str) -> Tuple[bool, List[str]]:
        """Validate that only allowed imports are used.

        Args:
            code: Python code

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split(".")[0]
                    if module_name not in self.ALLOWED_IMPORTS:
                        errors.append(
                            f"Disallowed import: {module_name}. "
                            f"Allowed: {', '.join(sorted(self.ALLOWED_IMPORTS))}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split(".")[0]
                    if module_name not in self.ALLOWED_IMPORTS:
                        errors.append(
                            f"Disallowed import from: {module_name}. "
                            f"Allowed: {', '.join(sorted(self.ALLOWED_IMPORTS))}"
                        )

        return len(errors) == 0, errors

    def validate_security(self, code: str) -> Tuple[bool, List[str]]:
        """Validate code for dangerous operations.

        Args:
            code: Python code

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        tree = ast.parse(code)

        # Check for dangerous function calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in self.DANGEROUS_OPERATIONS:
                        errors.append(
                            f"Dangerous operation detected: {func_name}()"
                        )
                elif isinstance(node.func, ast.Attribute):
                    # Check for os.system, subprocess.call, etc.
                    if isinstance(node.func.value, ast.Name):
                        module_name = node.func.value.id
                        attr_name = node.func.attr
                        full_name = f"{module_name}.{attr_name}"
                        if full_name in self.DANGEROUS_MODULES:
                            errors.append(
                                f"Dangerous operation detected: {full_name}()"
                            )

        # Check for eval/exec in string literals (basic check)
        if "eval(" in code or "exec(" in code:
            errors.append("Dangerous operation detected: eval() or exec()")

        return len(errors) == 0, errors

    def get_imports(self, code: str) -> Set[str]:
        """Extract all imports from code.

        Args:
            code: Python code

        Returns:
            Set of imported module names
        """
        imports = set()
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])

        return imports
