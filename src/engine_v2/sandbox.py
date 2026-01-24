"""Sandbox environment for safe code execution."""

from pathlib import Path
from typing import Optional
import tempfile
import shutil


class Sandbox:
    """Isolated execution environment."""

    def __init__(
        self,
        base_dir: Optional[str] = None,
        cleanup: bool = True
    ):
        """Initialize sandbox.

        Args:
            base_dir: Base directory for sandbox (default: temp directory)
            cleanup: Whether to cleanup on exit
        """
        if base_dir:
            self.work_dir = Path(base_dir)
            self.work_dir.mkdir(parents=True, exist_ok=True)
            self.temp_dir = None
        else:
            self.temp_dir = tempfile.mkdtemp(prefix="automl_sandbox_")
            self.work_dir = Path(self.temp_dir)

        self.cleanup = cleanup

    def get_work_dir(self) -> Path:
        """Get working directory.

        Returns:
            Path to working directory
        """
        return self.work_dir

    def create_subdir(self, name: str) -> Path:
        """Create subdirectory in sandbox.

        Args:
            name: Subdirectory name

        Returns:
            Path to subdirectory
        """
        subdir = self.work_dir / name
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir

    def cleanup(self):
        """Cleanup sandbox directory."""
        if self.cleanup and self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.cleanup:
            self.cleanup()
