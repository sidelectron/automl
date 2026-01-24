"""Project structure generator."""

from pathlib import Path
from typing import Dict, Any, Optional


class ProjectStructure:
    """Generate project directory structure."""

    def __init__(self, base_dir: str = "data/generated_projects"):
        """Initialize project structure generator.

        Args:
            base_dir: Base directory for generated projects
        """
        self.base_dir = Path(base_dir)

    def create_structure(
        self,
        project_name: str,
        experiment_id: Optional[str] = None
    ) -> Path:
        """Create project directory structure.

        Args:
            project_name: Name of the project
            experiment_id: Optional experiment ID

        Returns:
            Path to project root
        """
        if experiment_id:
            project_dir = self.base_dir / f"{project_name}_{experiment_id[:8]}"
        else:
            project_dir = self.base_dir / project_name

        # Create directories
        (project_dir / "data" / "processed").mkdir(parents=True, exist_ok=True)
        (project_dir / "models").mkdir(parents=True, exist_ok=True)
        (project_dir / "src").mkdir(parents=True, exist_ok=True)

        return project_dir
