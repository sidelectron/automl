"""Main project generator."""

from pathlib import Path
from typing import Dict, Any, Optional

from .structure import ProjectStructure
from .code_generator import CodeGenerator
from .documentation import DocumentationGenerator


class ProjectGenerator:
    """Generate complete Python project from results."""

    def __init__(self, base_dir: str = "data/generated_projects"):
        """Initialize project generator.

        Args:
            base_dir: Base directory for generated projects
        """
        self.base_dir = Path(base_dir)
        self.structure_gen = ProjectStructure(base_dir)

    def generate(
        self,
        intent: Dict[str, Any],
        winner: Dict[str, Any],
        comparison: Dict[str, Any],
        experiment_id: Optional[str] = None
    ) -> str:
        """Generate complete project.

        Args:
            intent: Parsed intent
            winner: Winner dictionary with model and strategy info
            comparison: Comparison results
            experiment_id: Optional experiment ID

        Returns:
            Path to generated project directory
        """
        # Create project structure
        project_name = intent.get("target_variable", "ml_project").replace(" ", "_").lower()
        project_dir = self.structure_gen.create_structure(project_name, experiment_id)

        # Generate code files
        code_gen = CodeGenerator(project_dir)

        # Get strategy from winner (would need to be passed or retrieved)
        strategy = {
            "name": winner.get("strategy_name", "unknown"),
            "preprocessing_steps": []  # Would need to retrieve from version store
        }

        code_gen.generate_preprocessing(
            strategy,
            winner.get("preprocessing_path")
        )

        code_gen.generate_train(
            winner,
            winner.get("model_path")
        )

        code_gen.generate_predict(
            winner,
            intent
        )

        code_gen.generate_requirements()

        # Generate documentation
        doc_gen = DocumentationGenerator(project_dir)
        doc_gen.generate_readme(intent, winner, comparison)

        return str(project_dir)
