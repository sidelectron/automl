"""EDA Agent for generating target-focused insights."""

import json
from typing import Dict, Any, Optional
from pathlib import Path

import pandas as pd

from ..llm.llm_interface import LLMInterface
from ..visualization import PlotGenerator


class EDAAgent:
    """Agent for generating target-focused EDA."""

    def __init__(self, llm_provider: LLMInterface):
        """Initialize EDA Agent.

        Args:
            llm_provider: LLM provider instance
        """
        self.llm = llm_provider
        self.agent_name = "eda"
        self.plot_generator = PlotGenerator()

    def _load_prompt_template(self) -> str:
        """Load the EDA generation prompt template.

        Returns:
            Prompt template string
        """
        prompt_path = Path(__file__).parent.parent / "llm" / "prompts" / "eda_generation.txt"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return "Generate target-focused EDA insights."

    def generate(
        self,
        profile: Dict[str, Any],
        intent: Dict[str, Any],
        dataset_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate target-focused EDA.

        Args:
            profile: Data profile from Profiler Agent
            intent: Parsed intent from Intent Agent
            dataset_path: Optional path to dataset for generating plots

        Returns:
            EDA results dictionary with visualizations and insights
        """
        system_prompt = self._load_prompt_template()

        target_variable = intent.get("target_variable")
        task_type = intent.get("task_type")

        user_prompt = f"""Generate target-focused EDA for the following dataset profile and intent.

# Data Profile #
```json
{json.dumps(profile, indent=2)}
```

# User Intent #
```json
{json.dumps(intent, indent=2)}
```

Generate visualizations and insights focused on the target variable: {target_variable}
All analysis should relate to understanding {target_variable} and help achieve the business goal.
"""

        try:
            llm_response = self.llm.generate_json(
                prompt=system_prompt + "\n\n" + user_prompt,
                temperature=0.2,
                agent=self.agent_name,
                timeout=180
            )
        except Exception as e:
            # Fallback: generate basic EDA
            llm_response = {
                "visualizations": [],
                "insights": ["Basic EDA analysis complete."]
            }

        # Generate actual plots if dataset provided
        plots = []
        if dataset_path:
            try:
                df = pd.read_csv(dataset_path)
                visualizations = llm_response.get("visualizations", [])
                if not isinstance(visualizations, list):
                    visualizations = []

                for viz_spec in visualizations[:5]:  # Limit to 5 plots
                    try:
                        fig = self.plot_generator.generate_plot(
                            df,
                            viz_spec,
                            target_variable
                        )
                        # Convert to JSON for storage
                        plots.append({
                            "spec": viz_spec,
                            "plot_json": fig.to_json()
                        })
                    except Exception as e:
                        print(f"Error generating plot: {e}")
                        continue
            except Exception as e:
                print(f"Error loading dataset for plots: {e}")

        return {
            "visualizations": plots,
            "insights": llm_response.get("insights", []),
            "target_variable": target_variable,
            "task_type": task_type
        }
