"""Page 6: History - Past experiments."""

import streamlit as st
import pandas as pd
from datetime import datetime


def render():
    """Render the history page."""
    st.header("📜 Experiment History")

    try:
        orchestrator = st.session_state.orchestrator
        version_store = orchestrator.version_store

        # List experiments
        experiments = version_store.list_experiments(limit=50)

        if experiments:
            st.subheader("Past Experiments")

            # Create DataFrame
            exp_data = []
            for exp in experiments:
                exp_data.append({
                    "Experiment ID": exp["experiment_id"][:8] + "...",
                    "Timestamp": exp["timestamp"],
                    "Description": exp.get("description", "N/A")[:50] + "..."
                })

            df = pd.DataFrame(exp_data)
            st.dataframe(df, use_container_width=True)

            # Select experiment to view
            selected_exp_id = st.selectbox(
                "Select experiment to view details",
                [exp["experiment_id"] for exp in experiments]
            )

            if selected_exp_id:
                # Get experiment details
                experiment = version_store.get_experiment(selected_exp_id)
                if experiment:
                    st.subheader("Experiment Details")
                    st.json(experiment)

                    # Get results
                    query_manager = orchestrator.query_manager
                    results = query_manager.get_experiment_results(selected_exp_id)

                    if results:
                        st.subheader("Results")
                        results_df = pd.DataFrame([
                            {
                                "Strategy": r.get("strategy_name"),
                                "Model": r.get("model_name"),
                                "Threshold": r.get("threshold"),
                                "F1": r.get("metrics", {}).get("f1", 0),
                                "Net Value": r.get("business_metrics", {}).get("net_value", 0) if r.get("business_metrics") else "N/A"
                            }
                            for r in results
                        ])
                        st.dataframe(results_df, use_container_width=True)

                    # Get comparison
                    comparison = query_manager.get_comparison(selected_exp_id)
                    if comparison:
                        st.subheader("Comparison")
                        st.json(comparison)

        else:
            st.info("No past experiments found. Complete a full pipeline to see history here.")

    except Exception as e:
        st.error(f"Error loading history: {e}")
        st.exception(e)
