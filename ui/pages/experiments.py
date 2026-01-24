"""Page 4: Experiments - Training progress."""

import streamlit as st
import time


def render():
    """Render the experiments page."""
    st.header("⚙️ Training Experiments")

    if not st.session_state.strategies:
        st.warning("Please complete 'Strategy' page first")
        return

    if not st.session_state.selected_strategies:
        st.warning("Please select strategies in the 'Strategy' page")
        return

    # Check if training should start
    if st.session_state.get("training_started", False) and not st.session_state.training_results:
        with st.spinner("Training models... This may take a while."):
            try:
                orchestrator = st.session_state.orchestrator
                dataset_path = st.session_state.intent.get("dataset_path")

                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Simulate progress (in real implementation, this would come from orchestrator)
                for i in range(100):
                    time.sleep(0.1)
                    progress_bar.progress(i + 1)
                    status_text.text(f"Training progress: {i + 1}%")

                # Train models
                results = orchestrator.train_models(st.session_state.selected_strategies)
                st.session_state.training_results = results

                progress_bar.progress(100)
                status_text.text("Training complete!")
                st.success("Training completed successfully!")

            except Exception as e:
                st.error(f"Error during training: {e}")
                st.exception(e)

    # Display results
    if st.session_state.training_results:
        st.subheader("Training Results")

        st.markdown(f"**Total Results**: {len(st.session_state.training_results)}")

        # Show summary table
        import pandas as pd

        results_data = []
        for result in st.session_state.training_results:
            results_data.append({
                "Strategy": result.get("strategy_name", "N/A"),
                "Model": result.get("model_name", "N/A"),
                "Threshold": result.get("threshold", "N/A"),
                "F1": result.get("metrics", {}).get("f1", 0),
                "Recall": result.get("metrics", {}).get("recall", 0),
                "Precision": result.get("metrics", {}).get("precision", 0),
                "Net Value": result.get("business_metrics", {}).get("net_value", 0) if result.get("business_metrics") else "N/A",
                "ROI": result.get("business_metrics", {}).get("roi", 0) if result.get("business_metrics") else "N/A"
            })

        df = pd.DataFrame(results_data)
        st.dataframe(df, use_container_width=True)

        st.info("Training complete! Proceed to 'Results' page to see the winner and business impact.")

    else:
        st.info("Training will start automatically when you navigate to this page after selecting strategies.")
