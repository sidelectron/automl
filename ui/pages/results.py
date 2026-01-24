"""Page 5: Results - Winner and business impact."""

import streamlit as st
import json


def render():
    """Render the results page."""
    st.header("🏆 Results & Business Impact")

    if not st.session_state.training_results:
        st.warning("Please complete 'Experiments' page first")
        return

    # Generate comparison if not done
    if not st.session_state.comparison:
        with st.spinner("Comparing results and selecting winner..."):
            try:
                orchestrator = st.session_state.orchestrator
                comparison = orchestrator.compare_results()
                st.session_state.comparison = comparison
                st.success("Comparison complete!")

            except Exception as e:
                st.error(f"Error during comparison: {e}")
                st.exception(e)

    if st.session_state.comparison:
        comparison = st.session_state.comparison
        winner = comparison.get("winner", {})
        business_impact = comparison.get("business_impact", {})
        financial_impact = comparison.get("financial_impact", {})

        # Winner display
        st.subheader("🥇 Winner")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Strategy", winner.get("strategy_name", "N/A"))
        with col2:
            st.metric("Model", winner.get("model_name", "N/A"))
        with col3:
            st.metric("Threshold", winner.get("threshold", "N/A"))

        # Business impact
        st.subheader("📊 Business Impact")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("True Positives", business_impact.get("true_positives", 0))
        with col2:
            st.metric("False Positives", business_impact.get("false_positives", 0))
        with col3:
            st.metric("False Negatives", business_impact.get("false_negatives", 0))
        with col4:
            st.metric("Total Actions", business_impact.get("total_actions", 0))

        explanation = business_impact.get("explanation", "")
        if explanation:
            st.info(explanation)

        # Financial impact
        st.subheader("💰 Financial Impact")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Potential Value", f"${financial_impact.get('potential_value', 0):,.0f}")
        with col2:
            st.metric("Total Cost", f"${financial_impact.get('total_cost', 0):,.0f}")
        with col3:
            st.metric("Net Value", f"${financial_impact.get('net_value', 0):,.0f}")
        with col4:
            st.metric("ROI", f"{financial_impact.get('roi', 0):.1f}x")

        # Interactive threshold adjustment
        st.subheader("🎚️ Interactive Threshold Adjustment")
        current_threshold = winner.get("threshold", 0.5)
        new_threshold = st.slider(
            "Adjust Threshold",
            min_value=0.1,
            max_value=0.9,
            value=float(current_threshold),
            step=0.05,
            help="Adjust the decision threshold and see how it affects business metrics"
        )

        if new_threshold != current_threshold:
            st.info(f"Threshold adjusted to {new_threshold}. Business metrics would be recalculated in a full implementation.")

        # Comparison text
        comparison_text = comparison.get("comparison_text", "")
        if comparison_text:
            st.subheader("📝 Detailed Explanation")
            st.markdown(comparison_text)

        # Alternatives
        alternatives = comparison.get("alternatives", [])
        if alternatives:
            st.subheader("🔀 Alternative Options")
            for alt in alternatives:
                with st.expander(f"{alt.get('model_name')} - Threshold {alt.get('threshold')}"):
                    bm = alt.get("business_metrics", {})
                    if bm:
                        st.metric("Net Value", f"${bm.get('net_value', 0):,.0f}")
                        st.metric("ROI", f"{bm.get('roi', 0):.1f}x")

    else:
        st.info("Comparison will be generated automatically when you navigate to this page.")
