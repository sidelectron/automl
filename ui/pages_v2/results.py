"""Page 6: Results - Show results from executed code."""

import streamlit as st


def render():
    """Render the results page."""
    st.header("🏆 Results & Business Impact")

    if not st.session_state.get("training_result"):
        st.warning("Please complete 'Execution' page first")
        return

    # Generate comparison if not done
    if not st.session_state.get("comparison"):
        with st.spinner("Comparing results and selecting winner..."):
            try:
                orchestrator = st.session_state.orchestrator
                comparison = orchestrator.compare_results()
                st.session_state.comparison = comparison
                st.success("Comparison complete!")

            except Exception as e:
                st.error(f"Error during comparison: {e}")
                st.exception(e)

    if st.session_state.get("comparison"):
        comparison = st.session_state.comparison

        if comparison.get("error"):
            st.error(comparison["error"])
            if "execution_results" in comparison:
                st.json(comparison["execution_results"])
        else:
            winner = comparison.get("winner", {})
            business_impact = comparison.get("business_impact", {})

            # Winner display
            st.subheader("🥇 Winner")
            if winner:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Code Type", winner.get("code_type", "N/A"))
                with col2:
                    st.metric("Status", "Success" if winner.get("success") else "Failed")

            # Business impact
            if business_impact:
                st.subheader("📊 Business Impact")
                explanation = business_impact.get("explanation", "")
                if explanation:
                    st.info(explanation)

            # Comparison text
            comparison_text = comparison.get("comparison_text", "")
            if comparison_text:
                st.subheader("📝 Detailed Explanation")
                st.markdown(comparison_text)

            # Execution output
            training_result = st.session_state.get("training_result", {})
            if training_result.get("stdout"):
                st.subheader("📄 Execution Output")
                st.text_area("Output", value=training_result["stdout"], height=300)

    else:
        st.info("Comparison will be generated automatically when you navigate to this page.")
