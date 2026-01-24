"""Page 4: Code Generation - Show generated Python code."""

import streamlit as st


def render():
    """Render the code generation page."""
    st.header("💻 Generated Python Code")

    if not st.session_state.get("preprocessing_plan"):
        st.warning("Please complete 'Plan' page first")
        return

    if st.button("Generate Code", type="primary"):
        with st.spinner("Generating Python code from plans..."):
            try:
                orchestrator = st.session_state.orchestrator
                code = orchestrator.generate_code()
                st.session_state.generated_code = code
                st.success("Code generated successfully!")

            except Exception as e:
                st.error(f"Error generating code: {e}")
                st.exception(e)

    # Display generated code
    if st.session_state.get("generated_code"):
        code = st.session_state.generated_code

        # Preprocessing code
        if "preprocessing" in code:
            st.subheader("🔧 Preprocessing Code")
            st.code(code["preprocessing"], language="python")

        # Training code
        if "training" in code:
            st.subheader("🤖 Training Code")
            st.code(code["training"], language="python")

        # Prediction code
        if "prediction" in code:
            st.subheader("🔮 Prediction Code")
            st.code(code["prediction"], language="python")

        # Continue button
        st.markdown("---")
        if st.button("➡️ Continue to Execution", type="primary", use_container_width=True):
            st.session_state.code_generated = True
            st.rerun()

    else:
        st.info("Click 'Generate Code' to generate Python code from plans")
