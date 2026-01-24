"""Page 5: Execution - Show code execution progress and results."""

import streamlit as st
import time


def render():
    """Render the execution page."""
    st.header("⚙️ Code Execution")

    if not st.session_state.get("generated_code"):
        st.warning("Please complete 'Code Generation' page first")
        return

    # Execution controls
    col1, col2 = st.columns(2)
    with col1:
        execute_preprocessing = st.button("Execute Preprocessing", type="primary")
    with col2:
        execute_training = st.button("Execute Training", type="primary")

    # Execute preprocessing
    if execute_preprocessing:
        with st.spinner("Executing preprocessing code..."):
            try:
                orchestrator = st.session_state.orchestrator
                result = orchestrator.execute_code("preprocessing")
                st.session_state.preprocessing_result = result

                if result["success"]:
                    st.success("Preprocessing executed successfully!")
                    st.text_area("Output", value=result["stdout"], height=200)
                else:
                    st.error("Preprocessing execution failed!")
                    st.text_area("Error", value=result["stderr"], height=200)

            except Exception as e:
                st.error(f"Error executing preprocessing: {e}")
                st.exception(e)

    # Execute training
    if execute_training:
        with st.spinner("Executing training code..."):
            try:
                orchestrator = st.session_state.orchestrator
                result = orchestrator.execute_code("training")
                st.session_state.training_result = result

                if result["success"]:
                    st.success("Training executed successfully!")
                    st.text_area("Output", value=result["stdout"], height=200)
                else:
                    st.error("Training execution failed!")
                    st.text_area("Error", value=result["stderr"], height=200)
                    st.info("The system will automatically retry with error fixes")

            except Exception as e:
                st.error(f"Error executing training: {e}")
                st.exception(e)

    # Show execution results
    if st.session_state.get("preprocessing_result"):
        st.subheader("📊 Preprocessing Results")
        result = st.session_state.preprocessing_result
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", "Success" if result["success"] else "Failed")
        with col2:
            st.metric("Execution Time", f"{result['execution_time']:.2f}s")
        with col3:
            st.metric("Return Code", result["return_code"])

    if st.session_state.get("training_result"):
        st.subheader("📊 Training Results")
        result = st.session_state.training_result
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", "Success" if result["success"] else "Failed")
        with col2:
            st.metric("Execution Time", f"{result['execution_time']:.2f}s")
        with col3:
            st.metric("Return Code", result["return_code"])

        # Continue button if both executed successfully
        if (st.session_state.get("preprocessing_result", {}).get("success") and
            st.session_state.get("training_result", {}).get("success")):
            st.markdown("---")
            if st.button("➡️ Continue to Results", type="primary", use_container_width=True):
                st.session_state.execution_complete = True
                st.rerun()

    else:
        st.info("Click buttons above to execute generated code")
