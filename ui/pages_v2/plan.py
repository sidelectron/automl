"""Page 3: Plan - Show text-based plans for dynamic version."""

import streamlit as st


def render():
    """Render the plan page."""
    st.header("📋 Preprocessing & Modeling Plans")

    if not st.session_state.profile:
        st.warning("Please complete 'Understanding' page first")
        return

    if st.button("Generate Plans", type="primary"):
        with st.spinner("Generating text-based plans..."):
            try:
                orchestrator = st.session_state.orchestrator
                plans = orchestrator.generate_plans()
                st.session_state.preprocessing_plan = plans["preprocessing_plan"]
                st.session_state.modeling_plan = plans["modeling_plan"]
                st.success("Plans generated successfully!")

            except Exception as e:
                st.error(f"Error generating plans: {e}")
                st.exception(e)

    # Display preprocessing plan
    if st.session_state.get("preprocessing_plan"):
        st.subheader("🔧 Preprocessing Plan")
        st.text_area(
            "Preprocessing Instructions",
            value=st.session_state.preprocessing_plan,
            height=300,
            disabled=True,
            key="preprocessing_plan_display"
        )

    # Display modeling plan
    if st.session_state.get("modeling_plan"):
        st.subheader("🤖 Modeling Plan")
        st.text_area(
            "Modeling Instructions",
            value=st.session_state.modeling_plan,
            height=300,
            disabled=True,
            key="modeling_plan_display"
        )

    # Continue button
    if st.session_state.get("preprocessing_plan") and st.session_state.get("modeling_plan"):
        st.markdown("---")
        if st.button("➡️ Continue to Code Generation", type="primary", use_container_width=True):
            st.session_state.plans_generated = True
            st.rerun()

    else:
        st.info("Click 'Generate Plans' to see preprocessing and modeling plans")
