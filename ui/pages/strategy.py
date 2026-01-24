"""Page 3: Strategy selection."""

import streamlit as st
import json


def render():
    """Render the strategy page."""
    st.header("🎯 Preprocessing Strategies")

    if not st.session_state.profile:
        st.warning("Please complete 'Understanding' page first")
        return

    if st.button("Generate Strategies", type="primary"):
        with st.spinner("Generating preprocessing strategies..."):
            try:
                orchestrator = st.session_state.orchestrator
                strategies = orchestrator.propose_strategies()
                st.session_state.strategies = strategies
                st.success(f"Generated {len(strategies)} strategies!")

            except Exception as e:
                st.error(f"Error generating strategies: {e}")
                st.exception(e)

    # Display strategies
    if st.session_state.strategies:
        st.subheader("Available Strategies")

        selected_strategies = []

        for idx, strategy in enumerate(st.session_state.strategies):
            strategy_name = strategy.get("name", f"Strategy {idx + 1}")

            with st.expander(f"📋 {strategy_name}", expanded=idx == 0):
                # Selection checkbox
                selected = st.checkbox(
                    f"Select {strategy_name}",
                    key=f"strategy_{idx}",
                    value=idx == 0  # Select first by default
                )

                if selected:
                    selected_strategies.append(strategy_name)

                # Strategy details
                st.markdown(f"**Rationale**: {strategy.get('rationale', 'N/A')}")

                # Preprocessing steps
                steps = strategy.get("preprocessing_steps", [])
                if steps:
                    st.markdown("**Preprocessing Steps:**")
                    for step in steps:
                        st.markdown(f"- {step.get('step_type')}: {step.get('method')}")

                # Model candidates
                models = strategy.get("model_candidates", [])
                if models:
                    st.markdown(f"**Model Candidates**: {', '.join(models)}")

        st.session_state.selected_strategies = selected_strategies

        # Start training button
        if selected_strategies:
            st.markdown("---")
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                st.session_state.training_started = True
                st.rerun()

    else:
        st.info("Click 'Generate Strategies' to see available preprocessing strategies")
