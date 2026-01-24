"""Page 2: Understanding - Display profile and EDA."""

import streamlit as st
import json
from pathlib import Path


def render():
    """Render the understanding page."""
    st.header("📊 Understanding Your Data")

    if not st.session_state.intent:
        st.warning("Please complete 'Upload & Intent' page first")
        return

    if st.button("Generate Profile & EDA", type="primary"):
        with st.spinner("Analyzing data and generating insights..."):
            try:
                orchestrator = st.session_state.orchestrator

                # Profile data
                dataset_path = st.session_state.intent.get("dataset_path")
                if not dataset_path:
                    st.error("Dataset path not found")
                    return

                profile = orchestrator.profile_data(dataset_path)
                st.session_state.profile = profile

                # Generate EDA
                eda = orchestrator.generate_eda()
                st.session_state.eda = eda

                st.success("Analysis complete!")

            except Exception as e:
                st.error(f"Error during analysis: {e}")
                st.exception(e)

    # Display profile
    if st.session_state.profile:
        st.subheader("Data Profile")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Rows", st.session_state.profile.get("statistics", {}).get("shape", [0, 0])[0])
            st.metric("Columns", st.session_state.profile.get("statistics", {}).get("shape", [0, 0])[1])

        with col2:
            missing_total = sum(st.session_state.profile.get("missing_values", {}).values())
            st.metric("Total Missing Values", missing_total)

        # Intent flags
        intent_flags = st.session_state.profile.get("intent_flags", [])
        if intent_flags:
            st.subheader("Intent-Aware Warnings")
            for flag in intent_flags:
                severity = flag.get("severity", "info")
                if severity == "high":
                    st.error(f"⚠️ {flag.get('message')}")
                elif severity == "medium":
                    st.warning(f"⚠️ {flag.get('message')}")
                else:
                    st.info(f"ℹ️ {flag.get('message')}")

        # Insights
        insights = st.session_state.profile.get("insights", [])
        if insights:
            st.subheader("Insights")
            for insight in insights:
                st.markdown(f"- {insight}")

    # Display EDA
    if st.session_state.eda:
        st.subheader("Exploratory Data Analysis")

        insights = st.session_state.eda.get("insights", [])
        if insights:
            st.markdown("### Key Insights")
            for insight in insights:
                st.markdown(f"- {insight}")

        # Visualizations (if available)
        visualizations = st.session_state.eda.get("visualizations", [])
        if visualizations:
            st.markdown("### Visualizations")
            st.info("Visualizations will be displayed here when plot generation is complete")
