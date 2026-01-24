"""Page 1: Upload dataset and describe intent."""

import streamlit as st
import json
from pathlib import Path


def render():
    """Render the upload and intent page."""
    st.header("📤 Upload Dataset & Describe Intent")

    st.markdown("""
    Upload your dataset and describe your business goal. The system will parse your intent
    and extract business context (costs, values, priorities).
    """)

    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="Upload your dataset in CSV format"
    )

    if uploaded_file is not None:
        # Save uploaded file
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / uploaded_file.name

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"File uploaded: {uploaded_file.name}")
        st.session_state.dataset_path = str(file_path)

    # Intent description
    st.subheader("Describe Your Business Goal")
    intent_text = st.text_area(
        "Intent Description",
        height=150,
        placeholder="""Example: I want to predict customer churn. Catching churners is my top priority 
because a saved customer is worth $500, but each retention call costs $20.""",
        help="Describe your machine learning task and business context"
    )

    if st.button("Parse Intent", type="primary"):
        if not intent_text:
            st.error("Please provide an intent description")
            return

        if "dataset_path" not in st.session_state:
            st.error("Please upload a dataset first")
            return

        with st.spinner("Parsing intent..."):
            try:
                orchestrator = st.session_state.orchestrator
                intent = orchestrator.parse_intent(intent_text, st.session_state.dataset_path)
                st.session_state.intent = intent
                st.session_state.experiment_id = orchestrator.experiment_id

                st.success("Intent parsed successfully!")

                # Display parsed intent
                st.subheader("Parsed Intent")
                st.json(intent)

                # Show business context
                if "business_context" in intent:
                    bc = intent["business_context"]
                    st.info(f"""
                    **Priority Metric**: {bc.get('priority_metric', 'N/A')}  
                    **True Positive Value**: ${bc.get('true_positive_value', 0):,.0f}  
                    **False Positive Cost**: ${bc.get('false_positive_cost', 0):,.0f}  
                    **Cost Ratio**: {bc.get('cost_ratio', 0):.4f}
                    """)

            except Exception as e:
                st.error(f"Error parsing intent: {e}")
                st.exception(e)

    # Show current intent if available
    if st.session_state.intent:
        st.subheader("Current Intent")
        st.json(st.session_state.intent)
