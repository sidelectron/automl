"""
Streamlit UI for Intent-Driven AutoML Platform.

Run with: streamlit run app.py
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Page config
st.set_page_config(
    page_title="AutoML Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "df" not in st.session_state:
        st.session_state.df = None
    if "profile" not in st.session_state:
        st.session_state.profile = None
    if "intent" not in st.session_state:
        st.session_state.intent = None
    if "results" not in st.session_state:
        st.session_state.results = None
    if "pipeline_version" not in st.session_state:
        st.session_state.pipeline_version = "V1"
    if "dataset_path" not in st.session_state:
        st.session_state.dataset_path = None


def load_data_page():
    """Data upload and preview page."""
    st.markdown('<p class="main-header">📊 Data Upload</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload your dataset to begin analysis</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a CSV file for analysis"
        )

        # Or use sample dataset
        st.markdown("**Or use a sample dataset:**")
        sample_datasets = {
            "None": None,
            "Titanic": project_root / "archive" / "Titanic Dataset.csv",
            "Titanic (alt)": project_root / "archive" / "The Titanic dataset.csv",
        }

        # Filter to only existing files
        available_samples = {k: v for k, v in sample_datasets.items()
                           if v is None or (v and Path(v).exists())}

        sample_choice = st.selectbox(
            "Sample Dataset",
            options=list(available_samples.keys()),
            index=0
        )

    with col2:
        st.info("""
        **Supported formats:**
        - CSV files
        - UTF-8, Latin-1 encoding

        **Recommended size:**
        - Up to 100MB
        - Up to 1M rows
        """)

    # Load data
    df = None
    dataset_path = None

    if uploaded_file is not None:
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if df is not None:
                # Save to temp file for pipeline
                temp_path = project_root / "temp_upload.csv"
                df.to_csv(temp_path, index=False)
                dataset_path = str(temp_path)
                st.success(f"Loaded {len(df):,} rows x {len(df.columns)} columns")
        except Exception as e:
            st.error(f"Error loading file: {e}")

    elif sample_choice != "None" and available_samples[sample_choice]:
        try:
            dataset_path = str(available_samples[sample_choice])
            df = pd.read_csv(dataset_path)
            st.success(f"Loaded {sample_choice}: {len(df):,} rows x {len(df.columns)} columns")
        except Exception as e:
            st.error(f"Error loading sample: {e}")

    # Store in session
    if df is not None:
        st.session_state.df = df
        st.session_state.dataset_path = dataset_path

        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Numeric", len(df.select_dtypes(include=[np.number]).columns))
        with col4:
            st.metric("Missing", f"{df.isnull().sum().sum():,}")

        # Column info
        with st.expander("Column Details"):
            col_info = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.astype(str),
                "Non-Null": df.count().values,
                "Missing": df.isnull().sum().values,
                "Unique": df.nunique().values
            })
            st.dataframe(col_info, use_container_width=True)


def profile_page():
    """Data profiling page."""
    st.markdown('<p class="main-header">🔍 Data Profile</p>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state.df

    # Profile the data
    if st.button("Run Profiler", type="primary"):
        with st.spinner("Profiling data..."):
            profile = generate_profile(df)
            st.session_state.profile = profile

    if st.session_state.profile:
        profile = st.session_state.profile

        # Overview
        st.subheader("Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{profile.get('row_count', 0):,}")
        with col2:
            st.metric("Total Columns", profile.get('column_count', 0))
        with col3:
            dup_count = profile.get('duplicates', {}).get('count', 0)
            st.metric("Duplicate Rows", f"{dup_count:,}")

        # Data types
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Types")
            dtype_counts = {}
            for col, dtype in profile.get('data_types', {}).items():
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

            if dtype_counts:
                fig = px.pie(
                    values=list(dtype_counts.values()),
                    names=list(dtype_counts.keys()),
                    title="Column Types Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Missing Values")
            missing = profile.get('missing_values', {})
            if missing:
                missing_df = pd.DataFrame([
                    {"Column": k, "Missing": v, "Percentage": f"{v/len(df)*100:.1f}%"}
                    for k, v in missing.items() if v > 0
                ])
                if not missing_df.empty:
                    st.dataframe(missing_df, use_container_width=True)
                else:
                    st.success("No missing values!")
            else:
                st.success("No missing values!")

        # Numeric statistics
        st.subheader("Numeric Column Statistics")
        numeric_summary = profile.get('numeric_summary', {})
        if numeric_summary:
            stats_data = []
            for col, stats in numeric_summary.items():
                stats_data.append({
                    "Column": col,
                    "Mean": f"{stats.get('mean', 0):.2f}",
                    "Std": f"{stats.get('std', 0):.2f}",
                    "Min": f"{stats.get('min', 0):.2f}",
                    "Max": f"{stats.get('max', 0):.2f}",
                    "Skewness": f"{stats.get('skewness', 0):.2f}",
                })
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

        # Outliers
        outliers = profile.get('outliers', {})
        if outliers:
            st.subheader("Outlier Detection (IQR Method)")
            outlier_data = []
            for col, info in outliers.items():
                if info.get('count', 0) > 0:
                    outlier_data.append({
                        "Column": col,
                        "Outliers": info.get('count', 0),
                        "Percentage": f"{info.get('percentage', 0):.1f}%",
                        "Lower Bound": f"{info.get('lower_bound', 0):.2f}",
                        "Upper Bound": f"{info.get('upper_bound', 0):.2f}"
                    })
            if outlier_data:
                st.dataframe(pd.DataFrame(outlier_data), use_container_width=True)
            else:
                st.success("No significant outliers detected!")

        # Correlations
        correlations = profile.get('correlations', {})
        if correlations:
            st.subheader("Feature Correlations")
            corr_matrix = pd.DataFrame(correlations)
            fig = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Correlation Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)


def generate_profile(df: pd.DataFrame) -> dict:
    """Generate data profile."""
    profile = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "data_types": {},
        "missing_values": {},
        "numeric_summary": {},
        "categorical_summary": {},
        "duplicates": {"count": int(df.duplicated().sum())},
        "outliers": {},
        "correlations": {}
    }

    # Data types
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            profile["data_types"][col] = "numeric"
        else:
            profile["data_types"][col] = "categorical"

    # Missing values
    profile["missing_values"] = df.isnull().sum().to_dict()

    # Numeric summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        profile["numeric_summary"][col] = {
            "mean": float(df[col].mean()) if not df[col].isnull().all() else 0,
            "std": float(df[col].std()) if not df[col].isnull().all() else 0,
            "min": float(df[col].min()) if not df[col].isnull().all() else 0,
            "max": float(df[col].max()) if not df[col].isnull().all() else 0,
            "skewness": float(df[col].skew()) if not df[col].isnull().all() else 0,
            "kurtosis": float(df[col].kurtosis()) if not df[col].isnull().all() else 0,
        }

        # Outlier detection (IQR)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)][col]
        profile["outliers"][col] = {
            "count": len(outliers),
            "percentage": len(outliers) / len(df) * 100,
            "lower_bound": float(lower),
            "upper_bound": float(upper)
        }

    # Correlations
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        profile["correlations"] = corr.to_dict()

    # Categorical summary
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        profile["categorical_summary"][col] = {
            "unique": int(df[col].nunique()),
            "top_values": df[col].value_counts().head(5).to_dict()
        }

    return profile


def visualize_page():
    """Visualization page."""
    st.markdown('<p class="main-header">📈 Visualizations</p>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state.df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    # Plot type selection
    col1, col2 = st.columns([1, 2])

    with col1:
        plot_type = st.selectbox(
            "Plot Type",
            ["Histogram", "Box Plot", "Scatter Plot", "Correlation Heatmap",
             "Bar Chart", "Violin Plot", "Pair Plot"]
        )

    with col2:
        if plot_type == "Histogram":
            column = st.selectbox("Select Column", numeric_cols)
            if column:
                fig = px.histogram(df, x=column, marginal="box", title=f"Distribution of {column}")
                st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Box Plot":
            column = st.selectbox("Select Column", numeric_cols)
            group_by = st.selectbox("Group By (optional)", ["None"] + cat_cols)
            if column:
                if group_by != "None":
                    fig = px.box(df, x=group_by, y=column, title=f"{column} by {group_by}")
                else:
                    fig = px.box(df, y=column, title=f"Box Plot of {column}")
                st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Scatter Plot":
            col_a, col_b = st.columns(2)
            with col_a:
                x_col = st.selectbox("X Axis", numeric_cols)
            with col_b:
                y_col = st.selectbox("Y Axis", numeric_cols, index=min(1, len(numeric_cols)-1))
            color_by = st.selectbox("Color By (optional)", ["None"] + cat_cols)

            if x_col and y_col:
                if color_by != "None":
                    fig = px.scatter(df, x=x_col, y=y_col, color=color_by,
                                    title=f"{x_col} vs {y_col}")
                else:
                    fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Correlation Heatmap":
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                               color_continuous_scale="RdBu_r",
                               title="Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for correlation heatmap.")

        elif plot_type == "Bar Chart":
            column = st.selectbox("Select Column", cat_cols if cat_cols else all_cols)
            if column:
                value_counts = df[column].value_counts().head(20)
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                            title=f"Value Counts of {column}",
                            labels={"x": column, "y": "Count"})
                st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Violin Plot":
            column = st.selectbox("Select Column", numeric_cols)
            group_by = st.selectbox("Group By (optional)", ["None"] + cat_cols)
            if column:
                if group_by != "None":
                    fig = px.violin(df, x=group_by, y=column, box=True,
                                   title=f"{column} by {group_by}")
                else:
                    fig = px.violin(df, y=column, box=True,
                                   title=f"Violin Plot of {column}")
                st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Pair Plot":
            selected_cols = st.multiselect(
                "Select Columns (max 5)",
                numeric_cols,
                default=numeric_cols[:min(4, len(numeric_cols))]
            )
            if selected_cols and len(selected_cols) >= 2:
                if len(selected_cols) > 5:
                    st.warning("Showing first 5 columns only.")
                    selected_cols = selected_cols[:5]
                fig = px.scatter_matrix(df[selected_cols], title="Pair Plot")
                st.plotly_chart(fig, use_container_width=True)


def intent_page():
    """Intent configuration page."""
    st.markdown('<p class="main-header">🎯 Define Your Task</p>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state.df

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Task Description")
        user_input = st.text_area(
            "Describe what you want to analyze",
            placeholder="e.g., Predict which passengers survived the Titanic disaster based on their characteristics.",
            height=100
        )

        task_type = st.selectbox(
            "Task Type",
            ["binary_classification", "multiclass_classification", "regression"],
            help="Select the type of ML task"
        )

        target_variable = st.selectbox(
            "Target Variable",
            df.columns.tolist(),
            help="Select the column you want to predict"
        )

    with col2:
        st.subheader("Business Context")
        priority_metric = st.selectbox(
            "Priority Metric",
            ["f1", "accuracy", "precision", "recall", "roc_auc"],
            help="The primary metric to optimize"
        )

        st.markdown("**Cost/Value Configuration (for classification)**")
        col_a, col_b = st.columns(2)
        with col_a:
            true_positive_value = st.number_input("True Positive Value ($)", value=100, step=10)
        with col_b:
            false_positive_cost = st.number_input("False Positive Cost ($)", value=50, step=10)

    # Pipeline version
    st.subheader("Pipeline Version")
    pipeline_version = st.radio(
        "Select Pipeline",
        ["V1 - Hybrid (JSON Strategy)", "V2 - Dynamic (LLM-Generated Code)"],
        help="V1 uses predefined transformers, V2 generates Python code dynamically"
    )
    st.session_state.pipeline_version = "V1" if "V1" in pipeline_version else "V2"

    # Save intent
    if st.button("Save Configuration", type="primary"):
        intent = {
            "user_input": user_input,
            "task_type": task_type,
            "target_variable": target_variable,
            "business_context": {
                "priority_metric": priority_metric,
                "true_positive_value": true_positive_value,
                "false_positive_cost": false_positive_cost
            },
            "dataset_path": st.session_state.dataset_path
        }
        st.session_state.intent = intent
        st.success("Configuration saved!")

        # Show summary
        st.json(intent)


def train_page():
    """Model training page."""
    st.markdown('<p class="main-header">🚀 Train Models</p>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return

    if st.session_state.intent is None:
        st.warning("Please configure your task intent first.")
        return

    intent = st.session_state.intent
    df = st.session_state.df

    # Show configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Task Type", intent["task_type"])
    with col2:
        st.metric("Target", intent["target_variable"])
    with col3:
        st.metric("Pipeline", st.session_state.pipeline_version)

    st.markdown("---")

    # LLM Configuration
    st.subheader("LLM Configuration")
    llm_model = st.text_input("Ollama Model", value="qwen2.5-coder:3b",
                              help="Make sure Ollama is running with this model")

    # Run training
    if st.button("Start Training", type="primary"):
        try:
            run_pipeline(intent, df, llm_model)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())


def run_pipeline(intent: dict, df: pd.DataFrame, llm_model: str):
    """Run the ML pipeline."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Import modules
        status_text.text("Loading modules...")
        progress_bar.progress(10)

        from src.llm.ollama_provider import OllamaProvider
        from src.version_store import VersionStore

        # Initialize LLM
        status_text.text("Initializing LLM provider...")
        progress_bar.progress(20)

        llm = OllamaProvider(model=llm_model)
        store = VersionStore()

        if st.session_state.pipeline_version == "V1":
            from src.agents.orchestrator import Orchestrator
            status_text.text("Running V1 Hybrid Pipeline...")
            progress_bar.progress(30)

            orch = Orchestrator(llm, store)

            # Run pipeline
            status_text.text("Parsing intent...")
            progress_bar.progress(40)

            results = orch.run_full_pipeline(
                user_input=intent.get("user_input", ""),
                dataset_path=intent.get("dataset_path", ""),
                selected_strategies=None  # V1 doesn't use work_dir, uses selected_strategies
            )

        else:  # V2
            from src.orchestrator_v2 import OrchestratorV2
            status_text.text("Running V2 Dynamic Pipeline...")
            progress_bar.progress(30)

            orch = OrchestratorV2(llm, store)

            results = orch.run_full_pipeline(
                user_input=intent.get("user_input", ""),
                dataset_path=intent.get("dataset_path", ""),
                work_dir=str(project_root)
            )

        progress_bar.progress(90)
        status_text.text("Processing results...")

        # Store results
        st.session_state.results = results

        progress_bar.progress(100)
        status_text.text("Complete!")

        # Show results
        if "error" in results:
            st.error(f"Pipeline Error: {results['error']}")
        else:
            st.success("Training completed successfully!")
            display_results(results)

    except Exception as e:
        st.error(f"Error: {e}")
        raise


def display_results(results: dict):
    """Display training results."""
    st.subheader("Results")

    # Model comparison
    if "comparison" in results and results["comparison"]:
        comp = results["comparison"]
        winner = comp.get("winner", {})

        # Display winner information
        if winner:
            st.subheader("🥇 Winner")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Model", winner.get("model_name", "N/A"))
            with col2:
                st.metric("Strategy", winner.get("strategy_name", "N/A"))
            with col3:
                threshold = winner.get("threshold", 0.5)
                if threshold is not None:
                    st.metric("Threshold", f"{threshold:.2f}")
                else:
                    st.metric("Threshold", "N/A")
            with col4:
                winner_metrics = winner.get("metrics", {})
                f1_score = winner_metrics.get("f1", 0)
                st.metric("F1 Score", f"{f1_score:.4f}")

            # Winner metrics
            if winner_metrics:
                st.markdown("**Winner Metrics:**")
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                with metrics_col1:
                    st.metric("Accuracy", f"{winner_metrics.get('accuracy', 0):.4f}")
                with metrics_col2:
                    st.metric("Precision", f"{winner_metrics.get('precision', 0):.4f}")
                with metrics_col3:
                    st.metric("Recall", f"{winner_metrics.get('recall', 0):.4f}")
                with metrics_col4:
                    st.metric("F1", f"{winner_metrics.get('f1', 0):.4f}")

            # Business metrics
            business_metrics = winner.get("business_metrics", {})
            if business_metrics:
                st.markdown("**Business Impact:**")
                biz_col1, biz_col2, biz_col3, biz_col4 = st.columns(4)
                with biz_col1:
                    st.metric("Net Value", f"${business_metrics.get('net_value', 0):.2f}")
                with biz_col2:
                    roi = business_metrics.get('roi', 0)
                    if roi == float('inf') or roi == float('-inf'):
                        st.metric("ROI", "∞")
                    else:
                        st.metric("ROI", f"{roi:.2f}%")
                with biz_col3:
                    st.metric("Total Value", f"${business_metrics.get('total_value', 0):.2f}")
                with biz_col4:
                    st.metric("Total Cost", f"${business_metrics.get('total_cost', 0):.2f}")

            # Business impact explanation
            if "business_impact" in comp:
                business_impact = comp.get("business_impact", {})
                if isinstance(business_impact, dict):
                    explanation = business_impact.get("explanation", "")
                    if explanation:
                        st.info(f"💡 {explanation}")

            # Comparison text
            if "comparison_text" in comp:
                comparison_text = comp.get("comparison_text", "")
                if comparison_text:
                    st.markdown("**Detailed Analysis:**")
                    st.markdown(comparison_text)

        # Fallback for best_result if winner not available
        elif "best_result" in comp:
            best = comp["best_result"]
            st.metric("Best F1 Score", f"{best.get('metrics', {}).get('f1', 0):.4f}")

        # Results table
        if "all_results" in comp:
            st.subheader("Model Comparison")
            results_data = []
            chart_data = []  # Separate data for chart with numeric values
            for r in comp["all_results"]:
                metrics = r.get("metrics", {})
                model_name = r.get("model_name", "")
                threshold = r.get("threshold", 0.5)
                
                # Data for display (formatted strings)
                results_data.append({
                    "Model": model_name,
                    "Threshold": f"{threshold:.2f}" if threshold is not None else "N/A",
                    "Accuracy": f"{metrics.get('accuracy', 0):.4f}",
                    "F1": f"{metrics.get('f1', 0):.4f}",
                    "Precision": f"{metrics.get('precision', 0):.4f}",
                    "Recall": f"{metrics.get('recall', 0):.4f}"
                })
                
                # Data for chart (numeric values)
                chart_data.append({
                    "Model": model_name,
                    "F1": float(metrics.get('f1', 0)),
                    "Accuracy": float(metrics.get('accuracy', 0)),
                    "Precision": float(metrics.get('precision', 0)),
                    "Recall": float(metrics.get('recall', 0))
                })

            if results_data:
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)

                # Chart with numeric values
                if chart_data:
                    chart_df = pd.DataFrame(chart_data)
                    fig = px.bar(
                        chart_df, x="Model", y="F1",
                        title="Model Performance (F1 Score)",
                        color="F1",
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # Generated code (V2)
    if "generated_code" in results and results["generated_code"]:
        with st.expander("Generated Code"):
            st.code(results["generated_code"], language="python")

    # Full results JSON
    with st.expander("Full Results JSON"):
        st.json(results)


def results_page():
    """Results page."""
    st.markdown('<p class="main-header">📋 Results</p>', unsafe_allow_html=True)

    if st.session_state.results is None:
        st.warning("No results yet. Please run the training pipeline first.")
        return

    display_results(st.session_state.results)


def main():
    """Main app."""
    init_session_state()

    # Sidebar navigation
    st.sidebar.title("🤖 AutoML Platform")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["📊 Data Upload", "🔍 Data Profile", "📈 Visualizations",
         "🎯 Define Task", "🚀 Train Models", "📋 Results"]
    )

    st.sidebar.markdown("---")

    # Status indicators
    st.sidebar.subheader("Status")
    if st.session_state.df is not None:
        st.sidebar.success(f"Data: {len(st.session_state.df):,} rows")
    else:
        st.sidebar.warning("No data loaded")

    if st.session_state.intent is not None:
        st.sidebar.success(f"Task: {st.session_state.intent.get('task_type', 'N/A')}")
    else:
        st.sidebar.warning("Task not configured")

    if st.session_state.results is not None:
        st.sidebar.success("Results available")

    st.sidebar.markdown("---")
    st.sidebar.caption("Built with Streamlit + Ollama")

    # Page routing
    if "Data Upload" in page:
        load_data_page()
    elif "Data Profile" in page:
        profile_page()
    elif "Visualizations" in page:
        visualize_page()
    elif "Define Task" in page:
        intent_page()
    elif "Train Models" in page:
        train_page()
    elif "Results" in page:
        results_page()


if __name__ == "__main__":
    main()
