"""Main Streamlit app for Intent-Driven AutoML Platform."""

import streamlit as st
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.ollama_provider import OllamaProvider
from src.agents.orchestrator import Orchestrator
from src.orchestrator_v2 import OrchestratorV2
from src.version_store import VersionStore


def initialize_session_state():
    """Initialize session state variables."""
    # Version selection
    if "version" not in st.session_state:
        st.session_state.version = "hybrid"  # Default to hybrid
    
    if "orchestrator" not in st.session_state:
        # Initialize LLM provider
        llm_provider = OllamaProvider(model="qwen2.5-coder:3b")
        
        # Initialize orchestrator based on version
        version_store = VersionStore()
        if st.session_state.version == "dynamic":
            orchestrator = OrchestratorV2(llm_provider, version_store)
        else:
            orchestrator = Orchestrator(llm_provider, version_store)
        st.session_state.orchestrator = orchestrator

    if "intent" not in st.session_state:
        st.session_state.intent = None

    if "profile" not in st.session_state:
        st.session_state.profile = None

    if "eda" not in st.session_state:
        st.session_state.eda = None

    if "strategies" not in st.session_state:
        st.session_state.strategies = None

    if "selected_strategies" not in st.session_state:
        st.session_state.selected_strategies = None

    if "training_results" not in st.session_state:
        st.session_state.training_results = None

    if "comparison" not in st.session_state:
        st.session_state.comparison = None

    if "experiment_id" not in st.session_state:
        st.session_state.experiment_id = None


def main():
    """Main app function."""
    st.set_page_config(
        page_title="Intent-Driven AutoML Platform",
        page_icon="🤖",
        layout="wide"
    )

    initialize_session_state()

    st.title("🤖 Intent-Driven AutoML Platform")
    st.markdown("**An AI-powered AutoML platform that understands your business goals**")

    # Version selector in sidebar
    st.sidebar.title("Configuration")
    version = st.sidebar.radio(
        "AutoML Version",
        ["Hybrid (Recommended)", "Fully Dynamic (Experimental)"],
        help="Hybrid: JSON strategies → sklearn Pipeline. Dynamic: LLM generates Python code",
        index=0 if st.session_state.get("version", "hybrid") == "hybrid" else 1
    )
    
    # Update version if changed
    new_version = "dynamic" if "Dynamic" in version else "hybrid"
    if new_version != st.session_state.get("version"):
        st.session_state.version = new_version
        # Reinitialize orchestrator
        llm_provider = OllamaProvider(model="qwen2.5-coder:3b")
        version_store = VersionStore()
        if new_version == "dynamic":
            st.session_state.orchestrator = OrchestratorV2(llm_provider, version_store)
        else:
            st.session_state.orchestrator = Orchestrator(llm_provider, version_store)
        # Clear previous state
        for key in ["intent", "profile", "eda", "strategies", "selected_strategies", 
                   "training_results", "comparison", "experiment_id"]:
            if key in st.session_state:
                del st.session_state[key]

    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Different pages for different versions
    if st.session_state.version == "dynamic":
        pages = [
            "1. Upload & Intent",
            "2. Understanding",
            "3. Plan",
            "4. Code Generation",
            "5. Execution",
            "6. Results",
            "7. History"
        ]
    else:
        pages = [
            "1. Upload & Intent",
            "2. Understanding",
            "3. Strategy",
            "4. Experiments",
            "5. Results",
            "6. History"
        ]
    
    page = st.sidebar.radio("Go to", pages)

    # Route to appropriate page
    if page == "1. Upload & Intent":
        from ui.pages import upload_intent
        upload_intent.render()
    elif page == "2. Understanding":
        from ui.pages import understanding
        understanding.render()
    elif page == "3. Strategy" or page == "3. Plan":
        if st.session_state.version == "dynamic":
            from ui.pages_v2 import plan
            plan.render()
        else:
            from ui.pages import strategy
            strategy.render()
    elif page == "4. Experiments" or page == "4. Code Generation":
        if st.session_state.version == "dynamic":
            from ui.pages_v2 import code_generation
            code_generation.render()
        else:
            from ui.pages import experiments
            experiments.render()
    elif page == "5. Results" or page == "5. Execution":
        if st.session_state.version == "dynamic":
            from ui.pages_v2 import execution
            execution.render()
        else:
            from ui.pages import results
            results.render()
    elif page == "6. Results" or page == "6. History" or page == "7. History":
        if st.session_state.version == "dynamic" and page == "6. Results":
            from ui.pages_v2 import results as results_v2
            results_v2.render()
        else:
            from ui.pages import history
            history.render()


if __name__ == "__main__":
    main()
