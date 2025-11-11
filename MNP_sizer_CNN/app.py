"""
StepReaderCNN - Streamlit GUI Application
Main entry point for the web-based GUI interface.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import GUI components
from gui.data_viewer import DataViewer
from gui.upload_handler import UploadHandler
from gui.training_page import TrainingPage
from gui.evaluation_page import EvaluationPage
from gui.synthetic_page import SyntheticDataPage

# Page configuration
st.set_page_config(
    page_title="StepReaderCNN - Electrochemical Sensor Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    </style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point."""

    # Header
    st.markdown('<p class="main-header">🔬 StepReaderCNN</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">CNN-based Electrochemical Sensor Signal Processing</p>',
                unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Module",
        ["📊 Data Explorer", "🧠 Model Training", "📈 Evaluation", "🔬 Synthetic Data", "⚙️ Settings"],
        index=0
    )

    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'data_summary' not in st.session_state:
        st.session_state.data_summary = None

    # Route to appropriate page
    if page == "📊 Data Explorer":
        show_data_explorer()
    elif page == "🧠 Model Training":
        show_model_training()
    elif page == "📈 Evaluation":
        show_evaluation()
    elif page == "🔬 Synthetic Data":
        show_synthetic_data()
    elif page == "⚙️ Settings":
        show_settings()


def show_data_explorer():
    """Display the data exploration interface."""
    st.header("Data Explorer")

    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["📁 Upload Data", "📊 Dataset Overview", "🔍 Signal Viewer"])

    with tab1:
        st.subheader("Upload Sensor Data")
        upload_handler = UploadHandler()
        upload_handler.render()

    with tab2:
        st.subheader("Dataset Overview")
        if st.session_state.data_loaded:
            data_viewer = DataViewer()
            data_viewer.render_dataset_overview()
        else:
            st.info("👆 Please upload data or load from TestData folder first.")

    with tab3:
        st.subheader("Interactive Signal Viewer")
        if st.session_state.data_loaded:
            data_viewer = DataViewer()
            data_viewer.render_signal_viewer()
        else:
            st.info("👆 Please upload data or load from TestData folder first.")


def show_model_training():
    """Display the model training interface (Phase 5)."""
    training_page = TrainingPage()
    training_page.render()


def show_evaluation():
    """Display the evaluation interface (Phase 7)."""
    evaluation_page = EvaluationPage()
    evaluation_page.render()


def show_synthetic_data():
    """Display the synthetic data generation interface (Phase 8)."""
    synthetic_page = SyntheticDataPage()
    synthetic_page.render()


def show_settings():
    """Display settings and configuration."""
    st.header("⚙️ Settings")

    st.subheader("Project Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Data Settings")
        st.text_input("Default Data Directory", value="TestData", disabled=True)
        st.selectbox("Data Format", ["CSV", "NPY", "HDF5"], index=0)
        st.number_input("Target Sequence Length", value=10000, step=1000)

    with col2:
        st.markdown("#### Visualization Settings")
        st.selectbox("Plot Theme", ["Plotly", "Matplotlib", "Seaborn"], index=0)
        st.color_picker("Primary Color", "#1f77b4")
        st.checkbox("Show Grid Lines", value=True)

    st.markdown("---")

    st.subheader("System Information")

    # Display system info
    import torch
    import numpy as np
    import pandas as pd

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.metric("PyTorch Version", torch.__version__)
        st.metric("CUDA Available", "Yes" if torch.cuda.is_available() else "No")

    with info_col2:
        st.metric("NumPy Version", np.__version__)
        st.metric("Pandas Version", pd.__version__)

    with info_col3:
        st.metric("Streamlit Version", st.__version__)
        st.metric("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    if st.button("🔄 Reset All Settings"):
        st.session_state.clear()
        st.success("Settings reset successfully!")
        st.rerun()


if __name__ == "__main__":
    main()
