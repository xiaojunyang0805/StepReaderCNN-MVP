"""
Evaluation GUI Page
Interactive evaluation and prediction interface.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import torch

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from evaluation.evaluator import ModelEvaluator, load_all_trained_models
from data.data_loader import SensorDataLoader


class EvaluationPage:
    """Evaluation and prediction interface."""

    def __init__(self):
        """Initialize evaluation page."""
        # Initialize session state
        if 'loaded_models' not in st.session_state:
            st.session_state.loaded_models = {}

        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = None

        if 'prediction_results' not in st.session_state:
            st.session_state.prediction_results = None

    def render(self):
        """Render the evaluation page."""
        st.header("Model Evaluation & Prediction")

        # Check if models exist
        models_dir = Path("outputs/trained_models")

        if not models_dir.exists() or not list(models_dir.glob("*.pth")):
            st.warning("No trained models found. Please train models first in the Model Training tab.")
            return

        # Evaluation tabs
        tab1, tab2, tab3 = st.tabs(["🔍 Model Selection", "🎯 Prediction", "📊 Model Comparison"])

        with tab1:
            self._render_model_selection()

        with tab2:
            self._render_prediction_interface()

        with tab3:
            self._render_model_comparison()

    def _render_model_selection(self):
        """Render model selection and loading interface."""
        st.subheader("Load Trained Models")

        models_dir = Path("outputs/trained_models")

        # Find available models
        available_models = [p.stem for p in models_dir.glob("*.pth")]

        if not available_models:
            st.info("No trained models found.")
            return

        st.markdown(f"**Available Models**: {len(available_models)}")

        # Load all models button
        col1, col2 = st.columns([1, 3])

        with col1:
            if st.button("Load All Models", type="primary"):
                with st.spinner("Loading models..."):
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    st.session_state.loaded_models = load_all_trained_models(
                        str(models_dir),
                        device=device
                    )
                st.success(f"Loaded {len(st.session_state.loaded_models)} models!")
                st.rerun()

        with col2:
            if st.session_state.loaded_models:
                st.info(f"✓ {len(st.session_state.loaded_models)} models loaded in memory")

        # Display loaded models
        if st.session_state.loaded_models:
            st.markdown("---")
            st.markdown("**Loaded Models**")

            for model_name, evaluator in st.session_state.loaded_models.items():
                with st.expander(f"📦 {model_name}", expanded=False):
                    info = evaluator.get_model_info()

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Model Class", info['model_class'])
                        st.metric("Parameters", f"{info['num_parameters']:,}")

                    with col2:
                        if 'training_results' in info:
                            results = info['training_results']
                            st.metric("Best Val Acc", f"{results.get('best_val_acc', 0):.2%}")
                            st.metric("Test Accuracy", f"{results.get('test_accuracy', 0):.2%}")

                    with col3:
                        st.metric("Classes", info['num_classes'])
                        st.metric("Device", info['device'].upper())

                    # Class mapping
                    st.markdown("**Class Mapping**:")
                    st.json(info['label_mapping'])

                    # Select this model for prediction
                    if st.button(f"Use for Prediction", key=f"select_{model_name}"):
                        st.session_state.selected_model = model_name
                        st.success(f"Selected {model_name} for prediction")
                        st.rerun()

    def _render_prediction_interface(self):
        """Render prediction interface."""
        st.subheader("Make Predictions")

        if not st.session_state.loaded_models:
            st.warning("Please load models first in the Model Selection tab.")
            return

        # Model selection
        model_names = list(st.session_state.loaded_models.keys())

        if st.session_state.selected_model and st.session_state.selected_model in model_names:
            default_idx = model_names.index(st.session_state.selected_model)
        else:
            default_idx = 0

        selected_model = st.selectbox(
            "Select Model",
            model_names,
            index=default_idx,
            help="Choose a model for prediction"
        )

        st.session_state.selected_model = selected_model
        evaluator = st.session_state.loaded_models[selected_model]

        st.info(f"Using **{selected_model}** ({evaluator.model_class})")

        # Prediction source
        st.markdown("---")
        st.markdown("**Data Source**")

        source = st.radio(
            "Choose data source",
            ["Load from TestData", "Upload CSV File"],
            horizontal=True
        )

        if source == "Load from TestData":
            self._render_testdata_prediction(evaluator)
        else:
            self._render_upload_prediction(evaluator)

    def _render_testdata_prediction(self, evaluator):
        """Render prediction interface for TestData samples."""
        # Check if dataset is loaded
        if 'dataset' not in st.session_state or st.session_state.dataset is None:
            # Load TestData
            if st.button("Load TestData"):
                with st.spinner("Loading TestData..."):
                    data_dir = Path("TestData")
                    loader = SensorDataLoader(str(data_dir))
                    st.session_state.dataset = loader.load_dataset("*.csv")
                st.success("TestData loaded!")
                st.rerun()
            return

        dataset = st.session_state.dataset

        # Select class and sample
        col1, col2 = st.columns(2)

        with col1:
            selected_class = st.selectbox(
                "Select Class",
                sorted(dataset.keys()),
                help="Choose a class to predict"
            )

        with col2:
            num_samples = len(dataset[selected_class])
            sample_idx = st.selectbox(
                "Select Sample",
                range(num_samples),
                format_func=lambda x: f"Sample {x+1}",
                help="Choose a sample from the class"
            )

        # Get signal
        item = dataset[selected_class][sample_idx]
        time_data, current_data = item[0], item[1]
        filename = item[2] if len(item) > 2 else "Unknown"

        # Display signal info
        st.markdown(f"**File**: `{filename}`")
        st.markdown(f"**True Class**: `{selected_class}`")
        st.markdown(f"**Signal Length**: {len(current_data):,} points")

        # Predict button
        if st.button("🎯 Predict", type="primary", use_container_width=True):
            with st.spinner("Making prediction..."):
                result = evaluator.predict(time_data, current_data)
                st.session_state.prediction_results = result

                # Display results
                self._display_prediction_results(result, selected_class)

                # Plot signal with prediction
                self._plot_signal_with_prediction(
                    time_data, current_data, result, selected_class
                )

    def _render_upload_prediction(self, evaluator):
        """Render prediction interface for uploaded files."""
        st.markdown("Upload a CSV file with time-series data")

        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="CSV file with two columns: time, current"
        )

        if uploaded_file is not None:
            try:
                # Load data
                import pandas as pd
                df = pd.read_csv(uploaded_file)

                if df.shape[1] < 2:
                    st.error("CSV must have at least 2 columns (time, current)")
                    return

                time_data = df.iloc[:, 0].values
                current_data = df.iloc[:, 1].values

                st.success(f"Loaded {len(current_data):,} data points")

                # Plot uploaded signal
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=time_data,
                    y=current_data,
                    mode='lines',
                    name='Uploaded Signal',
                    line=dict(color='blue', width=1)
                ))

                fig.update_layout(
                    title='Uploaded Signal',
                    xaxis_title='Time (ms)',
                    yaxis_title='Current',
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

                # Predict button
                if st.button("🎯 Predict", type="primary", use_container_width=True):
                    with st.spinner("Making prediction..."):
                        result = evaluator.predict(time_data, current_data)
                        st.session_state.prediction_results = result

                        # Display results
                        self._display_prediction_results(result, None)

            except Exception as e:
                st.error(f"Error loading file: {e}")

    def _display_prediction_results(self, result, true_class=None):
        """Display prediction results."""
        st.markdown("---")
        st.markdown("### Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Predicted Class", result['predicted_class'])

        with col2:
            confidence = result['confidence']
            st.metric("Confidence", f"{confidence:.2%}")

        with col3:
            if true_class:
                correct = result['predicted_class'] == true_class
                st.metric(
                    "Result",
                    "✓ Correct" if correct else "✗ Incorrect",
                    delta=None,
                    delta_color="normal" if correct else "inverse"
                )

        # Probabilities bar chart
        st.markdown("**Class Probabilities**")

        probabilities = result['probabilities']
        prob_df = pd.DataFrame({
            'Class': list(probabilities.keys()),
            'Probability': list(probabilities.values())
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=prob_df['Class'],
            y=prob_df['Probability'],
            marker_color=['green' if c == result['predicted_class'] else 'lightblue'
                         for c in prob_df['Class']],
            text=[f"{p:.1%}" for p in prob_df['Probability']],
            textposition='outside'
        ))

        fig.update_layout(
            xaxis_title='Class',
            yaxis_title='Probability',
            yaxis_range=[0, 1],
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    def _plot_signal_with_prediction(self, time_data, current_data, result, true_class=None):
        """Plot signal with prediction overlay."""
        st.markdown("**Signal Visualization**")

        fig = go.Figure()

        # Plot signal
        fig.add_trace(go.Scatter(
            x=time_data,
            y=current_data,
            mode='lines',
            name='Signal',
            line=dict(color='blue', width=1)
        ))

        # Add prediction annotation
        pred_text = f"Predicted: {result['predicted_class']} ({result['confidence']:.1%})"
        if true_class:
            pred_text += f"<br>True: {true_class}"

        fig.add_annotation(
            text=pred_text,
            xref="paper", yref="paper",
            x=0.98, y=0.98,
            showarrow=False,
            bgcolor="white",
            bordercolor=("green" if true_class and result['predicted_class'] == true_class else "red"),
            borderwidth=2,
            xanchor="right",
            yanchor="top"
        )

        fig.update_layout(
            title='Signal with Prediction',
            xaxis_title='Time (ms)',
            yaxis_title='Current',
            height=500,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_model_comparison(self):
        """Render model comparison interface."""
        st.subheader("Model Comparison")

        if not st.session_state.loaded_models:
            st.warning("Please load models first in the Model Selection tab.")
            return

        if len(st.session_state.loaded_models) < 2:
            st.info("Load at least 2 models to compare.")
            return

        # Load test data
        if st.button("Run Comparison on TestData"):
            with st.spinner("Evaluating all models..."):
                # Load test data
                from data.data_split import stratified_split

                data_dir = Path("TestData")
                loader = SensorDataLoader(str(data_dir))
                dataset = loader.load_dataset("*.csv")

                _, _, test_data = stratified_split(dataset, seed=42)

                # Evaluate each model
                comparison_results = []

                for model_name, evaluator in st.session_state.loaded_models.items():
                    tracker = evaluator.evaluate_dataset(test_data)
                    metrics = tracker.compute_all_metrics()

                    comparison_results.append({
                        'Model': model_name,
                        'Model Class': evaluator.model_class,
                        'Parameters': f"{evaluator.get_model_info()['num_parameters']:,}",
                        'Accuracy': f"{metrics['accuracy']:.2%}",
                        'Macro F1': f"{metrics['precision_recall_f1']['macro']['f1']:.4f}",
                        'Macro Precision': f"{metrics['precision_recall_f1']['macro']['precision']:.4f}",
                        'Macro Recall': f"{metrics['precision_recall_f1']['macro']['recall']:.4f}"
                    })

                # Display comparison table
                st.markdown("### Comparison Results")
                df_comparison = pd.DataFrame(comparison_results)
                st.dataframe(df_comparison, use_container_width=True)

                # Plot comparison
                st.markdown("### Performance Comparison")

                # Extract numeric values for plotting
                models = [r['Model'] for r in comparison_results]
                accuracies = [float(r['Accuracy'].strip('%'))/100 for r in comparison_results]
                f1_scores = [float(r['Macro F1']) for r in comparison_results]

                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Test Accuracy', 'Macro F1-Score')
                )

                fig.add_trace(
                    go.Bar(x=models, y=accuracies, name='Accuracy',
                          marker_color='lightblue'),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Bar(x=models, y=f1_scores, name='F1-Score',
                          marker_color='lightgreen'),
                    row=1, col=2
                )

                fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=1, col=1)
                fig.update_yaxes(title_text="F1-Score", range=[0, 1], row=1, col=2)
                fig.update_layout(height=400, showlegend=False)

                st.plotly_chart(fig, use_container_width=True)

                st.success("Comparison complete!")
