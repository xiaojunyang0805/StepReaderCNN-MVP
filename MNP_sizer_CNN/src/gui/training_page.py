"""
Training GUI Page
Interactive training interface with real-time monitoring.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import torch
import time
from typing import Dict, Optional

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from models.cnn_models import SimpleCNN1D, ResNet1D, MultiScaleCNN1D, count_parameters
from data.dataset import create_dataloaders
from data.data_split import stratified_split, create_class_weights
from training.trainer import create_trainer
from training.metrics import evaluate_model, MetricsTracker


class TrainingPage:
    """Training interface with configuration and monitoring."""

    def __init__(self):
        """Initialize training page."""
        # Initialize session state
        if 'training_state' not in st.session_state:
            st.session_state.training_state = 'idle'  # idle, training, completed

        if 'training_history' not in st.session_state:
            st.session_state.training_history = None

        if 'trained_model' not in st.session_state:
            st.session_state.trained_model = None

        if 'training_config' not in st.session_state:
            st.session_state.training_config = {}

    def render(self):
        """Render the training page."""
        st.header("Model Training")

        # Check if dataset is loaded
        if 'dataset' not in st.session_state or st.session_state.dataset is None:
            st.warning("Please load a dataset first in the Data Explorer.")
            return

        # Training tabs
        tab1, tab2, tab3 = st.tabs(["⚙️ Configuration", "📈 Training Monitor", "📊 Results"])

        with tab1:
            self._render_configuration()

        with tab2:
            self._render_training_monitor()

        with tab3:
            self._render_results()

    def _render_configuration(self):
        """Render training configuration interface."""
        st.subheader("Training Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Model Settings**")

            model_name = st.selectbox(
                "Model Architecture",
                ['SimpleCNN1D', 'ResNet1D', 'MultiScaleCNN1D'],
                help="Choose CNN architecture"
            )

            base_filters = st.select_slider(
                "Base Filters",
                options=[16, 32, 64, 128],
                value=32,
                help="Number of filters in first layer"
            )

            dropout = st.slider(
                "Dropout Rate",
                min_value=0.0,
                max_value=0.8,
                value=0.5,
                step=0.1,
                help="Dropout probability for regularization"
            )

        with col2:
            st.markdown("**Training Settings**")

            batch_size = st.selectbox(
                "Batch Size",
                [4, 8, 16, 32, 64],
                index=2,
                help="Number of samples per batch"
            )

            num_epochs = st.number_input(
                "Number of Epochs",
                min_value=1,
                max_value=200,
                value=50,
                help="Training epochs"
            )

            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                value=0.001,
                format_func=lambda x: f"{x:.4f}",
                help="Optimizer learning rate"
            )

        # Advanced settings
        with st.expander("Advanced Settings"):
            col3, col4 = st.columns(2)

            with col3:
                early_stopping = st.checkbox(
                    "Early Stopping",
                    value=True,
                    help="Stop training if validation doesn't improve"
                )

                patience = st.number_input(
                    "Patience",
                    min_value=3,
                    max_value=50,
                    value=10,
                    help="Epochs to wait before stopping",
                    disabled=not early_stopping
                )

                use_class_weights = st.checkbox(
                    "Use Class Weights",
                    value=True,
                    help="Balance classes with weighted loss"
                )

            with col4:
                target_length = st.number_input(
                    "Signal Length",
                    min_value=1000,
                    max_value=50000,
                    value=10000,
                    step=1000,
                    help="Target signal length"
                )

                normalize_method = st.selectbox(
                    "Normalization",
                    ['zscore', 'minmax', 'robust'],
                    help="Signal normalization method"
                )

                augment_train = st.checkbox(
                    "Data Augmentation",
                    value=True,
                    help="Apply augmentation to training data"
                )

        # Data split configuration
        st.markdown("**Data Split**")
        col5, col6, col7 = st.columns(3)

        with col5:
            train_ratio = st.slider("Train %", 60, 85, 70, 5)
        with col6:
            val_ratio = st.slider("Validation %", 10, 25, 15, 5)
        with col7:
            test_ratio = 100 - train_ratio - val_ratio
            st.metric("Test %", f"{test_ratio}%")

        # Device selection
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        st.info(f"Training device: **{device.upper()}**")

        if device == 'cpu':
            st.warning("GPU not available. Training will be slower on CPU.")

        # Start training button
        st.markdown("---")

        if st.session_state.training_state == 'training':
            st.warning("Training in progress...")
            if st.button("Stop Training", type="secondary"):
                st.session_state.training_state = 'idle'
                st.rerun()
        else:
            if st.button("Start Training", type="primary", use_container_width=True):
                # Save configuration
                config = {
                    'model_name': model_name,
                    'base_filters': base_filters,
                    'dropout': dropout,
                    'batch_size': batch_size,
                    'num_epochs': num_epochs,
                    'learning_rate': learning_rate,
                    'early_stopping': early_stopping,
                    'patience': patience if early_stopping else None,
                    'use_class_weights': use_class_weights,
                    'target_length': target_length,
                    'normalize_method': normalize_method,
                    'augment_train': augment_train,
                    'train_ratio': train_ratio / 100,
                    'val_ratio': val_ratio / 100,
                    'test_ratio': test_ratio / 100,
                    'device': device
                }

                st.session_state.training_config = config
                self._start_training(config)

    def _start_training(self, config: Dict):
        """Start model training."""
        try:
            # Show progress
            progress_container = st.container()

            with progress_container:
                st.info("Initializing training...")
                progress_bar = st.progress(0)
                status_text = st.empty()

                # 1. Split data
                status_text.text("Splitting dataset...")
                progress_bar.progress(10)

                train_data, val_data, test_data = stratified_split(
                    st.session_state.dataset,
                    train_ratio=config['train_ratio'],
                    val_ratio=config['val_ratio'],
                    test_ratio=config['test_ratio'],
                    seed=42
                )

                # 2. Create label mapping
                status_text.text("Creating label mapping...")
                progress_bar.progress(20)

                unique_labels = sorted(st.session_state.dataset.keys())
                label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                st.session_state.label_to_idx = label_to_idx
                st.session_state.idx_to_label = {idx: label for label, idx in label_to_idx.items()}

                # 3. Create dataloaders
                status_text.text("Creating dataloaders...")
                progress_bar.progress(30)

                train_loader, val_loader, test_loader = create_dataloaders(
                    train_data, val_data, test_data, label_to_idx,
                    batch_size=config['batch_size'],
                    target_length=config['target_length'],
                    normalize_method=config['normalize_method'],
                    augment_train=config['augment_train'],
                    num_workers=0
                )

                st.session_state.test_loader = test_loader

                # 4. Create model
                status_text.text("Creating model...")
                progress_bar.progress(40)

                num_classes = len(unique_labels)

                if config['model_name'] == 'SimpleCNN1D':
                    model = SimpleCNN1D(
                        in_channels=1,
                        num_classes=num_classes,
                        base_filters=config['base_filters'],
                        dropout=config['dropout']
                    )
                elif config['model_name'] == 'ResNet1D':
                    model = ResNet1D(
                        in_channels=1,
                        num_classes=num_classes,
                        base_filters=config['base_filters'],
                        dropout=config['dropout']
                    )
                else:  # MultiScaleCNN1D
                    model = MultiScaleCNN1D(
                        in_channels=1,
                        num_classes=num_classes,
                        base_filters=config['base_filters'],
                        dropout=config['dropout']
                    )

                # 5. Calculate class weights
                class_weights = None
                if config['use_class_weights']:
                    status_text.text("Calculating class weights...")
                    progress_bar.progress(50)
                    class_weights_dict = create_class_weights(train_data, method='balanced')
                    # Convert to tensor (in order of label_to_idx)
                    class_weights = torch.tensor(
                        [class_weights_dict[label] for label in sorted(class_weights_dict.keys())],
                        dtype=torch.float32
                    )

                # 6. Create trainer
                status_text.text("Initializing trainer...")
                progress_bar.progress(60)

                trainer = create_trainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    num_classes=num_classes,
                    learning_rate=config['learning_rate'],
                    class_weights=class_weights,
                    device=config['device']
                )

                # 7. Train
                status_text.text("Training model...")
                progress_bar.progress(70)

                st.session_state.training_state = 'training'

                # Training callback for real-time updates
                epoch_placeholder = st.empty()

                def training_callback(event_type, info):
                    if event_type == 'epoch':
                        epoch_placeholder.text(
                            f"Epoch {info['epoch']+1}/{config['num_epochs']}: "
                            f"Train Loss={info['train_loss']:.4f}, "
                            f"Val Acc={info['val_acc']:.4f}"
                        )

                trainer.add_callback(training_callback)

                history = trainer.train(
                    num_epochs=config['num_epochs'],
                    early_stopping_patience=config['patience'],
                    verbose=True
                )

                # 8. Save results
                status_text.text("Saving results...")
                progress_bar.progress(90)

                st.session_state.training_history = history
                st.session_state.trained_model = model
                st.session_state.training_state = 'completed'

                # Done
                progress_bar.progress(100)
                status_text.empty()
                epoch_placeholder.empty()

                st.success(f"Training completed! Best validation accuracy: {trainer.best_val_acc:.4f}")
                st.rerun()

        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            st.session_state.training_state = 'idle'

    def _render_training_monitor(self):
        """Render real-time training monitor."""
        st.subheader("Training Monitor")

        if st.session_state.training_history is None:
            st.info("No training history available. Start training to see results.")
            return

        history = st.session_state.training_history

        # Metrics cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            final_train_loss = history['train_loss'][-1]
            st.metric("Final Train Loss", f"{final_train_loss:.4f}")

        with col2:
            final_train_acc = history['train_acc'][-1]
            st.metric("Final Train Acc", f"{final_train_acc:.4f}")

        with col3:
            final_val_loss = history['val_loss'][-1]
            best_val_loss = min(history['val_loss'])
            delta = final_val_loss - best_val_loss
            st.metric("Final Val Loss", f"{final_val_loss:.4f}", f"{delta:+.4f}")

        with col4:
            final_val_acc = history['val_acc'][-1]
            best_val_acc = max(history['val_acc'])
            delta = final_val_acc - best_val_acc
            st.metric("Final Val Acc", f"{final_val_acc:.4f}", f"{delta:+.4f}")

        # Training curves
        st.markdown("**Training Curves**")

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training & Validation Loss', 'Training & Validation Accuracy',
                          'Learning Rate Schedule', 'Loss Difference'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        epochs = list(range(1, len(history['train_loss']) + 1))

        # Loss plot
        fig.add_trace(
            go.Scatter(x=epochs, y=history['train_loss'], name='Train Loss',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_loss'], name='Val Loss',
                      line=dict(color='red', width=2)),
            row=1, col=1
        )

        # Accuracy plot
        fig.add_trace(
            go.Scatter(x=epochs, y=history['train_acc'], name='Train Acc',
                      line=dict(color='blue', width=2), showlegend=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_acc'], name='Val Acc',
                      line=dict(color='red', width=2), showlegend=False),
            row=1, col=2
        )

        # Learning rate plot
        fig.add_trace(
            go.Scatter(x=epochs, y=history['learning_rates'], name='Learning Rate',
                      line=dict(color='green', width=2), showlegend=False),
            row=2, col=1
        )

        # Loss difference (overfitting indicator)
        loss_diff = [t - v for t, v in zip(history['train_loss'], history['val_loss'])]
        fig.add_trace(
            go.Scatter(x=epochs, y=loss_diff, name='Loss Difference',
                      line=dict(color='purple', width=2), showlegend=False),
            row=2, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)

        # Update layout
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=2)

        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2)
        fig.update_yaxes(title_text="Learning Rate", row=2, col=1, type="log")
        fig.update_yaxes(title_text="Train - Val Loss", row=2, col=2)

        fig.update_layout(height=700, showlegend=True, hovermode='x unified')

        st.plotly_chart(fig, use_container_width=True)

        # Training summary
        with st.expander("Training Summary"):
            st.markdown(f"""
            - **Epochs Trained**: {len(history['train_loss'])}
            - **Best Val Loss**: {min(history['val_loss']):.4f} (Epoch {history['val_loss'].index(min(history['val_loss']))+1})
            - **Best Val Accuracy**: {max(history['val_acc']):.4f} (Epoch {history['val_acc'].index(max(history['val_acc']))+1})
            - **Final Learning Rate**: {history['learning_rates'][-1]:.6f}
            - **Overfitting**: {'Yes' if loss_diff[-1] > 0.1 else 'No'} (Train-Val Loss: {loss_diff[-1]:.4f})
            """)

    def _render_results(self):
        """Render training results and evaluation."""
        st.subheader("Training Results")

        if st.session_state.trained_model is None:
            st.info("No trained model available. Complete training to see results.")
            return

        # Evaluate on test set
        if st.button("Evaluate on Test Set"):
            with st.spinner("Evaluating model on test set..."):
                model = st.session_state.trained_model
                test_loader = st.session_state.test_loader
                class_names = list(st.session_state.label_to_idx.keys())

                # Evaluate
                tracker = evaluate_model(
                    model, test_loader,
                    device=st.session_state.training_config['device'],
                    class_names=class_names
                )

                st.session_state.test_metrics = tracker

        # Display results
        if 'test_metrics' in st.session_state:
            tracker = st.session_state.test_metrics

            # Overall metrics
            col1, col2, col3, col4 = st.columns(4)

            metrics = tracker.compute_all_metrics()

            with col1:
                st.metric("Test Accuracy", f"{metrics['accuracy']:.4f}")

            prf = metrics['precision_recall_f1']
            with col2:
                st.metric("Macro F1", f"{prf['macro']['f1']:.4f}")

            with col3:
                st.metric("Macro Precision", f"{prf['macro']['precision']:.4f}")

            with col4:
                st.metric("Macro Recall", f"{prf['macro']['recall']:.4f}")

            # Per-class metrics
            st.markdown("**Per-Class Metrics**")

            per_class_data = []
            for class_name, class_metrics in prf['per_class'].items():
                per_class_data.append({
                    'Class': class_name,
                    'Accuracy': f"{metrics['per_class_accuracy'][class_name]:.4f}",
                    'Precision': f"{class_metrics['precision']:.4f}",
                    'Recall': f"{class_metrics['recall']:.4f}",
                    'F1-Score': f"{class_metrics['f1']:.4f}",
                    'Support': class_metrics['support']
                })

            df_metrics = pd.DataFrame(per_class_data)
            st.dataframe(df_metrics, use_container_width=True)

            # Confusion matrix
            st.markdown("**Confusion Matrix**")

            cm = metrics['confusion_matrix']
            class_names = list(st.session_state.label_to_idx.keys())

            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=class_names,
                y=class_names,
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16}
            ))

            fig.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted',
                yaxis_title='Actual',
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Save model
            st.markdown("**Save Model**")

            model_name = st.text_input(
                "Model Name",
                value=f"{st.session_state.training_config['model_name']}_best",
                help="Name for saved model"
            )

            if st.button("Save Model"):
                save_dir = Path("outputs/models")
                save_dir.mkdir(parents=True, exist_ok=True)

                save_path = save_dir / f"{model_name}.pth"

                torch.save({
                    'model_state_dict': st.session_state.trained_model.state_dict(),
                    'model_class': st.session_state.trained_model.__class__.__name__,
                    'config': st.session_state.training_config,
                    'label_mapping': st.session_state.label_to_idx,
                    'metrics': metrics
                }, save_path)

                st.success(f"Model saved to: {save_path}")
