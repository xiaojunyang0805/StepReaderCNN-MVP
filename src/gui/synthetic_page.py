"""
Synthetic Data Generation GUI Page
Interactive interface for generating synthetic sensor signals.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from data.synthetic_generator import SyntheticSignalGenerator, create_balanced_synthetic_dataset
from data.data_loader import SensorDataLoader


class SyntheticDataPage:
    """Synthetic data generation interface."""

    def __init__(self):
        """Initialize synthetic data page."""
        # Initialize session state
        if 'generator' not in st.session_state:
            st.session_state.generator = SyntheticSignalGenerator(seed=42)

        if 'generated_samples' not in st.session_state:
            st.session_state.generated_samples = {}

        if 'preview_signal' not in st.session_state:
            st.session_state.preview_signal = None

    def render(self):
        """Render the synthetic data generation page."""
        st.header("Synthetic Data Generation")

        st.markdown("""
        Generate synthetic electrochemical sensor signals to augment your training dataset.
        Synthetic data can help:
        - Balance class distribution
        - Increase training set size
        - Improve model generalization
        """)

        # Tabs
        tab1, tab2, tab3 = st.tabs([
            "Generate Signals",
            "Preview & Analysis",
            "Batch Generation"
        ])

        with tab1:
            self._render_single_generation()

        with tab2:
            self._render_preview()

        with tab3:
            self._render_batch_generation()

    def _render_single_generation(self):
        """Render single signal generation interface."""
        st.subheader("Generate Single Signal")

        col1, col2 = st.columns(2)

        with col1:
            # Class selection
            class_name = st.selectbox(
                "Select Class",
                ['1um', '2um', '3um'],
                help="Choose the particle size class"
            )

            # Signal parameters
            st.markdown("**Signal Parameters**")

            use_default_length = st.checkbox("Use Typical Length", value=True)

            if not use_default_length:
                signal_length = st.number_input(
                    "Signal Length (points)",
                    min_value=10000,
                    max_value=500000,
                    value=100000,
                    step=10000
                )
            else:
                signal_length = None

            noise_level = st.slider(
                "Noise Level",
                min_value=0.0,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="1.0 = typical noise from real data"
            )

        with col2:
            # Features
            st.markdown("**Signal Features**")

            # Optional: Manual control of number of steps
            manual_steps = st.checkbox("Manual Step Count", value=False,
                                      help="Enable to manually set number of collision steps")

            if manual_steps:
                num_steps = st.slider(
                    "Number of Collision Steps",
                    min_value=1,
                    max_value=15,
                    value=5,
                    help="Number of collision events"
                )
            else:
                num_steps = None  # Random 3-6 steps
                st.info("Auto: 3-6 random collision steps per signal")

            add_drift = st.checkbox("Add Baseline Drift", value=False)
            add_spikes = st.checkbox("Add Random Spikes", value=False)

            st.markdown("---")

            # Display class statistics
            stats = st.session_state.generator.get_class_statistics(class_name)
            st.markdown("**Class Statistics (from real data)**")
            st.markdown(f"- Mean Current: {stats['mean_current']:.3f}")
            st.markdown(f"- Std Current: {stats['std_current']:.3f}")
            st.markdown(f"- Typical Length: {stats['typical_length']:,} points")

        # Generate button
        if st.button("Generate Signal", type="primary", use_container_width=True):
            with st.spinner("Generating signal..."):
                time_data, current_data = st.session_state.generator.generate_signal(
                    class_name,
                    length=signal_length,
                    noise_level=noise_level,
                    num_steps=num_steps,
                    add_drift=add_drift,
                    add_spikes=add_spikes
                )

                # Store in session state
                st.session_state.preview_signal = {
                    'class': class_name,
                    'time': time_data,
                    'current': current_data,
                    'params': {
                        'noise_level': noise_level,
                        'num_steps': num_steps,
                        'add_drift': add_drift,
                        'add_spikes': add_spikes
                    }
                }

            st.success(f"Generated {class_name} signal with {len(current_data):,} points!")
            st.rerun()

        # Save signal
        if st.session_state.preview_signal is not None:
            st.markdown("---")

            col1, col2 = st.columns([2, 1])

            with col1:
                filename = st.text_input(
                    "Filename",
                    value=f"synthetic_{st.session_state.preview_signal['class']}_{np.random.randint(1000):03d}.csv"
                )

            with col2:
                if st.button("Save Signal", use_container_width=True):
                    signal_data = st.session_state.preview_signal
                    save_path = st.session_state.generator.save_signal(
                        signal_data['time'],
                        signal_data['current'],
                        filename,
                        'data/synthetic'
                    )
                    st.success(f"Saved to {save_path}")

    def _render_preview(self):
        """Render signal preview and analysis."""
        st.subheader("Signal Preview & Analysis")

        if st.session_state.preview_signal is None:
            st.info("Generate a signal in the 'Generate Signals' tab to preview it here.")
            return

        signal_data = st.session_state.preview_signal
        time_data = signal_data['time']
        current_data = signal_data['current']
        class_name = signal_data['class']

        # Signal statistics
        st.markdown(f"**Generated Signal: {class_name}**")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Length", f"{len(current_data):,} pts")

        with col2:
            st.metric("Duration", f"{time_data[-1]/1000:.1f} s")

        with col3:
            st.metric("Mean Current", f"{current_data.mean():.4f}")

        with col4:
            st.metric("Std Current", f"{current_data.std():.4f}")

        # Plot signal
        st.markdown("**Time Domain**")

        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=time_data / 1000,
            y=current_data,
            mode='lines',
            name='Current',
            line=dict(color='blue', width=0.5)
        ))

        fig_time.update_layout(
            xaxis_title='Time (s)',
            yaxis_title='Current',
            height=400,
            hovermode='x unified'
        )

        st.plotly_chart(fig_time, use_container_width=True)

        # Frequency analysis
        st.markdown("**Frequency Domain**")

        fft = np.fft.rfft(current_data)
        freqs = np.fft.rfftfreq(len(current_data), d=(time_data[1] - time_data[0])/1000)

        fig_freq = go.Figure()
        fig_freq.add_trace(go.Scatter(
            x=freqs[1:],
            y=np.abs(fft[1:]),
            mode='lines',
            name='Magnitude',
            line=dict(color='green', width=1)
        ))

        fig_freq.update_layout(
            xaxis_title='Frequency (Hz)',
            yaxis_title='Magnitude',
            xaxis_type='log',
            yaxis_type='log',
            height=400
        )

        st.plotly_chart(fig_freq, use_container_width=True)

        # Statistics comparison
        st.markdown("**Comparison with Real Data**")

        real_stats = st.session_state.generator.get_class_statistics(class_name)

        comparison_df = pd.DataFrame({
            'Metric': ['Mean Current', 'Std Current'],
            'Real Data': [real_stats['mean_current'], real_stats['std_current']],
            'Generated': [current_data.mean(), current_data.std()]
        })

        st.dataframe(comparison_df, use_container_width=True)

    def _render_batch_generation(self):
        """Render batch generation interface."""
        st.subheader("Batch Generation")

        # Two modes
        mode = st.radio(
            "Generation Mode",
            ["Balance Dataset", "Custom Batch"],
            horizontal=True
        )

        if mode == "Balance Dataset":
            self._render_balance_mode()
        else:
            self._render_custom_batch_mode()

    def _render_balance_mode(self):
        """Render balanced dataset generation."""
        st.markdown("**Balance Dataset Mode**")
        st.markdown("Generate synthetic samples to balance class distribution with real data.")

        # Check if real data is loaded
        if 'dataset' not in st.session_state or st.session_state.dataset is None:
            if st.button("Load Real Data (TestData)"):
                with st.spinner("Loading TestData..."):
                    data_dir = Path("TestData")
                    loader = SensorDataLoader(str(data_dir))
                    st.session_state.dataset = loader.load_dataset("*.csv")
                st.success("TestData loaded!")
                st.rerun()
            return

        dataset = st.session_state.dataset

        # Show current distribution
        st.markdown("**Current Distribution**")

        dist_df = pd.DataFrame({
            'Class': list(dataset.keys()),
            'Real Samples': [len(samples) for samples in dataset.values()]
        })

        st.dataframe(dist_df, use_container_width=True)

        # Target samples
        target_samples = st.number_input(
            "Target Samples per Class",
            min_value=1,
            max_value=500,
            value=50,
            step=10,
            help="Total samples per class (real + synthetic)"
        )

        # Calculate synthetic needed
        synthetic_needed = {}
        for class_name, samples in dataset.items():
            needed = max(0, target_samples - len(samples))
            synthetic_needed[class_name] = needed

        st.markdown("**Synthetic Samples to Generate**")

        gen_df = pd.DataFrame({
            'Class': list(synthetic_needed.keys()),
            'Synthetic Needed': list(synthetic_needed.values()),
            'Total After': [target_samples] * len(synthetic_needed)
        })

        st.dataframe(gen_df, use_container_width=True)

        # Generation parameters
        with st.expander("Advanced Parameters"):
            noise_min = st.slider("Min Noise Level", 0.0, 2.0, 0.8, 0.1, key="balance_noise_min")
            noise_max = st.slider("Max Noise Level", 0.0, 2.0, 1.2, 0.1, key="balance_noise_max")

            add_drift = st.checkbox("Add Baseline Drift", value=False, key="balance_add_drift")
            add_spikes = st.checkbox("Add Random Spikes", value=False, key="balance_add_spikes")

        # Generate button
        if st.button("Generate Balanced Dataset", type="primary", use_container_width=True):
            total_to_generate = sum(synthetic_needed.values())

            if total_to_generate == 0:
                st.warning("No synthetic samples needed. Dataset is already balanced!")
                return

            with st.spinner(f"Generating {total_to_generate} synthetic samples..."):
                synthetic_data = st.session_state.generator.generate_batch(
                    synthetic_needed,
                    noise_level_range=(noise_min, noise_max),
                    add_drift=add_drift,
                    add_spikes=add_spikes
                )

                # Save to disk
                count = st.session_state.generator.save_batch(synthetic_data, 'data/synthetic')

                st.session_state.generated_samples = synthetic_data

            st.success(f"Generated and saved {count} synthetic samples!")

            # Show results
            for class_name, samples in synthetic_data.items():
                st.markdown(f"- **{class_name}**: {len(samples)} samples generated")

    def _render_custom_batch_mode(self):
        """Render custom batch generation."""
        st.markdown("**Custom Batch Mode**")
        st.markdown("Generate a custom number of samples for each class.")

        col1, col2, col3 = st.columns(3)

        with col1:
            count_1um = st.number_input("1um Count", min_value=0, max_value=500, value=10, step=5)

        with col2:
            count_2um = st.number_input("2um Count", min_value=0, max_value=500, value=10, step=5)

        with col3:
            count_3um = st.number_input("3um Count", min_value=0, max_value=500, value=10, step=5)

        total_count = count_1um + count_2um + count_3um

        st.markdown(f"**Total Samples**: {total_count}")

        # Generation parameters
        with st.expander("Advanced Parameters"):
            noise_min = st.slider("Min Noise Level", 0.0, 2.0, 0.8, 0.1, key="custom_noise_min")
            noise_max = st.slider("Max Noise Level", 0.0, 2.0, 1.2, 0.1, key="custom_noise_max")

            add_drift = st.checkbox("Add Baseline Drift", value=False, key="custom_add_drift")
            add_spikes = st.checkbox("Add Random Spikes", value=False, key="custom_add_spikes")

        # Generate button
        if st.button("Generate Batch", type="primary", use_container_width=True):
            if total_count == 0:
                st.warning("Please specify at least one sample to generate.")
                return

            class_counts = {
                '1um': count_1um,
                '2um': count_2um,
                '3um': count_3um
            }

            # Remove classes with 0 count
            class_counts = {k: v for k, v in class_counts.items() if v > 0}

            with st.spinner(f"Generating {total_count} synthetic samples..."):
                synthetic_data = st.session_state.generator.generate_batch(
                    class_counts,
                    noise_level_range=(noise_min, noise_max),
                    add_drift=add_drift,
                    add_spikes=add_spikes
                )

                # Save to disk
                count = st.session_state.generator.save_batch(synthetic_data, 'data/synthetic')

                st.session_state.generated_samples = synthetic_data

            st.success(f"Generated and saved {count} synthetic samples!")

            # Show results
            for class_name, samples in synthetic_data.items():
                st.markdown(f"- **{class_name}**: {len(samples)} samples generated")


if __name__ == "__main__":
    # For testing
    st.set_page_config(page_title="Synthetic Data Generation", layout="wide")
    page = SyntheticDataPage()
    page.render()
