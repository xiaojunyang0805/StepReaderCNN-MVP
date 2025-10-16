"""
Data Viewer Component
Provides interactive visualization and analysis tools for sensor data.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


class DataViewer:
    """Interactive data visualization component for Streamlit GUI."""

    def __init__(self):
        """Initialize the DataViewer."""
        self.dataset = st.session_state.get('dataset', None)
        self.data_summary = st.session_state.get('data_summary', None)

    def render_dataset_overview(self):
        """Render the dataset overview dashboard."""
        if self.dataset is None or self.data_summary is None:
            st.warning("No data loaded. Please load data first.")
            return

        # Display summary statistics
        st.markdown("### 📊 Dataset Statistics")

        # Create metric cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_files = sum(len(files) for files in self.dataset.values())
            st.metric("Total Samples", total_files)

        with col2:
            num_classes = len(self.dataset.keys())
            st.metric("Number of Classes", num_classes)

        with col3:
            if not self.data_summary.empty:
                avg_length = int(self.data_summary['Num_Points'].mean())
                st.metric("Avg. Signal Length", f"{avg_length:,}")

        with col4:
            if not self.data_summary.empty:
                total_points = int(self.data_summary['Num_Points'].sum())
                st.metric("Total Data Points", f"{total_points:,}")

        st.markdown("---")

        # Class distribution
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Class Distribution")
            class_counts = {label: len(files) for label, files in self.dataset.items()}

            fig = go.Figure(data=[
                go.Bar(
                    x=list(class_counts.keys()),
                    y=list(class_counts.values()),
                    text=list(class_counts.values()),
                    textposition='auto',
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
                )
            ])

            fig.update_layout(
                title="Samples per Class",
                xaxis_title="Particle Size",
                yaxis_title="Number of Samples",
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Class Distribution (Pie Chart)")

            fig = go.Figure(data=[
                go.Pie(
                    labels=list(class_counts.keys()),
                    values=list(class_counts.values()),
                    hole=0.3,
                    marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c']
                )
            ])

            fig.update_layout(
                title="Class Proportions",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        # Detailed statistics table
        st.markdown("---")
        st.markdown("#### 📋 Detailed Statistics by Class")

        if not self.data_summary.empty:
            # Format the dataframe for better display
            display_df = self.data_summary.copy()

            # Round numeric columns
            numeric_cols = display_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if 'Length' in col or 'Points' in col:
                    display_df[col] = display_df[col].apply(lambda x: f"{int(x):,}")
                else:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

            st.dataframe(display_df, use_container_width=True)

        # Signal length distribution
        st.markdown("---")
        st.markdown("#### 📏 Signal Length Distribution")

        length_data = []
        for label, files in self.dataset.items():
            for item in files:
                # Handle both 2-tuple (time, current) and 3-tuple (time, current, filename)
                time_data = item[0]
                length_data.append({
                    'Class': label,
                    'Length': len(time_data)
                })

        length_df = pd.DataFrame(length_data)

        fig = go.Figure()

        for label in sorted(length_df['Class'].unique()):
            data = length_df[length_df['Class'] == label]['Length']
            fig.add_trace(go.Box(
                y=data,
                name=label,
                boxmean='sd'
            ))

        fig.update_layout(
            title="Signal Length Distribution by Class",
            yaxis_title="Signal Length (samples)",
            height=400,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_signal_viewer(self):
        """Render the interactive signal viewer."""
        if self.dataset is None:
            st.warning("No data loaded. Please load data first.")
            return

        st.markdown("### 🔍 Interactive Signal Viewer")

        # Create two columns for controls and visualization
        control_col, viz_col = st.columns([1, 3])

        with control_col:
            st.markdown("#### Controls")

            # Class selection
            available_classes = sorted(self.dataset.keys())
            selected_class = st.selectbox(
                "Select Class",
                available_classes,
                help="Choose particle size class"
            )

            # Sample selection with navigator buttons
            num_samples = len(self.dataset[selected_class])

            # Initialize session state for sample index if not exists
            if 'signal_viewer_sample_idx' not in st.session_state:
                st.session_state.signal_viewer_sample_idx = 0

            # Reset index if class changed
            if 'signal_viewer_prev_class' not in st.session_state or st.session_state.signal_viewer_prev_class != selected_class:
                st.session_state.signal_viewer_sample_idx = 0
                st.session_state.signal_viewer_prev_class = selected_class

            # Ensure index is within bounds
            if st.session_state.signal_viewer_sample_idx >= num_samples:
                st.session_state.signal_viewer_sample_idx = num_samples - 1
            if st.session_state.signal_viewer_sample_idx < 0:
                st.session_state.signal_viewer_sample_idx = 0

            st.markdown("**Sample Navigation**")

            # Navigator buttons
            nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])

            with nav_col1:
                if st.button("◀ Prev", use_container_width=True, disabled=(st.session_state.signal_viewer_sample_idx == 0)):
                    st.session_state.signal_viewer_sample_idx -= 1
                    st.rerun()

            with nav_col2:
                st.markdown(f"<div style='text-align: center; padding: 8px; background-color: rgba(128, 128, 128, 0.1); border-radius: 5px;'><b>Sample {st.session_state.signal_viewer_sample_idx} / {num_samples - 1}</b></div>", unsafe_allow_html=True)

            with nav_col3:
                if st.button("Next ▶", use_container_width=True, disabled=(st.session_state.signal_viewer_sample_idx == num_samples - 1)):
                    st.session_state.signal_viewer_sample_idx += 1
                    st.rerun()

            selected_sample_idx = st.session_state.signal_viewer_sample_idx

            # Visualization options
            st.markdown("---")
            st.markdown("#### Display Options")

            show_grid = st.checkbox("Show Grid", value=True)
            show_markers = st.checkbox("Show Markers", value=False)

            # Downsampling for large signals
            max_points = st.number_input(
                "Max Points to Display",
                min_value=1000,
                max_value=1000000,
                value=50000,
                step=1000,
                help="Downsample for faster rendering"
            )

            # Compare mode
            st.markdown("---")
            compare_mode = st.checkbox("Compare Multiple Samples", value=False)

            if compare_mode:
                num_compare = st.slider("Number of Samples", 2, min(5, num_samples), 2)
                compare_indices = st.multiselect(
                    "Select Samples to Compare",
                    range(num_samples),
                    default=list(range(min(num_compare, num_samples))),
                    max_selections=5
                )

        with viz_col:
            if not compare_mode:
                # Single signal view
                self._plot_single_signal(
                    selected_class,
                    selected_sample_idx,
                    show_grid,
                    show_markers,
                    max_points
                )
            else:
                # Comparison view
                if compare_indices:
                    self._plot_comparison(
                        selected_class,
                        compare_indices,
                        show_grid,
                        show_markers,
                        max_points
                    )
                else:
                    st.info("Select at least 2 samples to compare.")

        # Signal statistics
        st.markdown("---")
        st.markdown("### 📊 Signal Statistics")

        # Handle both 2-tuple and 3-tuple formats
        item = self.dataset[selected_class][selected_sample_idx]
        time_data, current_data = item[0], item[1]

        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

        with stat_col1:
            st.metric("Signal Length", f"{len(time_data):,}")
            st.metric("Duration (ms)", f"{time_data[-1] - time_data[0]:.2f}")

        with stat_col2:
            st.metric("Mean Current", f"{np.mean(current_data):.4e}")
            st.metric("Std Dev", f"{np.std(current_data):.4e}")

        with stat_col3:
            st.metric("Min Current", f"{np.min(current_data):.4e}")
            st.metric("Max Current", f"{np.max(current_data):.4e}")

        with stat_col4:
            sampling_rate = len(time_data) / (time_data[-1] - time_data[0])
            st.metric("Sampling Rate", f"{sampling_rate:.1f} Hz")
            st.metric("Peak-to-Peak", f"{np.ptp(current_data):.4e}")

    def _plot_single_signal(self, class_label: str, sample_idx: int,
                           show_grid: bool, show_markers: bool, max_points: int):
        """Plot a single signal."""
        # Handle both 2-tuple and 3-tuple formats
        item = self.dataset[class_label][sample_idx]
        time_data, current_data = item[0], item[1]

        # Downsample if needed
        if len(time_data) > max_points:
            step = len(time_data) // max_points
            time_data = time_data[::step]
            current_data = current_data[::step]
            st.info(f"Signal downsampled from {len(self.dataset[class_label][sample_idx][0]):,} to {len(time_data):,} points for visualization.")

        # Create plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=time_data,
            y=current_data,
            mode='lines+markers' if show_markers else 'lines',
            name=f"{class_label} - Sample {sample_idx}",
            line=dict(width=1.5),
            marker=dict(size=3) if show_markers else None
        ))

        fig.update_layout(
            title=f"Signal: {class_label} - Sample {sample_idx}",
            xaxis_title="Time (ms)",
            yaxis_title="Current (A)",
            height=500,
            hovermode='x unified',
            showlegend=True,
            xaxis=dict(showgrid=show_grid),
            yaxis=dict(showgrid=show_grid)
        )

        st.plotly_chart(fig, use_container_width=True)

    def _plot_comparison(self, class_label: str, sample_indices: List[int],
                        show_grid: bool, show_markers: bool, max_points: int):
        """Plot multiple signals for comparison."""
        fig = go.Figure()

        colors = px.colors.qualitative.Plotly

        for idx, sample_idx in enumerate(sample_indices):
            # Handle both 2-tuple and 3-tuple formats
            item = self.dataset[class_label][sample_idx]
            time_data, current_data = item[0], item[1]

            # Downsample if needed
            if len(time_data) > max_points:
                step = len(time_data) // max_points
                time_data = time_data[::step]
                current_data = current_data[::step]

            fig.add_trace(go.Scatter(
                x=time_data,
                y=current_data,
                mode='lines+markers' if show_markers else 'lines',
                name=f"Sample {sample_idx}",
                line=dict(width=1.5, color=colors[idx % len(colors)]),
                marker=dict(size=3) if show_markers else None
            ))

        fig.update_layout(
            title=f"Signal Comparison: {class_label}",
            xaxis_title="Time (ms)",
            yaxis_title="Current (A)",
            height=500,
            hovermode='x unified',
            showlegend=True,
            xaxis=dict(showgrid=show_grid),
            yaxis=dict(showgrid=show_grid)
        )

        st.plotly_chart(fig, use_container_width=True)
