"""
Upload Handler Component
Handles file upload, data loading, and dataset management.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple

# Import data loader
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from data.data_loader import SensorDataLoader


class UploadHandler:
    """Handles data upload and loading operations."""

    def __init__(self):
        """Initialize the UploadHandler."""
        self.data_loader = None

    def render(self):
        """Render the upload interface."""
        st.markdown("### Upload Options")

        # Create tabs for different upload methods
        tab1, tab2 = st.tabs(["📁 Load from TestData", "⬆️ Upload Files"])

        with tab1:
            self._render_testdata_loader()

        with tab2:
            self._render_file_uploader()

    def _render_testdata_loader(self):
        """Render the TestData folder loader."""
        st.markdown("#### Load Data from TestData Folder")

        # Get TestData path
        project_root = Path(__file__).parent.parent.parent
        testdata_path = project_root / "TestData"

        if not testdata_path.exists():
            st.error(f"TestData folder not found at: {testdata_path}")
            return

        # Count files
        csv_files = list(testdata_path.glob("*.csv"))
        num_files = len(csv_files)

        st.info(f"Found {num_files} CSV files in TestData folder")

        # Display some file examples
        if num_files > 0:
            with st.expander("Preview Files"):
                sample_files = csv_files[:10]
                for f in sample_files:
                    st.text(f"• {f.name}")
                if num_files > 10:
                    st.text(f"... and {num_files - 10} more files")

        # Load button
        col1, col2 = st.columns([1, 3])

        with col1:
            if st.button("📂 Load TestData", type="primary", use_container_width=True):
                self._load_testdata(str(testdata_path))

        with col2:
            if st.session_state.get('data_loaded', False):
                st.success(f"✅ Data loaded successfully!")

    def _render_file_uploader(self):
        """Render the file upload interface."""
        st.markdown("#### Upload Your Own Data")

        st.info("""
        **Supported formats:**
        - CSV files (time, current)
        - NPY files (NumPy arrays)
        - HDF5 files

        **File naming convention:**
        - Include class label in filename (e.g., "PS 1um 01.csv", "PS 2um sample1.csv")
        - Pattern: `PS <size>um <id>.csv`
        """)

        uploaded_files = st.file_uploader(
            "Choose files",
            type=['csv', 'npy', 'h5', 'hdf5'],
            accept_multiple_files=True,
            help="Upload multiple sensor data files"
        )

        if uploaded_files:
            st.write(f"**{len(uploaded_files)} files selected:**")

            # Preview uploaded files
            with st.expander("View uploaded files"):
                for file in uploaded_files[:20]:
                    st.text(f"• {file.name}")
                if len(uploaded_files) > 20:
                    st.text(f"... and {len(uploaded_files) - 20} more files")

            if st.button("📥 Process Uploaded Files", type="primary"):
                self._process_uploaded_files(uploaded_files)

    def _load_testdata(self, data_dir: str):
        """Load data from TestData directory."""
        try:
            with st.spinner("Loading data from TestData folder..."):
                # Initialize data loader
                loader = SensorDataLoader(data_dir)

                # Load dataset
                dataset = loader.load_dataset(pattern="*.csv")

                if not dataset:
                    st.error("No data files found!")
                    return

                # Get summary statistics
                summary_df = loader.get_dataset_summary(dataset)

                # Store in session state
                st.session_state.dataset = dataset
                st.session_state.data_summary = summary_df
                st.session_state.data_loaded = True
                st.session_state.data_source = "TestData"

                # Show success message with details
                total_files = sum(len(files) for files in dataset.values())
                num_classes = len(dataset.keys())

                st.success(f"""
                ✅ **Data loaded successfully!**

                - **Total samples:** {total_files}
                - **Classes:** {num_classes} ({', '.join(sorted(dataset.keys()))})
                - **Source:** TestData folder
                """)

                # Display summary
                st.markdown("#### Quick Summary")
                st.dataframe(summary_df, use_container_width=True)

                st.info("👉 Switch to the **Dataset Overview** or **Signal Viewer** tabs to explore the data.")

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.exception(e)

    def _process_uploaded_files(self, uploaded_files):
        """Process uploaded files."""
        try:
            with st.spinner("Processing uploaded files..."):
                dataset = {}
                errors = []

                for file in uploaded_files:
                    try:
                        # Determine file type
                        file_ext = Path(file.name).suffix.lower()

                        if file_ext == '.csv':
                            # Read CSV
                            df = pd.read_csv(file)
                            time_data = df.iloc[:, 0].values
                            current_data = df.iloc[:, 1].values
                        elif file_ext == '.npy':
                            # Read NPY
                            data = np.load(file)
                            if data.shape[1] == 2:
                                time_data = data[:, 0]
                                current_data = data[:, 1]
                            else:
                                raise ValueError("NPY file must have 2 columns (time, current)")
                        else:
                            errors.append(f"{file.name}: Unsupported format")
                            continue

                        # Extract label from filename
                        label = self._extract_label(file.name)

                        if label is None:
                            label = "unknown"

                        # Add to dataset
                        if label not in dataset:
                            dataset[label] = []

                        dataset[label].append((time_data, current_data))

                    except Exception as e:
                        errors.append(f"{file.name}: {str(e)}")

                # Check if any files were successfully loaded
                if not dataset:
                    st.error("No files could be loaded successfully!")
                    if errors:
                        with st.expander("View Errors"):
                            for error in errors:
                                st.error(error)
                    return

                # Calculate summary statistics (matching data_loader format)
                summary_data = []
                for label, files in dataset.items():
                    for time_data, current_data in files:
                        summary_data.append({
                            'Filename': 'uploaded',
                            'Label': label,
                            'Num_Points': len(current_data),
                            'Duration_ms': time_data[-1] - time_data[0] if len(time_data) > 1 else 0,
                            'Sampling_Rate_Hz': 1000.0 / (time_data[1] - time_data[0]) if len(time_data) > 1 else 0,
                            'Current_Mean': np.mean(current_data),
                            'Current_Std': np.std(current_data),
                            'Current_Min': np.min(current_data),
                            'Current_Max': np.max(current_data),
                            'Has_NaN': np.any(np.isnan(current_data)),
                        })

                summary_df = pd.DataFrame(summary_data)

                # Store in session state
                st.session_state.dataset = dataset
                st.session_state.data_summary = summary_df
                st.session_state.data_loaded = True
                st.session_state.data_source = "Uploaded Files"

                # Show success
                total_files = sum(len(files) for files in dataset.values())
                num_classes = len(dataset.keys())

                st.success(f"""
                ✅ **Files processed successfully!**

                - **Total samples:** {total_files}
                - **Classes:** {num_classes} ({', '.join(sorted(dataset.keys()))})
                - **Source:** Uploaded files
                """)

                if errors:
                    st.warning(f"{len(errors)} file(s) had errors:")
                    with st.expander("View Errors"):
                        for error in errors:
                            st.error(error)

                # Display summary
                st.markdown("#### Quick Summary")
                st.dataframe(summary_df, use_container_width=True)

                st.info("👉 Switch to the **Dataset Overview** or **Signal Viewer** tabs to explore the data.")

        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
            st.exception(e)

    def _extract_label(self, filename: str) -> str:
        """Extract label from filename."""
        import re

        # Pattern: PS <size>um
        pattern = r'PS\s*(\d+(?:\.\d+)?)\s*um'
        match = re.search(pattern, filename, re.IGNORECASE)

        if match:
            size = match.group(1)
            # Remove decimal point if it's .0
            if '.' in size and float(size) == int(float(size)):
                size = str(int(float(size)))
            return f"{size}um"

        return None
