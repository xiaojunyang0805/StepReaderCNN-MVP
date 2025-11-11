"""Data loading utilities for electrochemical sensor signals."""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

import numpy as np
import pandas as pd
import h5py


class SensorDataLoader:
    """Load and parse electrochemical sensor signal data."""
    
    def __init__(self, data_dir: str):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing sensor data files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
    
    def load_csv(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Tuple of (time_array, current_array)
        """
        df = pd.read_csv(file_path)
        
        # Handle different column name formats
        time_col = df.columns[0]  # First column is time
        current_col = df.columns[1]  # Second column is current
        
        time = df[time_col].values
        current = df[current_col].values
        
        return time, current
    
    def load_npy(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from NPY file.
        
        Args:
            file_path: Path to NPY file
            
        Returns:
            Tuple of (time_array, current_array)
        """
        data = np.load(file_path)
        if data.shape[1] != 2:
            raise ValueError(f"Expected 2 columns in NPY file, got {data.shape[1]}")
        return data[:, 0], data[:, 1]
    
    def load_hdf5(self, file_path: str, dataset_name: str = 'data') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from HDF5 file.
        
        Args:
            file_path: Path to HDF5 file
            dataset_name: Name of dataset within HDF5 file
            
        Returns:
            Tuple of (time_array, current_array)
        """
        with h5py.File(file_path, 'r') as f:
            data = f[dataset_name][:]
        return data[:, 0], data[:, 1]
    
    def load_file(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data file with automatic format detection.
        
        Args:
            file_path: Path to data file
            
        Returns:
            Tuple of (time_array, current_array)
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if suffix == '.csv':
            return self.load_csv(str(file_path))
        elif suffix == '.npy':
            return self.load_npy(str(file_path))
        elif suffix in ['.h5', '.hdf5']:
            return self.load_hdf5(str(file_path))
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def extract_label_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract label from filename (e.g., 'PS 1um 01.csv' -> '1um').
        
        Args:
            filename: Name of the file
            
        Returns:
            Label string or None if not found
        """
        # Pattern: PS <size>um
        match = re.search(r'PS\s+(\d+um)', filename, re.IGNORECASE)
        if match:
            return match.group(1)
        return None
    
    def load_dataset(self, pattern: str = "*.csv") -> Dict[str, List[Tuple[np.ndarray, np.ndarray, str]]]:
        """
        Load all data files matching pattern and group by label.
        
        Args:
            pattern: Glob pattern for file matching
            
        Returns:
            Dictionary mapping labels to list of (time, current, filename) tuples
        """
        data_by_label = {}
        
        for file_path in self.data_dir.glob(pattern):
            try:
                time, current = self.load_file(str(file_path))
                label = self.extract_label_from_filename(file_path.name)
                
                if label is None:
                    print(f"Warning: Could not extract label from {file_path.name}, skipping...")
                    continue
                
                if label not in data_by_label:
                    data_by_label[label] = []
                
                data_by_label[label].append((time, current, file_path.name))
                
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
                continue
        
        return data_by_label
    
    def get_dataset_summary(self, data_by_label: Dict) -> pd.DataFrame:
        """
        Generate summary statistics for loaded dataset.
        
        Args:
            data_by_label: Dictionary from load_dataset()
            
        Returns:
            DataFrame with summary statistics
        """
        summary_data = []
        
        for label, samples in data_by_label.items():
            for time, current, filename in samples:
                summary_data.append({
                    'Filename': filename,
                    'Label': label,
                    'Num_Points': len(current),
                    'Duration_ms': time[-1] - time[0] if len(time) > 1 else 0,
                    'Sampling_Rate_Hz': 1000.0 / (time[1] - time[0]) if len(time) > 1 else 0,
                    'Current_Mean': np.mean(current),
                    'Current_Std': np.std(current),
                    'Current_Min': np.min(current),
                    'Current_Max': np.max(current),
                    'Has_NaN': np.any(np.isnan(current)),
                })
        
        return pd.DataFrame(summary_data)


def create_label_mapping(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create bidirectional mapping between labels and integer indices.
    
    Args:
        labels: List of unique label strings
        
    Returns:
        Tuple of (label_to_idx, idx_to_label) dictionaries
    """
    sorted_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(sorted_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label


if __name__ == "__main__":
    # Example usage
    loader = SensorDataLoader("TestData")
    data = loader.load_dataset("*.csv")
    
    print(f"Loaded {sum(len(v) for v in data.values())} files")
    print(f"Labels found: {list(data.keys())}")
    
    for label, samples in data.items():
        print(f"\n{label}: {len(samples)} samples")
    
    # Generate summary
    summary = loader.get_dataset_summary(data)
    print("\nDataset Summary:")
    print(summary.groupby('Label').agg({
        'Num_Points': ['mean', 'std', 'min', 'max'],
        'Current_Mean': ['mean', 'std'],
        'Current_Std': 'mean'
    }))
