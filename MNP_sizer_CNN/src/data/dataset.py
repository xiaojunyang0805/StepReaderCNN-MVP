"""
PyTorch Dataset Classes for Sensor Signal Data
Custom Dataset and DataLoader for electrochemical sensor signals.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path

from .preprocessing import SignalPreprocessor, create_preprocessing_pipeline
from .augmentation import TimeSeriesAugmenter


class SensorSignalDataset(Dataset):
    """PyTorch Dataset for electrochemical sensor signals."""

    def __init__(self,
                 data_dict: Dict[str, List[Tuple]],
                 label_to_idx: Dict[str, int],
                 target_length: Optional[int] = None,
                 normalize_method: Optional[str] = 'zscore',
                 augment: bool = False,
                 augmenter: Optional[TimeSeriesAugmenter] = None,
                 transform: Optional[Callable] = None):
        """
        Initialize dataset.

        Args:
            data_dict: Dictionary mapping labels to list of (time, current, filename) tuples
            label_to_idx: Dictionary mapping label strings to integer indices
            target_length: Target sequence length (None = use original length)
            normalize_method: Normalization method ('zscore', 'minmax', 'robust', or None)
            augment: Whether to apply data augmentation
            augmenter: TimeSeriesAugmenter instance (created if None and augment=True)
            transform: Optional transform function to apply
        """
        self.label_to_idx = label_to_idx
        self.idx_to_label = {v: k for k, v in label_to_idx.items()}

        self.target_length = target_length
        self.normalize_method = normalize_method
        self.augment = augment
        self.transform = transform

        # Create augmenter if needed
        if self.augment and augmenter is None:
            self.augmenter = TimeSeriesAugmenter()
        else:
            self.augmenter = augmenter

        # Create preprocessor
        self.preprocessor = SignalPreprocessor()

        # Flatten dataset
        self.samples = []
        self.labels = []

        for label_str, sample_list in data_dict.items():
            label_idx = label_to_idx[label_str]

            for item in sample_list:
                # Handle both 2-tuple and 3-tuple formats
                time_data = item[0]
                current_data = item[1]
                filename = item[2] if len(item) > 2 else 'unknown'

                self.samples.append((time_data, current_data, filename))
                self.labels.append(label_idx)

        self.labels = np.array(self.labels)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (signal_tensor, label)
        """
        time_data, current_data, filename = self.samples[idx]
        label = self.labels[idx]

        # Convert to numpy if needed
        time_data = np.array(time_data, dtype=np.float32)
        current_data = np.array(current_data, dtype=np.float32)

        # Apply augmentation (only during training)
        if self.augment:
            time_data, current_data = self.augmenter.random_augment(
                time_data, current_data, prob=0.5
            )

        # Length normalization
        if self.target_length is not None:
            time_data = self.preprocessor.normalize_length(
                time_data, self.target_length
            )
            current_data = self.preprocessor.normalize_length(
                current_data, self.target_length
            )

        # Value normalization
        if self.normalize_method is not None:
            if self.normalize_method == 'zscore':
                current_data, _ = self.preprocessor.normalize_zscore(current_data)
            elif self.normalize_method == 'minmax':
                current_data, _ = self.preprocessor.normalize_minmax(current_data)
            elif self.normalize_method == 'robust':
                current_data, _ = self.preprocessor.normalize_robust(current_data)

        # Apply custom transform if provided
        if self.transform is not None:
            time_data, current_data = self.transform(time_data, current_data)

        # Convert to PyTorch tensor
        # Shape: (1, sequence_length) for single-channel signal
        signal_tensor = torch.FloatTensor(current_data).unsqueeze(0)

        return signal_tensor, label

    def get_label_counts(self) -> Dict[int, int]:
        """Get count of samples per label."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))

    def get_class_weights(self, method: str = 'balanced') -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced datasets.

        Args:
            method: Weight calculation method ('balanced' or 'sqrt')

        Returns:
            Tensor of class weights
        """
        label_counts = self.get_label_counts()
        n_classes = len(label_counts)
        n_total = len(self)

        weights = []

        if method == 'balanced':
            for class_idx in range(n_classes):
                count = label_counts.get(class_idx, 1)
                weight = n_total / (n_classes * count)
                weights.append(weight)

        elif method == 'sqrt':
            max_count = max(label_counts.values())
            for class_idx in range(n_classes):
                count = label_counts.get(class_idx, 1)
                weight = np.sqrt(max_count / count)
                weights.append(weight)

        # Normalize
        weights = np.array(weights)
        weights = weights / np.sum(weights) * n_classes

        return torch.FloatTensor(weights)


def collate_fn_variable_length(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for variable-length sequences.

    Args:
        batch: List of (signal_tensor, label) tuples

    Returns:
        Tuple of (padded_signals, labels, lengths)
    """
    signals = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Get lengths
    lengths = torch.LongTensor([s.size(-1) for s in signals])

    # Pad sequences
    padded_signals = torch.nn.utils.rnn.pad_sequence(
        [s.squeeze(0).T for s in signals],  # Transpose to (seq_len, channels)
        batch_first=True,
        padding_value=0.0
    ).transpose(1, 2)  # Back to (batch, channels, seq_len)

    labels = torch.LongTensor(labels)

    return padded_signals, labels, lengths


def collate_fn_fixed_length(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for fixed-length sequences.

    Args:
        batch: List of (signal_tensor, label) tuples

    Returns:
        Tuple of (signals, labels)
    """
    signals = torch.stack([item[0] for item in batch])
    labels = torch.LongTensor([item[1] for item in batch])

    return signals, labels


def create_dataloaders(train_data: Dict[str, List],
                      val_data: Dict[str, List],
                      test_data: Optional[Dict[str, List]],
                      label_to_idx: Dict[str, int],
                      batch_size: int = 32,
                      target_length: Optional[int] = None,
                      normalize_method: str = 'zscore',
                      augment_train: bool = True,
                      num_workers: int = 0,
                      pin_memory: bool = False) -> Tuple:
    """
    Create train, validation, and test DataLoaders.

    Args:
        train_data: Training data dictionary
        val_data: Validation data dictionary
        test_data: Test data dictionary (optional)
        label_to_idx: Label to index mapping
        batch_size: Batch size
        target_length: Target sequence length
        normalize_method: Normalization method
        augment_train: Whether to augment training data
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        test_loader is None if test_data is None
    """
    # Create datasets
    train_dataset = SensorSignalDataset(
        train_data,
        label_to_idx,
        target_length=target_length,
        normalize_method=normalize_method,
        augment=augment_train
    )

    val_dataset = SensorSignalDataset(
        val_data,
        label_to_idx,
        target_length=target_length,
        normalize_method=normalize_method,
        augment=False
    )

    if test_data is not None:
        test_dataset = SensorSignalDataset(
            test_data,
            label_to_idx,
            target_length=target_length,
            normalize_method=normalize_method,
            augment=False
        )
    else:
        test_dataset = None

    # Choose collate function
    if target_length is None:
        collate_func = collate_fn_variable_length
    else:
        collate_func = collate_fn_fixed_length

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_func
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_func
    )

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_func
        )
    else:
        test_loader = None

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    print("Testing SensorSignalDataset...")

    # Create dummy data
    dummy_data = {
        '1um': [(np.random.randn(100), np.random.randn(100), f'1um_{i}') for i in range(7)],
        '2um': [(np.random.randn(150), np.random.randn(150), f'2um_{i}') for i in range(9)],
        '3um': [(np.random.randn(120), np.random.randn(120), f'3um_{i}') for i in range(26)]
    }

    # Create label mapping
    label_to_idx = {'1um': 0, '2um': 1, '3um': 2}

    # Create dataset
    dataset = SensorSignalDataset(
        dummy_data,
        label_to_idx,
        target_length=200,
        normalize_method='zscore',
        augment=True
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Label counts: {dataset.get_label_counts()}")

    # Test getitem
    signal, label = dataset[0]
    print(f"\nSample shape: {signal.shape}")
    print(f"Sample label: {label} ({dataset.idx_to_label[label]})")

    # Create dataloader
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn_fixed_length)

    # Test batch
    batch_signals, batch_labels = next(iter(loader))
    print(f"\nBatch signals shape: {batch_signals.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")

    print("\nDataset testing complete!")
