"""
Test script for preprocessing pipeline with real TestData.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from data.data_loader import SensorDataLoader
from data.data_split import stratified_split, get_split_statistics, print_split_statistics, create_class_weights
from data.preprocessing import SignalPreprocessor, create_preprocessing_pipeline
from data.augmentation import TimeSeriesAugmenter
from data.dataset import SensorSignalDataset, create_dataloaders

import numpy as np
import torch


def main():
    print("=" * 80)
    print("PREPROCESSING PIPELINE TEST")
    print("=" * 80)

    # 1. Load data
    print("\n1. Loading TestData...")
    data_dir = Path(__file__).parent / "TestData"
    loader = SensorDataLoader(str(data_dir))
    dataset = loader.load_dataset("*.csv")

    print(f"   Loaded {sum(len(v) for v in dataset.values())} files")
    print(f"   Classes: {', '.join(sorted(dataset.keys()))}")

    # 2. Split data
    print("\n2. Splitting data (70/15/15)...")
    train_data, val_data, test_data = stratified_split(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )

    stats = get_split_statistics(train_data, val_data, test_data)
    print_split_statistics(stats)

    # 3. Test preprocessing
    print("\n3. Testing preprocessing functions...")
    preprocessor = SignalPreprocessor()

    # Get a sample signal
    sample_time, sample_current, _ = train_data['1um'][0]
    print(f"   Original signal length: {len(sample_current)}")
    print(f"   Original signal mean: {np.mean(sample_current):.6e}")
    print(f"   Original signal std: {np.std(sample_current):.6e}")

    # Test normalization
    normalized, params = preprocessor.normalize_zscore(sample_current)
    print(f"\n   Z-score normalized mean: {np.mean(normalized):.6f}")
    print(f"   Z-score normalized std: {np.std(normalized):.6f}")

    # Test length normalization
    target_length = 10000
    normalized_length = preprocessor.normalize_length(sample_current, target_length)
    print(f"\n   Length normalized to: {len(normalized_length)}")

    # 4. Test augmentation
    print("\n4. Testing augmentation...")
    augmenter = TimeSeriesAugmenter(seed=42)

    # Test different augmentations
    _, aug_warp = augmenter.time_warp(sample_time, sample_current)
    print(f"   Time warp: shape {aug_warp.shape}")

    _, aug_noise = augmenter.add_gaussian_noise(sample_time, sample_current, snr_db=20)
    print(f"   Gaussian noise (SNR=20dB): std increased by {np.std(aug_noise)/np.std(sample_current):.2f}x")

    _, aug_scale = augmenter.magnitude_scale(sample_time, sample_current, sigma=0.2)
    print(f"   Magnitude scale: mean changed from {np.mean(sample_current):.6e} to {np.mean(aug_scale):.6e}")

    # 5. Test PyTorch Dataset
    print("\n5. Testing PyTorch Dataset...")

    # Create label mapping
    unique_labels = sorted(dataset.keys())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    print(f"   Label mapping: {label_to_idx}")

    # Create dataset
    train_dataset = SensorSignalDataset(
        train_data,
        label_to_idx,
        target_length=10000,
        normalize_method='zscore',
        augment=True
    )

    print(f"   Train dataset size: {len(train_dataset)}")
    print(f"   Label counts: {train_dataset.get_label_counts()}")

    # Test getitem
    signal, label = train_dataset[0]
    print(f"\n   Sample tensor shape: {signal.shape}")
    print(f"   Sample label: {label} ({idx_to_label[label]})")

    # Calculate class weights
    class_weights = train_dataset.get_class_weights(method='balanced')
    print(f"\n   Class weights (balanced): {class_weights}")

    # 6. Test DataLoader
    print("\n6. Testing DataLoader...")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_data,
        val_data,
        test_data,
        label_to_idx,
        batch_size=8,
        target_length=10000,
        normalize_method='zscore',
        augment_train=True,
        num_workers=0
    )

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # Test batch
    batch_signals, batch_labels = next(iter(train_loader))
    print(f"\n   Batch signals shape: {batch_signals.shape}")
    print(f"   Batch labels shape: {batch_labels.shape}")
    print(f"   Batch labels: {batch_labels.numpy()}")

    # 7. Test preprocessing pipeline
    print("\n7. Testing preprocessing pipeline...")

    pipeline = create_preprocessing_pipeline(
        normalize_method='zscore',
        target_length=10000,
        filter_type='lowpass',
        cutoff_freq=100.0
    )

    time_proc, current_proc, pipeline_params = pipeline(sample_time, sample_current)
    print(f"   Processed signal length: {len(current_proc)}")
    print(f"   Pipeline parameters: {list(pipeline_params.keys())}")

    # 8. Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nDataset:")
    print(f"  - Total samples: {sum(len(v) for v in dataset.values())}")
    print(f"  - Classes: {len(dataset)}")
    print(f"  - Class distribution: {', '.join([f'{k}: {len(v)}' for k, v in sorted(dataset.items())])}")

    print(f"\nPreprocessing:")
    print(f"  - Target length: 10000")
    print(f"  - Normalization: Z-score")
    print(f"  - Augmentation: Enabled for training")

    print(f"\nDataLoaders:")
    print(f"  - Batch size: 8")
    print(f"  - Train batches: {len(train_loader)} ({len(train_dataset)} samples)")
    print(f"  - Val batches: {len(val_loader)} ({len(val_loader.dataset)} samples)")
    print(f"  - Test batches: {len(test_loader)} ({len(test_loader.dataset)} samples)")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
