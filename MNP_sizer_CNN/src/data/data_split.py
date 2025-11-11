"""
Data Splitting Utilities
Train/validation/test splitting with stratification for imbalanced datasets.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold
import random


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)


def stratified_split(data_by_label: Dict[str, List],
                    train_ratio: float = 0.7,
                    val_ratio: float = 0.15,
                    test_ratio: float = 0.15,
                    seed: int = 42) -> Tuple[Dict, Dict, Dict]:
    """
    Split data into train/val/test sets with stratification.

    Args:
        data_by_label: Dictionary mapping labels to list of samples
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dict, val_dict, test_dict)
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    set_seed(seed)

    train_data = {}
    val_data = {}
    test_data = {}

    for label, samples in data_by_label.items():
        n_samples = len(samples)

        # Shuffle samples
        indices = np.random.permutation(n_samples)
        shuffled_samples = [samples[i] for i in indices]

        # Calculate split sizes
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        # Remaining goes to test to handle rounding

        # Split
        train_data[label] = shuffled_samples[:n_train]
        val_data[label] = shuffled_samples[n_train:n_train + n_val]
        test_data[label] = shuffled_samples[n_train + n_val:]

    return train_data, val_data, test_data


def create_kfold_splits(data_by_label: Dict[str, List],
                       n_splits: int = 5,
                       seed: int = 42) -> List[Tuple[Dict, Dict]]:
    """
    Create stratified K-fold cross-validation splits.

    Args:
        data_by_label: Dictionary mapping labels to list of samples
        n_splits: Number of folds
        seed: Random seed for reproducibility

    Returns:
        List of (train_dict, val_dict) tuples for each fold
    """
    set_seed(seed)

    # Flatten data into arrays
    all_samples = []
    all_labels = []

    for label, samples in data_by_label.items():
        all_samples.extend(samples)
        all_labels.extend([label] * len(samples))

    all_samples = np.array(all_samples, dtype=object)
    all_labels = np.array(all_labels)

    # Create stratified K-fold splitter
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []

    for train_idx, val_idx in skf.split(all_samples, all_labels):
        # Create train dict
        train_samples = all_samples[train_idx]
        train_labels = all_labels[train_idx]

        train_dict = {}
        for label in np.unique(train_labels):
            mask = train_labels == label
            train_dict[label] = train_samples[mask].tolist()

        # Create val dict
        val_samples = all_samples[val_idx]
        val_labels = all_labels[val_idx]

        val_dict = {}
        for label in np.unique(val_labels):
            mask = val_labels == label
            val_dict[label] = val_samples[mask].tolist()

        folds.append((train_dict, val_dict))

    return folds


def get_split_statistics(train_data: Dict, val_data: Dict, test_data: Dict) -> Dict:
    """
    Get statistics about data splits.

    Args:
        train_data: Training data dictionary
        val_data: Validation data dictionary
        test_data: Test data dictionary

    Returns:
        Dictionary with split statistics
    """
    stats = {
        'train': {},
        'val': {},
        'test': {},
        'total': {}
    }

    # Count per split
    for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        for label, samples in split_data.items():
            stats[split_name][label] = len(samples)

    # Total counts
    all_labels = set(train_data.keys()) | set(val_data.keys()) | set(test_data.keys())

    for label in all_labels:
        train_count = stats['train'].get(label, 0)
        val_count = stats['val'].get(label, 0)
        test_count = stats['test'].get(label, 0)
        total_count = train_count + val_count + test_count

        stats['total'][label] = total_count

    # Add summary
    stats['summary'] = {
        'train_total': sum(stats['train'].values()),
        'val_total': sum(stats['val'].values()),
        'test_total': sum(stats['test'].values()),
        'grand_total': sum(stats['total'].values())
    }

    return stats


def print_split_statistics(stats: Dict):
    """
    Print formatted split statistics.

    Args:
        stats: Statistics dictionary from get_split_statistics()
    """
    print("=" * 60)
    print("DATA SPLIT STATISTICS")
    print("=" * 60)

    # Per-class breakdown
    labels = sorted(stats['total'].keys())

    print(f"\n{'Label':<10} {'Train':>10} {'Val':>10} {'Test':>10} {'Total':>10}")
    print("-" * 60)

    for label in labels:
        train = stats['train'].get(label, 0)
        val = stats['val'].get(label, 0)
        test = stats['test'].get(label, 0)
        total = stats['total'][label]

        print(f"{label:<10} {train:>10} {val:>10} {test:>10} {total:>10}")

    # Summary
    print("-" * 60)
    summary = stats['summary']
    print(f"{'TOTAL':<10} {summary['train_total']:>10} {summary['val_total']:>10} "
          f"{summary['test_total']:>10} {summary['grand_total']:>10}")

    # Percentages
    total = summary['grand_total']
    train_pct = (summary['train_total'] / total) * 100
    val_pct = (summary['val_total'] / total) * 100
    test_pct = (summary['test_total'] / total) * 100

    print("\n" + "=" * 60)
    print(f"Train: {train_pct:.1f}% | Val: {val_pct:.1f}% | Test: {test_pct:.1f}%")
    print("=" * 60)


def flatten_dataset(data_dict: Dict[str, List]) -> Tuple[List, List]:
    """
    Flatten dataset dictionary into lists of samples and labels.

    Args:
        data_dict: Dictionary mapping labels to list of samples

    Returns:
        Tuple of (samples_list, labels_list)
    """
    samples = []
    labels = []

    for label, sample_list in data_dict.items():
        samples.extend(sample_list)
        labels.extend([label] * len(sample_list))

    return samples, labels


def create_class_weights(data_dict: Dict[str, List],
                        method: str = 'balanced') -> Dict[str, float]:
    """
    Create class weights for handling imbalanced datasets.

    Args:
        data_dict: Dictionary mapping labels to list of samples
        method: Weight calculation method ('balanced' or 'sqrt')
            - 'balanced': n_samples / (n_classes * n_samples_per_class)
            - 'sqrt': Uses square root to reduce weight imbalance

    Returns:
        Dictionary mapping labels to weights
    """
    # Count samples per class
    class_counts = {label: len(samples) for label, samples in data_dict.items()}

    n_classes = len(class_counts)
    n_total = sum(class_counts.values())

    weights = {}

    if method == 'balanced':
        # Balanced: higher weight for minority classes
        for label, count in class_counts.items():
            weights[label] = n_total / (n_classes * count)

    elif method == 'sqrt':
        # Square root: less extreme than balanced
        max_count = max(class_counts.values())

        for label, count in class_counts.items():
            weights[label] = np.sqrt(max_count / count)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize weights to sum to n_classes
    total_weight = sum(weights.values())
    weights = {label: (weight / total_weight) * n_classes
               for label, weight in weights.items()}

    return weights


def oversample_minority_classes(train_data: Dict[str, List],
                                strategy: str = 'match_max',
                                target_count: Optional[int] = None,
                                seed: int = 42) -> Dict[str, List]:
    """
    Oversample minority classes to balance dataset.

    Args:
        train_data: Training data dictionary
        strategy: Oversampling strategy
            - 'match_max': Oversample all classes to match largest class
            - 'target': Oversample to specific target count
        target_count: Target count per class (required for 'target' strategy)
        seed: Random seed for reproducibility

    Returns:
        Oversampled data dictionary
    """
    set_seed(seed)

    class_counts = {label: len(samples) for label, samples in train_data.items()}
    max_count = max(class_counts.values())

    if strategy == 'match_max':
        target = max_count
    elif strategy == 'target':
        if target_count is None:
            raise ValueError("target_count must be specified for 'target' strategy")
        target = target_count
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    oversampled_data = {}

    for label, samples in train_data.items():
        current_count = len(samples)

        if current_count >= target:
            # No oversampling needed
            oversampled_data[label] = samples.copy()
        else:
            # Oversample by repeating samples
            n_repeats = target // current_count
            n_extra = target % current_count

            # Repeat all samples
            oversampled = samples * n_repeats

            # Add random samples for remainder
            if n_extra > 0:
                extra_indices = np.random.choice(current_count, n_extra, replace=False)
                extra_samples = [samples[i] for i in extra_indices]
                oversampled.extend(extra_samples)

            oversampled_data[label] = oversampled

    return oversampled_data


if __name__ == "__main__":
    # Example usage
    print("Testing data splitting utilities...")

    # Create dummy data
    dummy_data = {
        '1um': [(np.random.randn(100), np.random.randn(100), f'1um_{i}') for i in range(7)],
        '2um': [(np.random.randn(150), np.random.randn(150), f'2um_{i}') for i in range(9)],
        '3um': [(np.random.randn(120), np.random.randn(120), f'3um_{i}') for i in range(26)]
    }

    print("\n1. Stratified Train/Val/Test Split:")
    train, val, test = stratified_split(dummy_data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    stats = get_split_statistics(train, val, test)
    print_split_statistics(stats)

    print("\n2. Class Weights:")
    weights = create_class_weights(train, method='balanced')
    for label, weight in sorted(weights.items()):
        print(f"{label}: {weight:.3f}")

    print("\n3. Oversampling Minority Classes:")
    oversampled = oversample_minority_classes(train, strategy='match_max')
    print(f"Original train counts: {', '.join([f'{k}: {len(v)}' for k, v in train.items()])}")
    print(f"Oversampled counts: {', '.join([f'{k}: {len(v)}' for k, v in oversampled.items()])}")
