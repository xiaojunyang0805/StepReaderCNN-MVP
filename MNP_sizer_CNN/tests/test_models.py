"""
Test script for CNN models with preprocessing pipeline.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from data.data_loader import SensorDataLoader
from data.data_split import stratified_split
from data.dataset import create_dataloaders
from models.cnn_models import SimpleCNN1D, ResNet1D, MultiScaleCNN1D, count_parameters
from models.model_utils import print_model_summary, save_model, load_model

import torch
import torch.nn as nn


def main():
    print("=" * 80)
    print("CNN MODELS + PREPROCESSING INTEGRATION TEST")
    print("=" * 80)

    # 1. Load and split data
    print("\n1. Loading and splitting TestData...")
    data_dir = Path(__file__).parent / "TestData"
    loader = SensorDataLoader(str(data_dir))
    dataset = loader.load_dataset("*.csv")

    train_data, val_data, test_data = stratified_split(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )

    print(f"   Train: {sum(len(v) for v in train_data.values())} samples")
    print(f"   Val: {sum(len(v) for v in val_data.values())} samples")
    print(f"   Test: {sum(len(v) for v in test_data.values())} samples")

    # 2. Create label mapping
    unique_labels = sorted(dataset.keys())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    print(f"\n2. Label mapping: {label_to_idx}")

    # 3. Create dataloaders
    print("\n3. Creating DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data,
        val_data,
        test_data,
        label_to_idx,
        batch_size=4,  # Small batch for testing
        target_length=10000,
        normalize_method='zscore',
        augment_train=True,
        num_workers=0
    )

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # Get a sample batch
    sample_batch, sample_labels = next(iter(train_loader))
    print(f"\n   Sample batch shape: {sample_batch.shape}")
    print(f"   Sample labels shape: {sample_labels.shape}")

    # 4. Create models
    print("\n4. Creating CNN Models...")
    print("-" * 80)

    models = {
        'SimpleCNN1D': SimpleCNN1D(in_channels=1, num_classes=3, base_filters=32, dropout=0.5),
        'ResNet1D': ResNet1D(in_channels=1, num_classes=3, base_filters=32, dropout=0.5),
        'MultiScaleCNN1D': MultiScaleCNN1D(in_channels=1, num_classes=3, base_filters=32, dropout=0.5)
    }

    for name, model in models.items():
        params = count_parameters(model)
        print(f"   {name:<20} Parameters: {params:,}")

    # 5. Test forward pass with real data
    print("\n5. Testing Forward Pass with Real Data...")
    print("-" * 80)

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            output = model(sample_batch)

        print(f"   {name:<20} Input: {sample_batch.shape} -> Output: {output.shape}")

        # Check output shape
        assert output.shape == (sample_batch.shape[0], 3), \
            f"Output shape mismatch for {name}"

    # 6. Test loss calculation
    print("\n6. Testing Loss Calculation...")
    print("-" * 80)

    criterion = nn.CrossEntropyLoss()

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            output = model(sample_batch)
            loss = criterion(output, sample_labels)

        print(f"   {name:<20} Loss: {loss.item():.4f}")

    # 7. Test save/load
    print("\n7. Testing Model Save/Load...")
    print("-" * 80)

    model_to_test = models['SimpleCNN1D']
    save_path = Path("outputs/test_model.pth")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    save_model(model_to_test, str(save_path), epoch=0, metrics={'loss': 0.5, 'acc': 0.8})

    # Load
    loaded_model = SimpleCNN1D(in_channels=1, num_classes=3)
    loaded_model, _, info = load_model(loaded_model, str(save_path))

    # Verify outputs match
    model_to_test.eval()
    loaded_model.eval()

    with torch.no_grad():
        output1 = model_to_test(sample_batch)
        output2 = loaded_model(sample_batch)

    assert torch.allclose(output1, output2), "Loaded model outputs don't match!"
    print(f"   Model save/load verified: outputs match")

    # 8. Test full batch
    print("\n8. Testing Full Batch Processing...")
    print("-" * 80)

    model = models['SimpleCNN1D']
    model.eval()

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_signals, batch_labels in test_loader:
            outputs = model(batch_signals)
            predictions = torch.argmax(outputs, dim=1)

            total_correct += (predictions == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

    accuracy = total_correct / total_samples
    print(f"   Test accuracy (untrained): {accuracy:.2%}")
    print(f"   Total samples processed: {total_samples}")

    # 9. Summary
    print("\n" + "=" * 80)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 80)

    print(f"\nData Pipeline:")
    print(f"  - Loaded {sum(len(v) for v in dataset.values())} samples from TestData")
    print(f"  - Split into train/val/test: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")
    print(f"  - Batch shape: (4, 1, 10000)")

    print(f"\nModels:")
    for name, model in models.items():
        print(f"  - {name}: {count_parameters(model):,} parameters")

    print(f"\nAll Tests:")
    print(f"  - Data loading: PASS")
    print(f"  - DataLoader creation: PASS")
    print(f"  - Forward pass: PASS")
    print(f"  - Loss calculation: PASS")
    print(f"  - Model save/load: PASS")
    print(f"  - Batch processing: PASS")

    print("\n" + "=" * 80)
    print("ALL INTEGRATION TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
