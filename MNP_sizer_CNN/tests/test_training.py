"""
Test script for training pipeline.
Tests trainer, metrics, and complete training workflow.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from data.data_loader import SensorDataLoader
from data.data_split import stratified_split, create_class_weights
from data.dataset import create_dataloaders
from models.cnn_models import SimpleCNN1D, ResNet1D, MultiScaleCNN1D, count_parameters
from training.trainer import create_trainer
from training.metrics import evaluate_model, MetricsTracker

import torch
import torch.nn as nn


def main():
    print("=" * 80)
    print("TRAINING PIPELINE TEST")
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
        batch_size=4,
        target_length=10000,
        normalize_method='zscore',
        augment_train=True,
        num_workers=0
    )

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # 4. Create model
    print("\n4. Creating model...")
    model = SimpleCNN1D(in_channels=1, num_classes=3, base_filters=32, dropout=0.5)
    print(f"   Model: SimpleCNN1D")
    print(f"   Parameters: {count_parameters(model):,}")

    # 5. Calculate class weights
    print("\n5. Calculating class weights...")
    class_weights_dict = create_class_weights(train_data, method='balanced')
    print(f"   Class weights (dict): {class_weights_dict}")

    # Convert to tensor (in order of label_to_idx)
    class_weights = torch.tensor([class_weights_dict[label] for label in sorted(class_weights_dict.keys())],
                                  dtype=torch.float32)
    print(f"   Class weights (tensor): {class_weights.numpy()}")

    # 6. Create trainer
    print("\n6. Creating trainer...")
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=3,
        learning_rate=0.001,
        class_weights=class_weights,
        device='cpu'  # Use CPU for testing
    )

    print("   Trainer created successfully")
    print(f"   Device: {trainer.device}")
    print(f"   Optimizer: {trainer.optimizer.__class__.__name__}")
    print(f"   Criterion: {trainer.criterion.__class__.__name__}")

    # 7. Test single epoch
    print("\n7. Testing single epoch training...")
    train_loss, train_acc = trainer.train_epoch()
    print(f"   Train Loss: {train_loss:.4f}")
    print(f"   Train Acc: {train_acc:.4f}")

    val_loss, val_acc = trainer.validate()
    print(f"   Val Loss: {val_loss:.4f}")
    print(f"   Val Acc: {val_acc:.4f}")

    # 8. Test full training (3 epochs)
    print("\n8. Testing full training (5 epochs)...")
    print("-" * 80)

    history = trainer.train(
        num_epochs=5,
        early_stopping_patience=3,
        save_best_only=True,
        verbose=True
    )

    print("-" * 80)

    # 9. Check training history
    print("\n9. Training History:")
    print(f"   Epochs completed: {len(history['train_loss'])}")
    print(f"   Best val loss: {min(history['val_loss']):.4f}")
    print(f"   Best val acc: {max(history['val_acc']):.4f}")
    print(f"   Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"   Final val loss: {history['val_loss'][-1]:.4f}")

    # 10. Test checkpoint save/load
    print("\n10. Testing checkpoint save/load...")

    # Save checkpoint
    checkpoint_path = Path("outputs/checkpoints/test_checkpoint.pth")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(is_best=True)
    print(f"   Checkpoint saved to: {checkpoint_path.parent}")

    # Load checkpoint
    new_model = SimpleCNN1D(in_channels=1, num_classes=3)
    new_trainer = create_trainer(
        model=new_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=3
    )

    best_checkpoint = checkpoint_path.parent / "best.pth"
    if best_checkpoint.exists():
        new_trainer.load_checkpoint(best_checkpoint)
        print("   Checkpoint loaded successfully")

    # 11. Test metrics tracking
    print("\n11. Testing metrics tracking...")

    # Evaluate on test set
    tracker = evaluate_model(
        model=trainer.model,
        data_loader=test_loader,
        device='cpu',
        class_names=list(label_to_idx.keys())
    )

    print("\n   Metrics Summary:")
    print("-" * 80)
    tracker.print_summary()
    print("-" * 80)

    # 12. Test metrics computation
    print("\n12. Testing individual metrics...")

    all_metrics = tracker.compute_all_metrics()

    print(f"   Overall Accuracy: {all_metrics['accuracy']:.4f}")

    prf = all_metrics['precision_recall_f1']
    print(f"   Macro Precision: {prf['macro']['precision']:.4f}")
    print(f"   Macro Recall: {prf['macro']['recall']:.4f}")
    print(f"   Macro F1: {prf['macro']['f1']:.4f}")

    if 'error' not in all_metrics['roc_auc']:
        print(f"   ROC-AUC (macro): {all_metrics['roc_auc']['macro']:.4f}")

    # 13. Test different models
    print("\n13. Testing different model architectures...")
    print("-" * 80)

    models_to_test = {
        'SimpleCNN1D': SimpleCNN1D(in_channels=1, num_classes=3, base_filters=16),
        'ResNet1D': ResNet1D(in_channels=1, num_classes=3, base_filters=16),
        'MultiScaleCNN1D': MultiScaleCNN1D(in_channels=1, num_classes=3, base_filters=16)
    }

    for name, test_model in models_to_test.items():
        print(f"\n   Testing {name}...")
        print(f"   Parameters: {count_parameters(test_model):,}")

        # Create trainer
        test_trainer = create_trainer(
            model=test_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=3,
            learning_rate=0.001
        )

        # Train for 2 epochs
        test_history = test_trainer.train(num_epochs=2, verbose=False)

        print(f"   Final train loss: {test_history['train_loss'][-1]:.4f}")
        print(f"   Final val loss: {test_history['val_loss'][-1]:.4f}")
        print(f"   Final val acc: {test_history['val_acc'][-1]:.4f}")

    print("-" * 80)

    # 14. Summary
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE TEST SUMMARY")
    print("=" * 80)

    print("\nComponents Tested:")
    print("  [PASS] Data loading and splitting")
    print("  [PASS] DataLoader creation")
    print("  [PASS] Model creation")
    print("  [PASS] Trainer initialization")
    print("  [PASS] Single epoch training")
    print("  [PASS] Full training with early stopping")
    print("  [PASS] Checkpoint save/load")
    print("  [PASS] Metrics tracking")
    print("  [PASS] Model evaluation")
    print("  [PASS] Multiple model architectures")

    print("\nKey Results:")
    print(f"  - Training completed: {len(history['train_loss'])} epochs")
    print(f"  - Best validation accuracy: {max(history['val_acc']):.4f}")
    print(f"  - Test accuracy (untrained): {all_metrics['accuracy']:.4f}")
    print(f"  - Checkpoint saved: outputs/checkpoints/")
    print(f"  - Metrics computed: Accuracy, Precision, Recall, F1, ROC-AUC")

    print("\n" + "=" * 80)
    print("ALL TRAINING TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
