"""
Comprehensive Model Training Script
Train all three CNN architectures and compare performance.
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from data.data_loader import SensorDataLoader
from data.data_split import stratified_split, create_class_weights
from data.dataset import create_dataloaders
from models.cnn_models import SimpleCNN1D, ResNet1D, MultiScaleCNN1D, count_parameters
from training.trainer import create_trainer
from training.metrics import evaluate_model

import torch
import pandas as pd
import numpy as np


def train_single_model(model_name, model, train_loader, val_loader, test_loader,
                       label_to_idx, class_weights, config, device='cpu'):
    """
    Train a single model with given configuration.

    Args:
        model_name: Name of the model
        model: PyTorch model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        label_to_idx: Label to index mapping
        class_weights: Class weights tensor
        config: Training configuration
        device: Device to train on

    Returns:
        Dictionary with training results
    """
    print("\n" + "=" * 80)
    print(f"TRAINING {model_name}")
    print("=" * 80)

    # Model info
    num_params = count_parameters(model)
    print(f"\nModel: {model_name}")
    print(f"Parameters: {num_params:,}")
    print(f"Device: {device}")

    # Create trainer
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=len(label_to_idx),
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        class_weights=class_weights if config['use_class_weights'] else None,
        device=device
    )

    # Train
    print(f"\nTraining for {config['num_epochs']} epochs...")
    print("-" * 80)

    history = trainer.train(
        num_epochs=config['num_epochs'],
        early_stopping_patience=config['early_stopping_patience'],
        save_best_only=True,
        verbose=True
    )

    print("-" * 80)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    class_names = list(label_to_idx.keys())

    tracker = evaluate_model(
        model=trainer.model,
        data_loader=test_loader,
        device=device,
        class_names=class_names
    )

    # Print test results
    print("\n" + "-" * 80)
    print("TEST SET RESULTS")
    print("-" * 80)
    tracker.print_summary()
    print("-" * 80)

    # Get all metrics
    test_metrics = tracker.compute_all_metrics()

    # Training summary
    summary = trainer.get_summary()

    # Combine results
    results = {
        'model_name': model_name,
        'num_parameters': num_params,
        'config': config,
        'training_summary': summary,
        'training_history': history,
        'test_metrics': test_metrics,
        'best_val_loss': trainer.best_val_loss,
        'best_val_acc': trainer.best_val_acc,
        'final_train_loss': history['train_loss'][-1],
        'final_train_acc': history['train_acc'][-1],
        'final_val_loss': history['val_loss'][-1],
        'final_val_acc': history['val_acc'][-1],
        'test_accuracy': test_metrics['accuracy'],
        'test_macro_f1': test_metrics['precision_recall_f1']['macro']['f1'],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save model
    model_dir = Path("outputs/trained_models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{model_name}_final.pth"

    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'model_class': model_name,
        'config': config,
        'label_mapping': label_to_idx,
        'results': results,
        'history': history,
        'test_metrics': test_metrics
    }, model_path)

    print(f"\nModel saved to: {model_path}")

    return results


def main():
    print("=" * 80)
    print("COMPREHENSIVE MODEL TRAINING - PHASE 6")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Configuration
    config = {
        'base_filters': 32,
        'dropout': 0.5,
        'batch_size': 8,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'early_stopping_patience': 10,
        'use_class_weights': True,
        'target_length': 10000,
        'normalize_method': 'zscore',
        'augment_train': True,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'seed': 42
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['device'] = device

    print("\n" + "=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    for key, value in config.items():
        print(f"{key:<25} {value}")
    print("=" * 80)

    # 1. Load and split data
    print("\n1. Loading and splitting TestData...")
    data_dir = Path(__file__).parent / "TestData"
    loader = SensorDataLoader(str(data_dir))
    dataset = loader.load_dataset("*.csv")

    train_data, val_data, test_data = stratified_split(
        dataset,
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        test_ratio=config['test_ratio'],
        seed=config['seed']
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
        batch_size=config['batch_size'],
        target_length=config['target_length'],
        normalize_method=config['normalize_method'],
        augment_train=config['augment_train'],
        num_workers=0
    )

    print(f"   Train batches: {len(train_loader)} ({len(train_loader.dataset)} samples)")
    print(f"   Val batches: {len(val_loader)} ({len(val_loader.dataset)} samples)")
    print(f"   Test batches: {len(test_loader)} ({len(test_loader.dataset)} samples)")

    # 4. Calculate class weights
    print("\n4. Calculating class weights...")
    class_weights_dict = create_class_weights(train_data, method='balanced')
    class_weights = torch.tensor(
        [class_weights_dict[label] for label in sorted(class_weights_dict.keys())],
        dtype=torch.float32
    )
    print(f"   Class weights: {class_weights.numpy()}")

    # 5. Train all models
    print("\n" + "=" * 80)
    print("TRAINING ALL MODELS")
    print("=" * 80)

    all_results = []

    # Model 1: SimpleCNN1D
    model1 = SimpleCNN1D(
        in_channels=1,
        num_classes=len(label_to_idx),
        base_filters=config['base_filters'],
        dropout=config['dropout']
    )

    results1 = train_single_model(
        'SimpleCNN1D',
        model1,
        train_loader,
        val_loader,
        test_loader,
        label_to_idx,
        class_weights,
        config,
        device
    )
    all_results.append(results1)

    # Model 2: ResNet1D
    model2 = ResNet1D(
        in_channels=1,
        num_classes=len(label_to_idx),
        base_filters=config['base_filters'],
        dropout=config['dropout']
    )

    results2 = train_single_model(
        'ResNet1D',
        model2,
        train_loader,
        val_loader,
        test_loader,
        label_to_idx,
        class_weights,
        config,
        device
    )
    all_results.append(results2)

    # Model 3: MultiScaleCNN1D
    model3 = MultiScaleCNN1D(
        in_channels=1,
        num_classes=len(label_to_idx),
        base_filters=config['base_filters'],
        dropout=config['dropout']
    )

    results3 = train_single_model(
        'MultiScaleCNN1D',
        model3,
        train_loader,
        val_loader,
        test_loader,
        label_to_idx,
        class_weights,
        config,
        device
    )
    all_results.append(results3)

    # 6. Compare results
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    # Create comparison table
    comparison_data = []
    for result in all_results:
        comparison_data.append({
            'Model': result['model_name'],
            'Parameters': f"{result['num_parameters']:,}",
            'Best Val Loss': f"{result['best_val_loss']:.4f}",
            'Best Val Acc': f"{result['best_val_acc']:.4f}",
            'Test Accuracy': f"{result['test_accuracy']:.4f}",
            'Test Macro F1': f"{result['test_macro_f1']:.4f}",
            'Epochs Trained': len(result['training_history']['train_loss'])
        })

    df_comparison = pd.DataFrame(comparison_data)
    print("\n", df_comparison.to_string(index=False))

    # Find best model
    best_model_idx = np.argmax([r['test_accuracy'] for r in all_results])
    best_model = all_results[best_model_idx]

    print("\n" + "-" * 80)
    print("BEST MODEL")
    print("-" * 80)
    print(f"Model: {best_model['model_name']}")
    print(f"Parameters: {best_model['num_parameters']:,}")
    print(f"Test Accuracy: {best_model['test_accuracy']:.4f}")
    print(f"Test Macro F1: {best_model['test_macro_f1']:.4f}")
    print(f"Validation Accuracy: {best_model['best_val_acc']:.4f}")

    # 7. Save comparison results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    results_dir = Path("outputs/training_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save comparison table
    comparison_path = results_dir / f"model_comparison_{timestamp}.csv"
    df_comparison.to_csv(comparison_path, index=False)
    print(f"Comparison table saved to: {comparison_path}")

    # Save detailed results
    results_path = results_dir / f"training_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump({
            'config': config,
            'results': all_results,
            'best_model': best_model['model_name'],
            'timestamp': timestamp
        }, f, indent=2, default=str)
    print(f"Detailed results saved to: {results_path}")

    # 8. Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nAll models trained successfully!")
    print(f"Best model: {best_model['model_name']}")
    print(f"Test accuracy: {best_model['test_accuracy']:.4f}")
    print(f"\nResults saved to: {results_dir}/")
    print(f"Models saved to: outputs/trained_models/")
    print("=" * 80)


if __name__ == "__main__":
    main()
