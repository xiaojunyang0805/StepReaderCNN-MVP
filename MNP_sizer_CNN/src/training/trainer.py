"""
Training Pipeline for CNN Models
Comprehensive training with early stopping, checkpointing, and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Callable, List, Tuple
import time
import json
from datetime import datetime
import numpy as np


class ModelTrainer:
    """
    Comprehensive training pipeline for PyTorch models.

    Features:
    - Training with validation
    - Early stopping
    - Model checkpointing (best & latest)
    - Learning rate scheduling
    - Training history logging
    - Resume from checkpoint
    - Real-time callback support
    """

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 checkpoint_dir: str = 'outputs/checkpoints',
                 log_dir: str = 'outputs/logs'):
        """
        Initialize trainer.

        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        # Create directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

        # Scheduler (optional)
        self.scheduler = None

        # Callbacks
        self.callbacks = []

    def set_scheduler(self, scheduler):
        """Set learning rate scheduler."""
        self.scheduler = scheduler

    def add_callback(self, callback: Callable):
        """Add a callback function to be called during training."""
        self.callbacks.append(callback)

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Batch callback
            if self.callbacks:
                batch_info = {
                    'epoch': self.current_epoch,
                    'batch': batch_idx,
                    'total_batches': len(self.train_loader),
                    'loss': loss.item(),
                    'batch_acc': (predicted == labels).sum().item() / labels.size(0)
                }
                for callback in self.callbacks:
                    callback('batch', batch_info)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = running_loss / total
        val_acc = correct / total

        return val_loss, val_acc

    def train(self,
              num_epochs: int,
              early_stopping_patience: Optional[int] = None,
              save_best_only: bool = True,
              verbose: bool = True) -> Dict:
        """
        Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Stop if no improvement for N epochs (None to disable)
            save_best_only: Only save checkpoint when validation improves
            verbose: Print training progress

        Returns:
            Training history dictionary
        """
        best_epoch = 0
        patience_counter = 0

        if verbose:
            print("=" * 80)
            print("TRAINING START")
            print("=" * 80)
            print(f"Device: {self.device}")
            print(f"Total epochs: {num_epochs}")
            print(f"Train batches: {len(self.train_loader)}")
            print(f"Val batches: {len(self.val_loader)}")
            print(f"Early stopping patience: {early_stopping_patience if early_stopping_patience else 'Disabled'}")
            print("=" * 80)

        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Epoch callback
            if self.callbacks:
                epoch_info = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'learning_rate': current_lr,
                    'time': time.time() - epoch_start
                }
                for callback in self.callbacks:
                    callback('epoch', epoch_info)

            # Check for improvement
            improved = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                improved = True
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                best_epoch = epoch
                improved = True

            # Save checkpoint
            if improved or not save_best_only:
                self.save_checkpoint(
                    is_best=improved,
                    metrics={
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc
                    }
                )
                if improved:
                    patience_counter = 0
            else:
                patience_counter += 1

            # Print progress
            if verbose:
                epoch_time = time.time() - epoch_start
                print(f"Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s) - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                      f"LR: {current_lr:.6f}"
                      f"{' [BEST]' if improved else ''}")

            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    print(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {best_epoch+1}")
                break

        total_time = time.time() - start_time

        if verbose:
            print("=" * 80)
            print("TRAINING COMPLETE")
            print("=" * 80)
            print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
            print(f"Best val loss: {self.best_val_loss:.4f}")
            print(f"Best val acc: {self.best_val_acc:.4f}")
            print("=" * 80)

        # Save final history
        self.save_history()

        return self.history

    def save_checkpoint(self, is_best: bool = False, metrics: Optional[Dict] = None):
        """
        Save model checkpoint.

        Args:
            is_best: Whether this is the best model so far
            metrics: Training metrics to save
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'model_class': self.model.__class__.__name__
        }

        if metrics:
            checkpoint['metrics'] = metrics

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save latest
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """
        Load checkpoint and resume training.

        Args:
            checkpoint_path: Path to checkpoint
            load_optimizer: Whether to load optimizer state
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Load model
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.history = checkpoint.get('history', self.history)

        print(f"Checkpoint loaded from: {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Best val acc: {self.best_val_acc:.4f}")

    def save_history(self):
        """Save training history to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = self.log_dir / f'history_{timestamp}.json'

        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"Training history saved to: {history_path}")

    def get_summary(self) -> Dict:
        """Get training summary."""
        if not self.history['train_loss']:
            return {'status': 'not_started'}

        return {
            'status': 'completed',
            'epochs_trained': len(self.history['train_loss']),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'final_train_loss': self.history['train_loss'][-1],
            'final_train_acc': self.history['train_acc'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'final_val_acc': self.history['val_acc'][-1]
        }


def create_trainer(model: nn.Module,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  num_classes: int,
                  learning_rate: float = 0.001,
                  weight_decay: float = 0.0001,
                  class_weights: Optional[torch.Tensor] = None,
                  device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> ModelTrainer:
    """
    Factory function to create a trainer with standard configuration.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_classes: Number of classes
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        class_weights: Class weights for imbalanced datasets
        device: Device to train on

    Returns:
        Configured ModelTrainer instance
    """
    # Loss function
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Create trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )

    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    trainer.set_scheduler(scheduler)

    return trainer


if __name__ == "__main__":
    # Example usage
    print("Testing ModelTrainer...")

    from pathlib import Path
    import sys

    # Add src to path
    src_path = Path(__file__).parent.parent
    sys.path.insert(0, str(src_path))

    from models.cnn_models import SimpleCNN1D
    from data.data_loader import SensorDataLoader
    from data.data_split import stratified_split
    from data.dataset import create_dataloaders

    # Load data
    data_dir = Path(__file__).parent.parent.parent / "TestData"
    loader = SensorDataLoader(str(data_dir))
    dataset = loader.load_dataset("*.csv")

    train_data, val_data, test_data = stratified_split(
        dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    # Create label mapping
    label_to_idx = {label: idx for idx, label in enumerate(sorted(dataset.keys()))}

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, label_to_idx,
        batch_size=4, target_length=10000, num_workers=0
    )

    # Create model and trainer
    model = SimpleCNN1D(in_channels=1, num_classes=3)
    trainer = create_trainer(
        model, train_loader, val_loader, num_classes=3, learning_rate=0.001
    )

    # Train for a few epochs
    print("\nTraining for 3 epochs...")
    history = trainer.train(num_epochs=3, verbose=True)

    # Print summary
    print("\nTraining Summary:")
    summary = trainer.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\nTrainer test complete!")
