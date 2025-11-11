"""
Metrics Tracking and Evaluation
Comprehensive metrics for classification tasks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
import json
from pathlib import Path


class MetricsTracker:
    """
    Track and compute classification metrics.

    Metrics:
    - Accuracy (overall, per-class)
    - Precision, Recall, F1-Score (macro, weighted, per-class)
    - Confusion Matrix
    - ROC-AUC (for binary/multiclass)
    """

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialize metrics tracker.

        Args:
            num_classes: Number of classes
            class_names: Optional class names for reporting
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]

        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []

    def update(self,
               predictions: torch.Tensor,
               labels: torch.Tensor,
               probabilities: Optional[torch.Tensor] = None):
        """
        Update metrics with new batch.

        Args:
            predictions: Predicted class indices (batch_size,)
            labels: True class indices (batch_size,)
            probabilities: Class probabilities (batch_size, num_classes), optional
        """
        # Convert to numpy
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()

        self.all_predictions.extend(predictions)
        self.all_labels.extend(labels)

        if probabilities is not None:
            probabilities = probabilities.cpu().numpy()
            self.all_probabilities.extend(probabilities)

    def compute_accuracy(self) -> float:
        """Compute overall accuracy."""
        correct = np.sum(np.array(self.all_predictions) == np.array(self.all_labels))
        total = len(self.all_labels)
        return correct / total if total > 0 else 0.0

    def compute_per_class_accuracy(self) -> Dict[str, float]:
        """Compute per-class accuracy."""
        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)

        per_class_acc = {}
        for i, class_name in enumerate(self.class_names):
            class_mask = labels == i
            if np.sum(class_mask) > 0:
                acc = np.sum(predictions[class_mask] == labels[class_mask]) / np.sum(class_mask)
                per_class_acc[class_name] = acc
            else:
                per_class_acc[class_name] = 0.0

        return per_class_acc

    def compute_precision_recall_f1(self) -> Dict[str, Dict[str, float]]:
        """
        Compute precision, recall, and F1-score.

        Returns:
            Dictionary with 'macro', 'weighted', and per-class metrics
        """
        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)

        # Compute metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )

        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )

        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )

        # Organize results
        results = {
            'macro': {
                'precision': float(macro_precision),
                'recall': float(macro_recall),
                'f1': float(macro_f1)
            },
            'weighted': {
                'precision': float(weighted_precision),
                'recall': float(weighted_recall),
                'f1': float(weighted_f1)
            },
            'per_class': {}
        }

        for i, class_name in enumerate(self.class_names):
            results['per_class'][class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }

        return results

    def compute_confusion_matrix(self) -> np.ndarray:
        """
        Compute confusion matrix.

        Returns:
            Confusion matrix (num_classes, num_classes)
        """
        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)

        return confusion_matrix(labels, predictions, labels=range(self.num_classes))

    def compute_roc_auc(self) -> Dict[str, float]:
        """
        Compute ROC-AUC scores.

        Returns:
            Dictionary with 'macro', 'weighted', and per-class ROC-AUC
        """
        if not self.all_probabilities:
            return {'error': 'No probabilities available for ROC-AUC computation'}

        labels = np.array(self.all_labels)
        probabilities = np.array(self.all_probabilities)

        # One-hot encode labels
        labels_onehot = np.zeros((len(labels), self.num_classes))
        labels_onehot[np.arange(len(labels)), labels] = 1

        try:
            # Macro average
            macro_auc = roc_auc_score(labels_onehot, probabilities, average='macro')

            # Weighted average
            weighted_auc = roc_auc_score(labels_onehot, probabilities, average='weighted')

            # Per-class
            per_class_auc = {}
            for i, class_name in enumerate(self.class_names):
                if np.sum(labels_onehot[:, i]) > 0:  # Check if class exists
                    auc = roc_auc_score(labels_onehot[:, i], probabilities[:, i])
                    per_class_auc[class_name] = float(auc)
                else:
                    per_class_auc[class_name] = 0.0

            return {
                'macro': float(macro_auc),
                'weighted': float(weighted_auc),
                'per_class': per_class_auc
            }

        except Exception as e:
            return {'error': str(e)}

    def get_classification_report(self) -> str:
        """Get sklearn classification report as string."""
        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)

        return classification_report(
            labels, predictions,
            target_names=self.class_names,
            zero_division=0
        )

    def compute_all_metrics(self) -> Dict:
        """
        Compute all available metrics.

        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'accuracy': self.compute_accuracy(),
            'per_class_accuracy': self.compute_per_class_accuracy(),
            'precision_recall_f1': self.compute_precision_recall_f1(),
            'confusion_matrix': self.compute_confusion_matrix().tolist(),
            'roc_auc': self.compute_roc_auc(),
            'num_samples': len(self.all_labels)
        }

        return metrics

    def save_metrics(self, save_path: str):
        """Save metrics to JSON file."""
        metrics = self.compute_all_metrics()

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved to: {save_path}")

    def print_summary(self):
        """Print formatted metrics summary."""
        print("=" * 80)
        print("METRICS SUMMARY")
        print("=" * 80)

        # Overall accuracy
        print(f"\nOverall Accuracy: {self.compute_accuracy():.4f}")

        # Per-class accuracy
        print("\nPer-Class Accuracy:")
        per_class_acc = self.compute_per_class_accuracy()
        for class_name, acc in per_class_acc.items():
            print(f"  {class_name:<15} {acc:.4f}")

        # Precision, Recall, F1
        prf = self.compute_precision_recall_f1()
        print("\nMacro Average:")
        print(f"  Precision: {prf['macro']['precision']:.4f}")
        print(f"  Recall:    {prf['macro']['recall']:.4f}")
        print(f"  F1-Score:  {prf['macro']['f1']:.4f}")

        print("\nWeighted Average:")
        print(f"  Precision: {prf['weighted']['precision']:.4f}")
        print(f"  Recall:    {prf['weighted']['recall']:.4f}")
        print(f"  F1-Score:  {prf['weighted']['f1']:.4f}")

        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = self.compute_confusion_matrix()
        print("  " + " ".join([f"{name[:8]:>8}" for name in self.class_names]))
        for i, row in enumerate(cm):
            print(f"  {' '.join([f'{val:>8}' for val in row])} [{self.class_names[i]}]")

        # ROC-AUC
        roc_auc = self.compute_roc_auc()
        if 'error' not in roc_auc:
            print("\nROC-AUC:")
            print(f"  Macro:    {roc_auc['macro']:.4f}")
            print(f"  Weighted: {roc_auc['weighted']:.4f}")

        print("=" * 80)


def evaluate_model(model: nn.Module,
                   data_loader: torch.utils.data.DataLoader,
                   device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                   class_names: Optional[List[str]] = None) -> MetricsTracker:
    """
    Evaluate model on a dataset.

    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to run on
        class_names: Optional class names

    Returns:
        MetricsTracker with computed metrics
    """
    model.eval()
    model = model.to(device)

    # Determine number of classes
    with torch.no_grad():
        sample_batch, _ = next(iter(data_loader))
        sample_output = model(sample_batch.to(device))
        num_classes = sample_output.shape[1]

    # Create metrics tracker
    tracker = MetricsTracker(num_classes, class_names)

    # Evaluate
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

            # Update metrics
            tracker.update(predictions, labels, probabilities)

    return tracker


def compute_loss(model: nn.Module,
                data_loader: torch.utils.data.DataLoader,
                criterion: nn.Module,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> float:
    """
    Compute average loss on a dataset.

    Args:
        model: PyTorch model
        data_loader: Data loader
        criterion: Loss function
        device: Device to run on

    Returns:
        Average loss
    """
    model.eval()
    model = model.to(device)

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    return total_loss / total_samples if total_samples > 0 else 0.0


if __name__ == "__main__":
    # Test metrics tracker
    print("Testing MetricsTracker...")

    num_classes = 3
    class_names = ['1um', '2um', '3um']

    tracker = MetricsTracker(num_classes, class_names)

    # Simulate some predictions
    np.random.seed(42)
    for _ in range(10):
        batch_size = 8
        predictions = torch.randint(0, num_classes, (batch_size,))
        labels = torch.randint(0, num_classes, (batch_size,))
        probabilities = torch.softmax(torch.randn(batch_size, num_classes), dim=1)

        tracker.update(predictions, labels, probabilities)

    # Print summary
    tracker.print_summary()

    # Test metrics computation
    print("\nAll Metrics:")
    all_metrics = tracker.compute_all_metrics()
    print(json.dumps(all_metrics, indent=2))

    print("\nMetricsTracker test complete!")
