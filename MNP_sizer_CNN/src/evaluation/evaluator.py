"""
Model Evaluation Utilities
Load trained models and perform predictions and evaluations.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json

from models.cnn_models import SimpleCNN1D, ResNet1D, MultiScaleCNN1D
from data.preprocessing import SignalPreprocessor
from training.metrics import MetricsTracker


class ModelEvaluator:
    """
    Comprehensive model evaluation and prediction interface.

    Features:
    - Load trained models from checkpoints
    - Make predictions on new signals
    - Compute confidence scores
    - Batch evaluation
    - Model comparison
    """

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize evaluator with a trained model.

        Args:
            model_path: Path to model checkpoint (.pth file)
            device: Device to run inference on
        """
        self.model_path = Path(model_path)
        self.device = device

        # Load checkpoint
        self.checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)

        # Extract model info
        self.model_class = self.checkpoint.get('model_class', 'Unknown')
        self.config = self.checkpoint.get('config', {})
        self.label_mapping = self.checkpoint.get('label_mapping', {})
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
        self.num_classes = len(self.label_mapping)

        # Initialize model
        self.model = self._create_model()
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        # Initialize preprocessor
        self.preprocessor = SignalPreprocessor()

        print(f"Model loaded: {self.model_class}")
        print(f"Classes: {list(self.label_mapping.keys())}")
        print(f"Device: {device}")

    def _create_model(self) -> nn.Module:
        """Create model instance based on checkpoint."""
        if self.model_class == 'SimpleCNN1D':
            model = SimpleCNN1D(
                in_channels=1,
                num_classes=self.num_classes,
                base_filters=self.config.get('base_filters', 32),
                dropout=self.config.get('dropout', 0.5)
            )
        elif self.model_class == 'ResNet1D':
            model = ResNet1D(
                in_channels=1,
                num_classes=self.num_classes,
                base_filters=self.config.get('base_filters', 32),
                dropout=self.config.get('dropout', 0.5)
            )
        elif self.model_class == 'MultiScaleCNN1D':
            model = MultiScaleCNN1D(
                in_channels=1,
                num_classes=self.num_classes,
                base_filters=self.config.get('base_filters', 32),
                dropout=self.config.get('dropout', 0.5)
            )
        else:
            raise ValueError(f"Unknown model class: {self.model_class}")

        return model

    def preprocess_signal(self,
                         time_data: np.ndarray,
                         current_data: np.ndarray,
                         target_length: Optional[int] = None) -> torch.Tensor:
        """
        Preprocess a signal for prediction.

        Args:
            time_data: Time values
            current_data: Current values
            target_length: Target signal length (from config if not specified)

        Returns:
            Preprocessed tensor (1, 1, length)
        """
        if target_length is None:
            target_length = self.config.get('target_length', 10000)

        normalize_method = self.config.get('normalize_method', 'zscore')

        # Normalize length
        current_normalized = self.preprocessor.normalize_length(
            current_data,
            target_length
        )

        # Normalize values
        if normalize_method == 'zscore':
            current_normalized, _ = self.preprocessor.normalize_zscore(current_normalized)
        elif normalize_method == 'minmax':
            current_normalized, _ = self.preprocessor.normalize_minmax(current_normalized)
        elif normalize_method == 'robust':
            current_normalized, _ = self.preprocessor.normalize_robust(current_normalized)

        # Convert to tensor (1, 1, length)
        tensor = torch.from_numpy(current_normalized).float().unsqueeze(0).unsqueeze(0)

        return tensor

    def predict(self,
               time_data: np.ndarray,
               current_data: np.ndarray,
               return_probabilities: bool = True) -> Dict:
        """
        Make prediction on a single signal.

        Args:
            time_data: Time values
            current_data: Current values
            return_probabilities: Whether to return class probabilities

        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        tensor = self.preprocess_signal(time_data, current_data)
        tensor = tensor.to(self.device)

        # Predict
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_idx].item()

        # Get predicted class
        predicted_class = self.idx_to_label[predicted_idx]

        result = {
            'predicted_class': predicted_class,
            'predicted_idx': predicted_idx,
            'confidence': confidence
        }

        if return_probabilities:
            probs_dict = {
                self.idx_to_label[i]: probabilities[0, i].item()
                for i in range(self.num_classes)
            }
            result['probabilities'] = probs_dict

        return result

    def predict_batch(self,
                     signals: List[Tuple[np.ndarray, np.ndarray]]) -> List[Dict]:
        """
        Make predictions on multiple signals.

        Args:
            signals: List of (time_data, current_data) tuples

        Returns:
            List of prediction dictionaries
        """
        results = []

        for time_data, current_data in signals:
            result = self.predict(time_data, current_data)
            results.append(result)

        return results

    def evaluate_dataset(self,
                        data_dict: Dict[str, List[Tuple]],
                        class_names: Optional[List[str]] = None) -> MetricsTracker:
        """
        Evaluate model on a dataset.

        Args:
            data_dict: Dictionary mapping labels to list of (time, current, filename) tuples
            class_names: Optional list of class names

        Returns:
            MetricsTracker with evaluation results
        """
        if class_names is None:
            class_names = list(self.label_mapping.keys())

        tracker = MetricsTracker(self.num_classes, class_names)

        all_predictions = []
        all_labels = []
        all_probabilities = []

        for label, samples in data_dict.items():
            label_idx = self.label_mapping[label]

            for sample in samples:
                time_data, current_data = sample[0], sample[1]

                # Predict
                result = self.predict(time_data, current_data)

                all_predictions.append(result['predicted_idx'])
                all_labels.append(label_idx)
                all_probabilities.append([
                    result['probabilities'][self.idx_to_label[i]]
                    for i in range(self.num_classes)
                ])

        # Update tracker
        predictions_tensor = torch.tensor(all_predictions)
        labels_tensor = torch.tensor(all_labels)
        probabilities_tensor = torch.tensor(all_probabilities)

        tracker.update(predictions_tensor, labels_tensor, probabilities_tensor)

        return tracker

    def get_model_info(self) -> Dict:
        """Get model information."""
        info = {
            'model_class': self.model_class,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'config': self.config,
            'label_mapping': self.label_mapping,
            'num_classes': self.num_classes,
            'device': self.device,
            'checkpoint_path': str(self.model_path)
        }

        # Add training results if available
        if 'results' in self.checkpoint:
            info['training_results'] = self.checkpoint['results']

        return info


def load_all_trained_models(models_dir: str = 'outputs/trained_models',
                           device: str = 'cpu') -> Dict[str, ModelEvaluator]:
    """
    Load all trained models from a directory.

    Args:
        models_dir: Directory containing model checkpoints
        device: Device to load models on

    Returns:
        Dictionary mapping model names to ModelEvaluator instances
    """
    models_dir = Path(models_dir)

    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return {}

    evaluators = {}

    for model_path in models_dir.glob("*.pth"):
        try:
            model_name = model_path.stem
            evaluator = ModelEvaluator(str(model_path), device=device)
            evaluators[model_name] = evaluator
            print(f"Loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading {model_path.name}: {e}")

    return evaluators


def compare_models(evaluators: Dict[str, ModelEvaluator],
                  test_data: Dict[str, List[Tuple]]) -> Dict:
    """
    Compare multiple models on the same test set.

    Args:
        evaluators: Dictionary of ModelEvaluator instances
        test_data: Test dataset

    Returns:
        Comparison results dictionary
    """
    results = {}

    for model_name, evaluator in evaluators.items():
        print(f"\nEvaluating {model_name}...")

        tracker = evaluator.evaluate_dataset(test_data)
        metrics = tracker.compute_all_metrics()

        results[model_name] = {
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['precision_recall_f1']['macro']['f1'],
            'macro_precision': metrics['precision_recall_f1']['macro']['precision'],
            'macro_recall': metrics['precision_recall_f1']['macro']['recall'],
            'per_class_accuracy': metrics['per_class_accuracy'],
            'confusion_matrix': metrics['confusion_matrix'],
            'model_info': evaluator.get_model_info()
        }

    return results


if __name__ == "__main__":
    # Example usage
    print("Testing ModelEvaluator...")

    import sys
    from pathlib import Path

    # Add src to path
    src_path = Path(__file__).parent.parent
    sys.path.insert(0, str(src_path))

    from data.data_loader import SensorDataLoader
    from data.data_split import stratified_split

    # Check if trained models exist
    models_dir = Path(__file__).parent.parent.parent / "outputs" / "trained_models"

    if not models_dir.exists() or not list(models_dir.glob("*.pth")):
        print("No trained models found. Please run train_models.py first.")
    else:
        # Load test data
        data_dir = Path(__file__).parent.parent.parent / "TestData"
        loader = SensorDataLoader(str(data_dir))
        dataset = loader.load_dataset("*.csv")

        _, _, test_data = stratified_split(dataset, seed=42)

        # Load all models
        print("\nLoading trained models...")
        evaluators = load_all_trained_models(str(models_dir))

        if evaluators:
            # Compare models
            print("\nComparing models on test set...")
            comparison = compare_models(evaluators, test_data)

            print("\n" + "=" * 80)
            print("MODEL COMPARISON RESULTS")
            print("=" * 80)

            for model_name, results in comparison.items():
                print(f"\n{model_name}:")
                print(f"  Accuracy: {results['accuracy']:.4f}")
                print(f"  Macro F1: {results['macro_f1']:.4f}")
                print(f"  Parameters: {results['model_info']['num_parameters']:,}")

        print("\nModelEvaluator test complete!")
