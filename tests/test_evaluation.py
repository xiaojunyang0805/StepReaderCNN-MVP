"""
Test Script for Evaluation Module (Phase 7)
Tests ModelEvaluator and prediction capabilities.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from evaluation.evaluator import ModelEvaluator, load_all_trained_models, compare_models
from data.data_loader import SensorDataLoader
from data.data_split import stratified_split


def test_model_loading():
    """Test loading trained models."""
    print("=" * 80)
    print("TEST 1: MODEL LOADING")
    print("=" * 80)

    models_dir = Path(__file__).parent / "outputs" / "trained_models"

    if not models_dir.exists():
        print("[X] Models directory not found. Please train models first.")
        return False

    model_files = list(models_dir.glob("*.pth"))
    if not model_files:
        print("[X] No trained models found. Please train models first.")
        return False

    print(f"\nFound {len(model_files)} model files:")
    for model_file in model_files:
        print(f"  - {model_file.name}")

    # Load first model
    print(f"\nLoading model: {model_files[0].name}")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        evaluator = ModelEvaluator(str(model_files[0]), device=device)

        print("\n[OK] Model loaded successfully!")

        # Display model info
        info = evaluator.get_model_info()
        print(f"\nModel Information:")
        print(f"  Model Class: {info['model_class']}")
        print(f"  Parameters: {info['num_parameters']:,}")
        print(f"  Classes: {info['num_classes']}")
        print(f"  Label Mapping: {info['label_mapping']}")
        print(f"  Device: {info['device']}")

        if 'training_results' in info:
            results = info['training_results']
            print(f"\nTraining Results:")
            print(f"  Best Val Acc: {results.get('best_val_acc', 0):.4f}")
            print(f"  Test Accuracy: {results.get('test_accuracy', 0):.4f}")
            print(f"  Test Macro F1: {results.get('test_macro_f1', 0):.4f}")

        return True

    except Exception as e:
        print(f"\n[ERROR] Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_prediction():
    """Test making a single prediction."""
    print("\n" + "=" * 80)
    print("TEST 2: SINGLE PREDICTION")
    print("=" * 80)

    # Load model
    models_dir = Path(__file__).parent / "outputs" / "trained_models"
    model_files = list(models_dir.glob("*.pth"))

    if not model_files:
        print("[X] No trained models found.")
        return False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluator = ModelEvaluator(str(model_files[0]), device=device)

    # Load test data
    print("\nLoading TestData...")
    data_dir = Path(__file__).parent / "TestData"
    loader = SensorDataLoader(str(data_dir))
    dataset = loader.load_dataset("*.csv")

    # Get a sample
    first_class = list(dataset.keys())[0]
    sample = dataset[first_class][0]
    time_data, current_data = sample[0], sample[1]
    filename = sample[2] if len(sample) > 2 else "Unknown"

    print(f"\nTest Sample:")
    print(f"  True Class: {first_class}")
    print(f"  Filename: {filename}")
    print(f"  Signal Length: {len(current_data):,} points")

    # Make prediction
    print("\nMaking prediction...")
    try:
        result = evaluator.predict(time_data, current_data)

        print("\n[OK] Prediction successful!")
        print(f"\nPrediction Results:")
        print(f"  Predicted Class: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Correct: {'[OK]' if result['predicted_class'] == first_class else '[X]'}")

        print(f"\nClass Probabilities:")
        for class_name, prob in result['probabilities'].items():
            bar = "█" * int(prob * 50)
            print(f"  {class_name:10s} [{prob:.2%}] {bar}")

        return True

    except Exception as e:
        print(f"\n[ERROR] Error making prediction: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_prediction():
    """Test batch predictions."""
    print("\n" + "=" * 80)
    print("TEST 3: BATCH PREDICTION")
    print("=" * 80)

    # Load model
    models_dir = Path(__file__).parent / "outputs" / "trained_models"
    model_files = list(models_dir.glob("*.pth"))

    if not model_files:
        print("[X] No trained models found.")
        return False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluator = ModelEvaluator(str(model_files[0]), device=device)

    # Load test data
    print("\nLoading TestData...")
    data_dir = Path(__file__).parent / "TestData"
    loader = SensorDataLoader(str(data_dir))
    dataset = loader.load_dataset("*.csv")

    # Get multiple samples
    signals = []
    true_labels = []

    for class_name, samples in dataset.items():
        for i, sample in enumerate(samples[:2]):  # Take first 2 from each class
            signals.append((sample[0], sample[1]))
            true_labels.append(class_name)

    print(f"\nPredicting on {len(signals)} samples...")

    try:
        results = evaluator.predict_batch(signals)

        print("\n[OK] Batch prediction successful!")

        # Calculate accuracy
        correct = sum(1 for result, true_label in zip(results, true_labels)
                     if result['predicted_class'] == true_label)
        accuracy = correct / len(results)

        print(f"\nBatch Prediction Results:")
        print(f"  Total Samples: {len(results)}")
        print(f"  Correct: {correct}")
        print(f"  Accuracy: {accuracy:.2%}")

        # Show individual predictions
        print(f"\nIndividual Predictions:")
        for i, (result, true_label) in enumerate(zip(results, true_labels)):
            correct_mark = "[OK]" if result['predicted_class'] == true_label else "[X]"
            print(f"  {i+1}. True: {true_label:10s} | Pred: {result['predicted_class']:10s} "
                  f"({result['confidence']:.1%}) {correct_mark}")

        return True

    except Exception as e:
        print(f"\n[ERROR] Error in batch prediction: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_evaluation():
    """Test evaluating on entire dataset."""
    print("\n" + "=" * 80)
    print("TEST 4: DATASET EVALUATION")
    print("=" * 80)

    # Load model
    models_dir = Path(__file__).parent / "outputs" / "trained_models"
    model_files = list(models_dir.glob("*.pth"))

    if not model_files:
        print("[X] No trained models found.")
        return False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluator = ModelEvaluator(str(model_files[0]), device=device)

    # Load and split data
    print("\nLoading and splitting TestData...")
    data_dir = Path(__file__).parent / "TestData"
    loader = SensorDataLoader(str(data_dir))
    dataset = loader.load_dataset("*.csv")

    _, _, test_data = stratified_split(dataset, seed=42)

    num_test_samples = sum(len(samples) for samples in test_data.values())
    print(f"\nEvaluating on test set ({num_test_samples} samples)...")

    try:
        tracker = evaluator.evaluate_dataset(test_data)

        print("\n[OK] Evaluation successful!")

        # Get metrics
        metrics = tracker.compute_all_metrics()

        print(f"\nTest Set Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Macro Precision: {metrics['precision_recall_f1']['macro']['precision']:.4f}")
        print(f"  Macro Recall: {metrics['precision_recall_f1']['macro']['recall']:.4f}")
        print(f"  Macro F1: {metrics['precision_recall_f1']['macro']['f1']:.4f}")

        print(f"\nPer-Class Metrics:")
        for class_name, class_metrics in metrics['precision_recall_f1']['per_class'].items():
            print(f"  {class_name}:")
            print(f"    Precision: {class_metrics['precision']:.4f}")
            print(f"    Recall: {class_metrics['recall']:.4f}")
            print(f"    F1-Score: {class_metrics['f1']:.4f}")

        print(f"\nConfusion Matrix:")
        conf_matrix = metrics['confusion_matrix']
        class_names = list(test_data.keys())

        # Print header
        print(f"{'':12s}", end='')
        for name in class_names:
            print(f"{name[:10]:>12s}", end='')
        print()

        # Print rows
        for i, row in enumerate(conf_matrix):
            print(f"{class_names[i][:10]:12s}", end='')
            for val in row:
                print(f"{val:12d}", end='')
            print()

        return True

    except Exception as e:
        print(f"\n[ERROR] Error in dataset evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_load_all_models():
    """Test loading all trained models."""
    print("\n" + "=" * 80)
    print("TEST 5: LOAD ALL MODELS")
    print("=" * 80)

    models_dir = Path(__file__).parent / "outputs" / "trained_models"

    print("\nLoading all trained models...")

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        evaluators = load_all_trained_models(str(models_dir), device=device)

        if not evaluators:
            print("[X] No models loaded.")
            return False

        print(f"\n[OK] Successfully loaded {len(evaluators)} models!")

        print(f"\nLoaded Models:")
        for model_name, evaluator in evaluators.items():
            info = evaluator.get_model_info()
            print(f"\n  {model_name}:")
            print(f"    Model Class: {info['model_class']}")
            print(f"    Parameters: {info['num_parameters']:,}")
            print(f"    Classes: {info['num_classes']}")

            if 'training_results' in info:
                results = info['training_results']
                print(f"    Val Acc: {results.get('best_val_acc', 0):.4f}")
                print(f"    Test Acc: {results.get('test_accuracy', 0):.4f}")

        return True

    except Exception as e:
        print(f"\n[ERROR] Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_comparison():
    """Test comparing multiple models."""
    print("\n" + "=" * 80)
    print("TEST 6: MODEL COMPARISON")
    print("=" * 80)

    models_dir = Path(__file__).parent / "outputs" / "trained_models"

    # Load all models
    print("\nLoading all models...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluators = load_all_trained_models(str(models_dir), device=device)

    if len(evaluators) < 2:
        print("[WARN] Need at least 2 models for comparison. Skipping test.")
        return True

    # Load test data
    print("\nLoading test data...")
    data_dir = Path(__file__).parent / "TestData"
    loader = SensorDataLoader(str(data_dir))
    dataset = loader.load_dataset("*.csv")

    _, _, test_data = stratified_split(dataset, seed=42)

    # Compare models
    print(f"\nComparing {len(evaluators)} models on test set...")

    try:
        comparison = compare_models(evaluators, test_data)

        print("\n[OK] Comparison successful!")

        print(f"\nComparison Results:")
        print(f"{'Model':<20s} {'Accuracy':>10s} {'Macro F1':>10s} {'Params':>12s}")
        print("-" * 55)

        for model_name, results in comparison.items():
            print(f"{model_name:<20s} "
                  f"{results['accuracy']:>10.4f} "
                  f"{results['macro_f1']:>10.4f} "
                  f"{results['model_info']['num_parameters']:>12,}")

        # Find best model
        best_model = max(comparison.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")

        return True

    except Exception as e:
        print(f"\n[ERROR] Error in model comparison: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("EVALUATION MODULE TEST SUITE - PHASE 7")
    print("=" * 80)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 80)

    tests = [
        ("Model Loading", test_model_loading),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Dataset Evaluation", test_dataset_evaluation),
        ("Load All Models", test_load_all_models),
        ("Model Comparison", test_model_comparison),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[ERROR] Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status:8s} - {test_name}")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print("-" * 80)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 80)

    if passed == total:
        print("\n[SUCCESS] All tests passed! Evaluation module is working correctly.")
    else:
        print(f"\n[WARN] {total - passed} test(s) failed. Please review the errors above.")


if __name__ == "__main__":
    main()
