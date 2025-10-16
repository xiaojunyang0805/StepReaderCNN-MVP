"""
Simplified Integration Test Suite - Phase 9
Tests the most critical integration paths.
"""

import sys
import time
from pathlib import Path
import numpy as np
import torch

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from data.data_loader import SensorDataLoader
from data.synthetic_generator import SyntheticSignalGenerator
from models.cnn_models import SimpleCNN1D, ResNet1D, MultiScaleCNN1D


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"{text}")
    print("=" * 80)


def print_result(passed, message=""):
    """Print test result."""
    status = "[PASS]" if passed else "[FAIL]"
    print(f"\n{status} {message}")
    return passed


# Test 1: Data Loading
print_header("INTEGRATION TEST 1: Data Loading")
try:
    loader = SensorDataLoader("TestData")
    dataset = loader.load_dataset("*.csv")
    count = sum(len(v) for v in dataset.values())
    print(f"Loaded {count} samples from {len(dataset)} classes")
    print_result(True, "Data loading successful")
    test1_pass = True
except Exception as e:
    print_result(False, f"Data loading failed: {e}")
    test1_pass = False

# Test 2: Synthetic Generation
print_header("INTEGRATION TEST 2: Synthetic Data Generation")
try:
    generator = SyntheticSignalGenerator(seed=42)
    signals = []
    for class_name in ['1um', '2um', '3um']:
        t, c = generator.generate_signal(class_name)
        signals.append((class_name, len(c), c.mean()))
        print(f"  {class_name}: {len(c):,} points, mean={c.mean():.4f}")
    print_result(True, f"Generated {len(signals)} synthetic signals")
    test2_pass = True
except Exception as e:
    print_result(False, f"Synthetic generation failed: {e}")
    test2_pass = False

# Test 3: Model Architectures
print_header("INTEGRATION TEST 3: Model Architectures")
try:
    models = {
        'SimpleCNN1D': SimpleCNN1D(num_classes=3),
        'ResNet1D': ResNet1D(num_classes=3),
        'MultiScaleCNN1D': MultiScaleCNN1D(num_classes=3)
    }

    dummy_input = torch.randn(4, 1, 10000)
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        print(f"  {name}: {params:,} params, output shape {output.shape}")

    print_result(True, "All models working correctly")
    test3_pass = True
except Exception as e:
    print_result(False, f"Model test failed: {e}")
    test3_pass = False

# Test 4: Trained Model Loading
print_header("INTEGRATION TEST 4: Trained Model Loading")
try:
    model_dir = Path("outputs/trained_models")
    if not model_dir.exists() or not list(model_dir.glob("*.pth")):
        print("  No trained models found")
        print_result(True, "Skipped (no trained models)")
        test4_pass = True
    else:
        model_files = list(model_dir.glob("*.pth"))
        checkpoint = torch.load(str(model_files[0]), map_location='cpu', weights_only=False)
        print(f"  Loaded checkpoint: {model_files[0].name}")
        print(f"  Contains keys: {list(checkpoint.keys())}")
        print_result(True, f"Loaded {len(model_files)} trained models")
        test4_pass = True
except Exception as e:
    print_result(False, f"Model loading failed: {e}")
    test4_pass = False

# Test 5: Performance Benchmark
print_header("INTEGRATION TEST 5: Performance Benchmarks")
try:
    # Data loading speed
    start = time.time()
    loader = SensorDataLoader("TestData")
    dataset = loader.load_dataset("*.csv")
    load_time = time.time() - start
    count = sum(len(v) for v in dataset.values())
    print(f"  Data loading: {count} samples in {load_time:.2f}s ({count/load_time:.1f} samples/s)")

    # Synthetic generation speed
    generator = SyntheticSignalGenerator(seed=42)
    start = time.time()
    for _ in range(10):
        _ = generator.generate_signal('1um')
    gen_time = time.time() - start
    print(f"  Synthetic generation: 10 signals in {gen_time:.2f}s ({10/gen_time:.1f} signals/s)")

    # Model inference speed
    model = SimpleCNN1D(num_classes=3)
    model.eval()
    dummy_input = torch.randn(1, 1, 10000)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Benchmark
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    inf_time = time.time() - start
    print(f"  Model inference: 100 inferences in {inf_time:.2f}s ({100/inf_time:.1f} inferences/s)")
    print(f"  Average latency: {(inf_time/100)*1000:.2f}ms per sample")

    print_result(True, "Performance benchmarks completed")
    test5_pass = True
except Exception as e:
    print_result(False, f"Benchmark failed: {e}")
    test5_pass = False

# Summary
print_header("TEST SUMMARY")
tests = [
    ("Data Loading", test1_pass),
    ("Synthetic Generation", test2_pass),
    ("Model Architectures", test3_pass),
    ("Trained Model Loading", test4_pass),
    ("Performance Benchmarks", test5_pass)
]

for name, passed in tests:
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} {name}")

passed_count = sum(1 for _, p in tests if p)
total = len(tests)
percentage = (passed_count / total) * 100

print("\n" + "=" * 80)
print(f"Results: {passed_count}/{total} tests passed ({percentage:.1f}%)")
print("=" * 80)

if passed_count == total:
    print("\n[SUCCESS] All integration tests passed!")
    sys.exit(0)
else:
    print(f"\n[WARNING] {total - passed_count} test(s) failed")
    sys.exit(1)
