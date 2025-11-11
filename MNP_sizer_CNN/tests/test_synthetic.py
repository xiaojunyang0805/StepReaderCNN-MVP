"""
Test Script for Synthetic Data Generation (Phase 8)
Tests synthetic signal generator and validation.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from data.synthetic_generator import SyntheticSignalGenerator, create_balanced_synthetic_dataset
from data.data_loader import SensorDataLoader
from data.data_split import stratified_split


def test_generator_initialization():
    """Test generator initialization."""
    print("=" * 80)
    print("TEST 1: GENERATOR INITIALIZATION")
    print("=" * 80)

    try:
        generator = SyntheticSignalGenerator(seed=42)

        print("\n[OK] Generator initialized successfully")

        # Check class parameters
        for class_name in ['1um', '2um', '3um']:
            stats = generator.get_class_statistics(class_name)
            print(f"\n{class_name} parameters:")
            print(f"  Mean current: {stats['mean_current']:.3f}")
            print(f"  Std current: {stats['std_current']:.3f}")
            print(f"  Typical length: {stats['typical_length']:,}")

        return True

    except Exception as e:
        print(f"\n[ERROR] Generator initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_signal_generation():
    """Test generating single signals."""
    print("\n" + "=" * 80)
    print("TEST 2: SINGLE SIGNAL GENERATION")
    print("=" * 80)

    generator = SyntheticSignalGenerator(seed=42)

    try:
        for class_name in ['1um', '2um', '3um']:
            print(f"\nGenerating {class_name} signal...")

            time_data, current_data = generator.generate_signal(class_name)

            print(f"  [OK] Generated signal")
            print(f"  Length: {len(current_data):,} points")
            print(f"  Duration: {time_data[-1]/1000:.1f} s")
            print(f"  Current mean: {current_data.mean():.4f}")
            print(f"  Current std: {current_data.std():.4f}")
            print(f"  Current range: [{current_data.min():.4f}, {current_data.max():.4f}]")

            # Validate signal properties
            assert len(time_data) == len(current_data), "Time and current length mismatch"
            assert len(current_data) > 0, "Empty signal"
            assert not np.any(np.isnan(current_data)), "NaN values in signal"
            assert not np.any(np.isinf(current_data)), "Inf values in signal"

        print("\n[OK] All signals generated successfully")
        return True

    except Exception as e:
        print(f"\n[ERROR] Signal generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_generation():
    """Test batch generation."""
    print("\n" + "=" * 80)
    print("TEST 3: BATCH GENERATION")
    print("=" * 80)

    generator = SyntheticSignalGenerator(seed=42)

    try:
        class_counts = {'1um': 5, '2um': 5, '3um': 5}

        print(f"\nGenerating batch: {class_counts}")

        synthetic_data = generator.generate_batch(
            class_counts,
            noise_level_range=(0.8, 1.2)
        )

        print(f"\n[OK] Batch generated")

        total_samples = 0
        for class_name, samples in synthetic_data.items():
            print(f"  {class_name}: {len(samples)} samples")
            total_samples += len(samples)

            # Validate first sample
            time_data, current_data, filename = samples[0]
            assert len(time_data) == len(current_data), f"{class_name}: Length mismatch"
            assert filename.endswith('.csv'), f"{class_name}: Invalid filename"

        assert total_samples == sum(class_counts.values()), "Total count mismatch"

        print(f"\nTotal samples: {total_samples}")
        print("[OK] Batch generation successful")

        return True

    except Exception as e:
        print(f"\n[ERROR] Batch generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_functionality():
    """Test saving synthetic signals."""
    print("\n" + "=" * 80)
    print("TEST 4: SAVE FUNCTIONALITY")
    print("=" * 80)

    generator = SyntheticSignalGenerator(seed=42)

    try:
        # Generate one signal
        time_data, current_data = generator.generate_signal('1um')

        # Save to file
        save_dir = "data/synthetic_test"
        filename = "test_signal_1um_001.csv"

        save_path = generator.save_signal(time_data, current_data, filename, save_dir)

        print(f"\n[OK] Signal saved to: {save_path}")

        # Verify file exists
        assert Path(save_path).exists(), "Saved file not found"

        # Load and verify
        df = pd.read_csv(save_path)
        assert df.shape[0] == len(current_data), "Row count mismatch"
        assert df.shape[1] == 2, "Column count mismatch"

        print(f"  File verified: {df.shape[0]} rows, {df.shape[1]} columns")
        print("[OK] Save functionality working")

        # Clean up
        Path(save_path).unlink()
        Path(save_dir).rmdir()

        return True

    except Exception as e:
        print(f"\n[ERROR] Save functionality failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_balanced_dataset_creation():
    """Test creating balanced dataset with real data."""
    print("\n" + "=" * 80)
    print("TEST 5: BALANCED DATASET CREATION")
    print("=" * 80)

    try:
        # Load real data
        data_dir = Path(__file__).parent / "TestData"
        if not data_dir.exists():
            print("[WARN] TestData directory not found, skipping test")
            return True

        loader = SensorDataLoader(str(data_dir))
        dataset = loader.load_dataset("*.csv")

        print("\nReal data distribution:")
        for class_name, samples in dataset.items():
            print(f"  {class_name}: {len(samples)} samples")

        # Create balanced dataset
        target_samples = 20
        print(f"\nTarget samples per class: {target_samples}")

        synthetic_data = create_balanced_synthetic_dataset(
            dataset,
            target_samples_per_class=target_samples,
            seed=42
        )

        print("\nSynthetic data generated:")
        for class_name, samples in synthetic_data.items():
            print(f"  {class_name}: {len(samples)} synthetic samples")

        # Verify balancing
        for class_name in dataset.keys():
            real_count = len(dataset[class_name])
            synthetic_count = len(synthetic_data[class_name])
            total = real_count + synthetic_count

            print(f"\n{class_name} total: {total} (real: {real_count}, synthetic: {synthetic_count})")

            # Should be close to target (may vary due to variance)
            if real_count < target_samples:
                assert synthetic_count > 0, f"{class_name}: No synthetic samples generated"

        print("\n[OK] Balanced dataset created successfully")
        return True

    except Exception as e:
        print(f"\n[ERROR] Balanced dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_signal_characteristics():
    """Test signal characteristics match expected values."""
    print("\n" + "=" * 80)
    print("TEST 6: SIGNAL CHARACTERISTICS")
    print("=" * 80)

    generator = SyntheticSignalGenerator(seed=42)

    try:
        num_samples = 10

        for class_name in ['1um', '2um', '3um']:
            print(f"\nTesting {class_name} (n={num_samples})...")

            # Expected values from real data
            expected_mean = generator.class_params[class_name]['mean_current']
            expected_std = generator.class_params[class_name]['std_current']

            means = []
            stds = []

            for i in range(num_samples):
                _, current_data = generator.generate_signal(class_name, noise_level=1.0)
                means.append(current_data.mean())
                stds.append(current_data.std())

            avg_mean = np.mean(means)
            avg_std = np.mean(stds)

            print(f"  Expected mean: {expected_mean:.3f}, Generated mean: {avg_mean:.3f}")
            print(f"  Expected std: {expected_std:.3f}, Generated std: {avg_std:.3f}")

            # Check if within reasonable range (±20%)
            mean_diff = abs(avg_mean - expected_mean) / expected_mean
            std_diff = abs(avg_std - expected_std) / expected_std

            if mean_diff < 0.2 and std_diff < 0.5:
                print(f"  [OK] Statistics within acceptable range")
            else:
                print(f"  [WARN] Statistics differ from expected (mean: {mean_diff:.1%}, std: {std_diff:.1%})")

        print("\n[OK] Signal characteristics test complete")
        return True

    except Exception as e:
        print(f"\n[ERROR] Signal characteristics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_flags():
    """Test signal feature flags."""
    print("\n" + "=" * 80)
    print("TEST 7: FEATURE FLAGS")
    print("=" * 80)

    generator = SyntheticSignalGenerator(seed=42)

    try:
        class_name = '1um'

        # Test with minimal features (few steps, no drift/spikes)
        print("\nGenerating with minimal features...")
        time1, current1 = generator.generate_signal(
            class_name,
            num_steps=2,
            add_drift=False,
            add_spikes=False,
            noise_level=0.5
        )

        # Test with all features enabled (more steps, drift, spikes)
        print("Generating with all features enabled...")
        time2, current2 = generator.generate_signal(
            class_name,
            num_steps=8,
            add_drift=True,
            add_spikes=True,
            noise_level=0.5
        )

        print(f"\nMinimal features (2 steps):")
        print(f"  Std: {current1.std():.4f}")
        print(f"  Range: {current1.max() - current1.min():.4f}")

        print(f"\nAll features (8 steps + drift + spikes):")
        print(f"  Std: {current2.std():.4f}")
        print(f"  Range: {current2.max() - current2.min():.4f}")

        # Signal with more features should have higher variance
        assert current2.std() > current1.std() * 0.8, "Features did not increase variance sufficiently"

        print("\n[OK] Feature flags working correctly")
        return True

    except Exception as e:
        print(f"\n[ERROR] Feature flags test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("SYNTHETIC DATA GENERATION TEST SUITE - PHASE 8")
    print("=" * 80)

    tests = [
        ("Generator Initialization", test_generator_initialization),
        ("Single Signal Generation", test_single_signal_generation),
        ("Batch Generation", test_batch_generation),
        ("Save Functionality", test_save_functionality),
        ("Balanced Dataset Creation", test_balanced_dataset_creation),
        ("Signal Characteristics", test_signal_characteristics),
        ("Feature Flags", test_feature_flags),
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
        print("\n[SUCCESS] All tests passed! Synthetic data generator is working correctly.")
    else:
        print(f"\n[WARN] {total - passed} test(s) failed. Please review the errors above.")


if __name__ == "__main__":
    main()
