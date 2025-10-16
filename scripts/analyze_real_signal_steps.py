"""
Analyze real signals to extract step characteristics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

def detect_steps(current, window=1000, threshold_percentile=90):
    """
    Detect steps in current signal.

    Args:
        current: Current array
        window: Smoothing window size
        threshold_percentile: Percentile for step detection

    Returns:
        step_positions, step_amplitudes, step_durations
    """
    # Smooth signal
    smoothed = np.convolve(current, np.ones(window)/window, mode='same')

    # Calculate derivative (rate of change)
    derivative = np.diff(smoothed)

    # Find negative peaks (downward steps)
    threshold = np.percentile(np.abs(derivative), threshold_percentile)
    neg_derivative = -derivative
    peaks, properties = find_peaks(neg_derivative, height=threshold, distance=window)

    step_positions = peaks
    step_amplitudes = -properties['peak_heights']  # Negative for downward

    # Estimate step duration (transition time)
    durations = []
    for pos in peaks:
        # Look at region around step
        start = max(0, pos - window//2)
        end = min(len(current), pos + window//2)

        # Find where signal stabilizes (derivative close to zero)
        local_deriv = np.abs(derivative[start:end])
        stable_threshold = np.mean(local_deriv) * 0.5

        # Count samples during transition
        transition_samples = np.sum(local_deriv > stable_threshold)
        durations.append(transition_samples)

    return step_positions, step_amplitudes, np.array(durations)


# Analyze all files
data_dir = Path('TestData')
classes = ['1um', '2um', '3um']

print("REAL SIGNAL STEP ANALYSIS")
print("=" * 80)

for class_name in classes:
    files = list(data_dir.glob(f'PS {class_name}*.csv'))

    print(f"\n{class_name.upper()} Class:")
    print("-" * 80)

    all_num_steps = []
    all_amplitudes = []
    all_durations = []

    # Analyze first 3 files per class
    for file_path in files[:3]:
        df = pd.read_csv(file_path)
        time = df.iloc[:, 0].values
        current = df.iloc[:, 1].values

        # Detect steps
        positions, amplitudes, durations = detect_steps(current, window=1000)

        num_steps = len(positions)
        all_num_steps.append(num_steps)

        if num_steps > 0:
            all_amplitudes.extend(amplitudes)
            all_durations.extend(durations)

            # Convert durations to ms
            sampling_rate = 1220.7  # Hz
            durations_ms = durations * (1000 / sampling_rate)

            print(f"\n  {file_path.name}:")
            print(f"    Length: {len(current):,} points ({time[-1]/1000:.1f} s)")
            print(f"    Mean current: {current.mean():.4f}")
            print(f"    Std current: {current.std():.4f}")
            print(f"    Steps detected: {num_steps}")

            if num_steps > 0:
                print(f"    Step amplitudes: {amplitudes.mean():.5f} ± {amplitudes.std():.5f}")
                print(f"    Step % of mean: {(np.abs(amplitudes.mean())/current.mean())*100:.2f}%")
                print(f"    Step durations: {durations_ms.mean():.1f} ± {durations_ms.std():.1f} ms")
                print(f"    Step positions (time): {[time[p]/1000 for p in positions[:5]]}")

    # Summary statistics
    if all_num_steps:
        print(f"\n  SUMMARY ({len(files[:3])} files):")
        print(f"    Avg steps per signal: {np.mean(all_num_steps):.1f} ± {np.std(all_num_steps):.1f}")

        if all_amplitudes:
            all_amplitudes = np.array(all_amplitudes)
            all_durations = np.array(all_durations)
            sampling_rate = 1220.7

            print(f"    Step amplitude range: [{np.min(all_amplitudes):.5f}, {np.max(all_amplitudes):.5f}]")
            print(f"    Avg step amplitude: {np.mean(all_amplitudes):.5f}")
            print(f"    Avg step duration: {np.mean(all_durations) * (1000/sampling_rate):.1f} ms")
            print(f"    Duration range: [{np.min(all_durations) * (1000/sampling_rate):.1f}, {np.max(all_durations) * (1000/sampling_rate):.1f}] ms")

print("\n" + "=" * 80)
print("Analysis complete!")
