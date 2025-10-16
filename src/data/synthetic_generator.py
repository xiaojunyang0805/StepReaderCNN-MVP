"""
Synthetic Signal Generator for Electrochemical Sensor Data
Generate realistic synthetic signals based on real data characteristics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d


class SyntheticSignalGenerator:
    """
    Generate synthetic electrochemical sensor signals.

    Based on analysis of real TestData:
    - 1um: Current ~1.11 ± 0.018, length ~115k points
    - 2um: Current ~0.96 ± 0.017, length ~175k points
    - 3um: Current ~1.83 ± 0.136, length ~71k points
    - Sampling rate: ~1220 Hz
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize generator.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        # Class characteristics from real data analysis
        self.class_params = {
            '1um': {
                'mean_current': 1.11,
                'std_current': 0.018,
                'typical_length': 115000,
                'length_variance': 0.15  # ±15%
            },
            '2um': {
                'mean_current': 0.96,
                'std_current': 0.017,
                'typical_length': 175000,
                'length_variance': 0.15
            },
            '3um': {
                'mean_current': 1.83,
                'std_current': 0.136,
                'typical_length': 71000,
                'length_variance': 0.25  # Higher variance
            }
        }

        # Sampling parameters
        self.sampling_rate = 1220.7  # Hz
        self.time_step = 1000.0 / self.sampling_rate  # ms

    def generate_signal(self,
                       class_name: str,
                       length: Optional[int] = None,
                       noise_level: float = 1.0,
                       num_steps: Optional[int] = None,
                       add_drift: bool = False,
                       add_spikes: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a synthetic electrochemical collision signal with stochastic steps.

        This generates signals characteristic of single-entity electrochemistry:
        - Random step positions (collision events)
        - Negative step heights (current decrease on collision)
        - Minimum distance between steps (physical constraint)
        - Variable rise times (instantaneous to gradual)

        Args:
            class_name: Class label ('1um', '2um', '3um')
            length: Signal length in points (None for typical length with variance)
            noise_level: Noise multiplier (1.0 = typical noise)
            num_steps: Number of collision steps (None for random 3-10)
            add_drift: Whether to add slow baseline drift
            add_spikes: Whether to add random spike artifacts

        Returns:
            Tuple of (time_data, current_data)
        """
        if class_name not in self.class_params:
            raise ValueError(f"Unknown class: {class_name}")

        params = self.class_params[class_name]

        # Determine signal length
        if length is None:
            typical_length = params['typical_length']
            variance = params['length_variance']
            length = int(typical_length * (1 + np.random.uniform(-variance, variance)))

        # Generate time axis
        time_data = np.arange(length) * self.time_step

        # Base signal (constant baseline)
        mean_current = params['mean_current']
        std_current = params['std_current']

        current_data = np.ones(length) * mean_current

        # Generate stochastic collision steps (KEY FEATURE!)
        # Based on real data: 3-6 major steps per signal
        if num_steps is None:
            num_steps = np.random.randint(3, 7)  # 3-6 collision events

        # Step parameters for single-entity electrochemistry
        min_step_distance = int(length * 0.1)  # Minimum 10% of signal between collisions

        # Generate random step positions
        step_positions = self._generate_step_positions(length, num_steps, min_step_distance)

        # Generate step heights (negative for collision/blocking current)
        # Based on real data analysis:
        # - 1um: small steps (~3-5% per step)
        # - 2um: medium steps (~5-8% per step)
        # - 3um: large steps (~8-12% per step)
        if '1um' in class_name:
            step_height_range = (mean_current * -0.05, mean_current * -0.03)  # Small steps
        elif '2um' in class_name:
            step_height_range = (mean_current * -0.08, mean_current * -0.05)  # Medium steps
        else:  # 3um
            step_height_range = (mean_current * -0.12, mean_current * -0.08)  # Large steps

        # Apply steps to signal
        for pos in step_positions:
            step_height = np.random.uniform(*step_height_range)

            # Rise time based on real data: fast but not instantaneous
            # At 1220 Hz, 100-400 samples = 0.08-0.33 seconds (matches real data ~0.1-0.4s transitions)
            rise_time = np.random.randint(100, 400)

            # Create step
            step = self._create_step(length, pos, step_height, rise_time)
            current_data += step

        # Add realistic noise (real data has very low noise on plateaus)
        # CRITICAL: std_current for 3um includes step variation, NOT noise level
        # Real plateau noise is very small (~0.01-0.02) for all classes
        # Use actual noise level based on visual inspection of plateaus
        actual_noise_std = 0.015  # Noise on plateaus is similar for all classes

        # Real signals show very smooth plateaus with minimal noise
        # 1. Gaussian white noise (instrument noise) - very low level
        white_noise = np.random.normal(0, actual_noise_std * noise_level * 0.3, length)
        current_data += white_noise

        # 2. Pink noise (1/f noise, common in electrochemical systems) - minimal
        pink_noise = self._generate_pink_noise(length)
        current_data += pink_noise * actual_noise_std * 0.05 * noise_level

        # 3. Filtered white noise (capacitive/RC effects) - very minimal
        hf_noise = self._add_filtered_noise(length, cutoff_freq=100)
        current_data += hf_noise * actual_noise_std * 0.01 * noise_level

        # Optional: Add slow baseline drift
        if add_drift:
            drift_amplitude = std_current * 0.5 * noise_level
            drift_frequency = np.random.uniform(0.3, 1.0)  # Very slow
            drift = drift_amplitude * np.sin(2 * np.pi * drift_frequency * np.linspace(0, 1, length))
            current_data += drift

        # Optional: Add random spike artifacts
        if add_spikes:
            num_spikes = np.random.randint(0, 3)  # 0-2 spikes
            if num_spikes > 0:
                spike_positions = np.random.choice(length, num_spikes, replace=False)

                for spike_pos in spike_positions:
                    spike_amplitude = np.random.uniform(2, 4) * std_current * noise_level
                    spike_sign = np.random.choice([-1, 1])
                    spike_width = np.random.randint(5, 20)

                    # Gaussian spike
                    x = np.arange(length) - spike_pos
                    spike = spike_sign * spike_amplitude * np.exp(-x**2 / (2 * spike_width**2))
                    current_data += spike

        return time_data, current_data

    def _generate_step_positions(self, length: int, num_steps: int, min_distance: int) -> List[int]:
        """
        Generate random step positions with minimum distance constraint.

        Args:
            length: Signal length
            num_steps: Number of steps
            min_distance: Minimum samples between steps

        Returns:
            Sorted list of step positions
        """
        positions = []
        max_attempts = 1000

        for _ in range(num_steps):
            for attempt in range(max_attempts):
                # Random position, avoiding edges
                pos = np.random.randint(int(length * 0.1), int(length * 0.9))

                # Check distance from existing positions
                if len(positions) == 0:
                    positions.append(pos)
                    break

                if all(abs(pos - p) >= min_distance for p in positions):
                    positions.append(pos)
                    break

        return sorted(positions)

    def _create_step(self, length: int, position: int, height: float, rise_time: int) -> np.ndarray:
        """
        Create a step function with variable rise time.

        Args:
            length: Signal length
            position: Step position
            height: Step height (negative for collision)
            rise_time: Rise time in samples (1 = instantaneous)

        Returns:
            Step array
        """
        step = np.zeros(length)

        if rise_time == 1:
            # Instantaneous step
            step[position:] = height
        else:
            # Gradual step with sigmoid (gentler slope)
            for i in range(length):
                if i < position:
                    step[i] = 0
                elif i < position + rise_time:
                    # Sigmoid transition with gentler slope (steepness=5 instead of 10)
                    x = (i - position) / rise_time
                    step[i] = height / (1 + np.exp(-5 * (x - 0.5)))
                else:
                    step[i] = height

        return step

    def _generate_pink_noise(self, length: int) -> np.ndarray:
        """
        Generate pink noise (1/f noise).

        Args:
            length: Signal length

        Returns:
            Pink noise array
        """
        # Generate white noise in frequency domain
        white = np.fft.rfft(np.random.randn(length))

        # Create 1/f filter
        freqs = np.fft.rfftfreq(length)
        freqs[0] = 1  # Avoid division by zero
        pink_filter = 1 / np.sqrt(freqs)

        # Apply filter and convert back to time domain
        pink = np.fft.irfft(white * pink_filter, n=length)

        # Normalize
        pink = pink / np.std(pink)

        return pink

    def _add_filtered_noise(self, length: int, cutoff_freq: float) -> np.ndarray:
        """
        Generate filtered white noise.

        Args:
            length: Signal length
            cutoff_freq: Cutoff frequency in Hz

        Returns:
            Filtered noise array
        """
        # Generate white noise
        white_noise = np.random.randn(length)

        # Design lowpass filter
        nyquist = self.sampling_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        b, a = scipy_signal.butter(4, normalized_cutoff, btype='low')

        # Apply filter
        filtered = scipy_signal.filtfilt(b, a, white_noise)

        # Normalize
        filtered = filtered / np.std(filtered)

        return filtered

    def generate_batch(self,
                      class_counts: Dict[str, int],
                      noise_level_range: Tuple[float, float] = (0.8, 1.2),
                      **kwargs) -> Dict[str, List[Tuple[np.ndarray, np.ndarray, str]]]:
        """
        Generate a batch of synthetic signals.

        Args:
            class_counts: Dictionary mapping class names to counts
            noise_level_range: Range of noise levels (min, max)
            **kwargs: Additional arguments for generate_signal()

        Returns:
            Dictionary mapping class names to list of (time, current, filename) tuples
        """
        synthetic_data = {}

        for class_name, count in class_counts.items():
            samples = []

            for i in range(count):
                # Random noise level
                noise_level = np.random.uniform(*noise_level_range)

                # Generate signal
                time_data, current_data = self.generate_signal(
                    class_name,
                    noise_level=noise_level,
                    **kwargs
                )

                # Create filename
                filename = f"synthetic_{class_name}_{i+1:03d}.csv"

                samples.append((time_data, current_data, filename))

            synthetic_data[class_name] = samples

        return synthetic_data

    def save_signal(self,
                   time_data: np.ndarray,
                   current_data: np.ndarray,
                   filename: str,
                   save_dir: str = 'data/synthetic') -> str:
        """
        Save synthetic signal to CSV file.

        Args:
            time_data: Time values
            current_data: Current values
            filename: Output filename
            save_dir: Output directory

        Returns:
            Full path to saved file
        """
        import pandas as pd
        from pathlib import Path

        # Create directory if needed
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Create DataFrame
        df = pd.DataFrame({
            'Time (ms)': time_data,
            'Current (Channel 1)': current_data
        })

        # Save to CSV
        output_path = save_path / filename
        df.to_csv(output_path, index=False)

        return str(output_path)

    def save_batch(self,
                  synthetic_data: Dict[str, List[Tuple[np.ndarray, np.ndarray, str]]],
                  save_dir: str = 'data/synthetic') -> int:
        """
        Save batch of synthetic signals to CSV files.

        Args:
            synthetic_data: Dictionary from generate_batch()
            save_dir: Output directory

        Returns:
            Total number of files saved
        """
        count = 0

        for class_name, samples in synthetic_data.items():
            for time_data, current_data, filename in samples:
                self.save_signal(time_data, current_data, filename, save_dir)
                count += 1

        return count

    def get_class_statistics(self, class_name: str) -> Dict:
        """
        Get statistics for a class.

        Args:
            class_name: Class label

        Returns:
            Dictionary with class parameters
        """
        if class_name not in self.class_params:
            raise ValueError(f"Unknown class: {class_name}")

        return self.class_params[class_name].copy()

    def visualize_signal(self,
                        time_data: np.ndarray,
                        current_data: np.ndarray,
                        class_name: str = None,
                        save_path: str = None):
        """
        Visualize a generated signal.

        Args:
            time_data: Time values
            current_data: Current values
            class_name: Optional class label for title
            save_path: Optional path to save figure
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Time domain
        axes[0].plot(time_data / 1000, current_data, linewidth=0.5)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Current')
        axes[0].set_title(f'Synthetic Signal{" - " + class_name if class_name else ""}')
        axes[0].grid(True, alpha=0.3)

        # Frequency domain
        fft = np.fft.rfft(current_data)
        freqs = np.fft.rfftfreq(len(current_data), d=self.time_step/1000)

        axes[1].loglog(freqs[1:], np.abs(fft[1:]))
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Magnitude')
        axes[1].set_title('Power Spectrum')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig


def create_balanced_synthetic_dataset(real_data: Dict[str, List],
                                      target_samples_per_class: int = 50,
                                      seed: Optional[int] = 42) -> Dict[str, List[Tuple]]:
    """
    Create a balanced synthetic dataset to augment real data.

    Args:
        real_data: Real dataset dictionary
        target_samples_per_class: Target number of samples per class
        seed: Random seed

    Returns:
        Dictionary with synthetic samples
    """
    generator = SyntheticSignalGenerator(seed=seed)

    # Calculate how many synthetic samples needed per class
    class_counts = {}
    for class_name, samples in real_data.items():
        real_count = len(samples)
        synthetic_needed = max(0, target_samples_per_class - real_count)
        class_counts[class_name] = synthetic_needed

    print(f"Generating synthetic samples:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} synthetic samples")

    # Generate synthetic data
    synthetic_data = generator.generate_batch(class_counts)

    return synthetic_data


if __name__ == "__main__":
    # Example usage
    print("Testing Synthetic Signal Generator...")
    print("=" * 80)

    generator = SyntheticSignalGenerator(seed=42)

    # Generate one signal per class
    for class_name in ['1um', '2um', '3um']:
        print(f"\nGenerating {class_name} signal...")
        time_data, current_data = generator.generate_signal(class_name)

        print(f"  Length: {len(current_data):,} points")
        print(f"  Duration: {time_data[-1]/1000:.1f} seconds")
        print(f"  Current range: {current_data.min():.4f} - {current_data.max():.4f}")
        print(f"  Current mean: {current_data.mean():.4f}")
        print(f"  Current std: {current_data.std():.4f}")

    # Generate batch
    print("\n" + "=" * 80)
    print("Generating batch (5 samples per class)...")

    class_counts = {'1um': 5, '2um': 5, '3um': 5}
    synthetic_data = generator.generate_batch(class_counts)

    print(f"\nGenerated {sum(len(v) for v in synthetic_data.values())} total samples")
    for class_name, samples in synthetic_data.items():
        print(f"  {class_name}: {len(samples)} samples")

    # Save samples
    print("\nSaving samples to data/synthetic/...")
    count = generator.save_batch(synthetic_data, 'data/synthetic')
    print(f"Saved {count} files")

    print("\n" + "=" * 80)
    print("Synthetic signal generator test complete!")
