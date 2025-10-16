"""
Data Augmentation for Time-Series Signals
Augmentation techniques specifically designed for electrochemical sensor signals.
"""

import numpy as np
from scipy import interpolate
from typing import Tuple, Optional


class TimeSeriesAugmenter:
    """Time-series data augmentation for sensor signals."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize augmenter.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

    # ========== Time Domain Augmentations ==========

    @staticmethod
    def time_warp(time: np.ndarray,
                  current: np.ndarray,
                  sigma: float = 0.2,
                  knot: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random time warping to signal.

        Args:
            time: Time array
            current: Current signal array
            sigma: Standard deviation of random warps
            knot: Number of control points for warping

        Returns:
            Tuple of (warped_time, warped_current)
        """
        from scipy.interpolate import CubicSpline

        n_samples = len(time)

        # Create random warping function
        warp = np.cumsum(np.random.randn(knot + 2, 1) * sigma, axis=0)
        warp = CubicSpline(
            np.linspace(0, n_samples - 1, num=knot + 2),
            warp,
            bc_type='clamped'
        )(np.arange(n_samples))
        warp = np.squeeze(warp)

        # Create warped indices
        time_warp_idx = np.linspace(0, n_samples - 1, num=n_samples) + warp
        time_warp_idx = np.clip(time_warp_idx, 0, n_samples - 1)

        # Interpolate
        interpolator = interpolate.interp1d(
            np.arange(n_samples),
            current,
            kind='cubic',
            fill_value='extrapolate'
        )

        warped_current = interpolator(time_warp_idx)

        return time, warped_current

    @staticmethod
    def time_shift(time: np.ndarray,
                   current: np.ndarray,
                   max_shift_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random time shift (circular shift).

        Args:
            time: Time array
            current: Current signal array
            max_shift_ratio: Maximum shift as ratio of signal length

        Returns:
            Tuple of (time, shifted_current)
        """
        n_samples = len(current)
        max_shift = int(n_samples * max_shift_ratio)

        shift_amount = np.random.randint(-max_shift, max_shift + 1)

        shifted_current = np.roll(current, shift_amount)

        return time, shifted_current

    # ========== Magnitude Augmentations ==========

    @staticmethod
    def magnitude_scale(time: np.ndarray,
                       current: np.ndarray,
                       sigma: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale magnitude by random factor.

        Args:
            time: Time array
            current: Current signal array
            sigma: Standard deviation of scaling factor

        Returns:
            Tuple of (time, scaled_current)
        """
        scale_factor = np.random.normal(1.0, sigma)
        scale_factor = max(0.5, min(scale_factor, 2.0))  # Clip to reasonable range

        scaled_current = current * scale_factor

        return time, scaled_current

    @staticmethod
    def magnitude_shift(time: np.ndarray,
                       current: np.ndarray,
                       sigma: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add random offset to magnitude.

        Args:
            time: Time array
            current: Current signal array
            sigma: Standard deviation of offset (relative to signal std)

        Returns:
            Tuple of (time, shifted_current)
        """
        signal_std = np.std(current)
        offset = np.random.normal(0, sigma * signal_std)

        shifted_current = current + offset

        return time, shifted_current

    # ========== Noise Augmentations ==========

    @staticmethod
    def add_gaussian_noise(time: np.ndarray,
                          current: np.ndarray,
                          snr_db: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add Gaussian noise with specified SNR.

        Args:
            time: Time array
            current: Current signal array
            snr_db: Signal-to-noise ratio in dB

        Returns:
            Tuple of (time, noisy_current)
        """
        # Calculate signal power
        signal_power = np.mean(current ** 2)

        # Calculate noise power from SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate and add noise
        noise = np.random.normal(0, np.sqrt(noise_power), len(current))
        noisy_current = current + noise

        return time, noisy_current

    @staticmethod
    def add_spike_noise(time: np.ndarray,
                       current: np.ndarray,
                       spike_prob: float = 0.01,
                       spike_magnitude: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add random spike noise.

        Args:
            time: Time array
            current: Current signal array
            spike_prob: Probability of spike at each point
            spike_magnitude: Magnitude of spikes (in units of signal std)

        Returns:
            Tuple of (time, noisy_current)
        """
        signal_std = np.std(current)

        # Generate random spikes
        spike_mask = np.random.rand(len(current)) < spike_prob
        spike_values = np.random.choice([-1, 1], len(current)) * spike_magnitude * signal_std

        noisy_current = current.copy()
        noisy_current[spike_mask] += spike_values[spike_mask]

        return time, noisy_current

    # ========== Window-based Augmentations ==========

    @staticmethod
    def window_warp(time: np.ndarray,
                   current: np.ndarray,
                   window_ratio: float = 0.1,
                   scale_range: Tuple[float, float] = (0.5, 2.0)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random scaling to a window of the signal.

        Args:
            time: Time array
            current: Current signal array
            window_ratio: Size of window as ratio of signal length
            scale_range: Range of scaling factors (min, max)

        Returns:
            Tuple of (time, warped_current)
        """
        n_samples = len(current)
        window_size = int(n_samples * window_ratio)

        # Random window position
        start_idx = np.random.randint(0, n_samples - window_size)
        end_idx = start_idx + window_size

        # Random scale factor
        scale_factor = np.random.uniform(*scale_range)

        # Apply scaling to window
        warped_current = current.copy()
        warped_current[start_idx:end_idx] *= scale_factor

        return time, warped_current

    @staticmethod
    def window_slice(time: np.ndarray,
                    current: np.ndarray,
                    slice_ratio: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract random slice of signal (cropping).

        Args:
            time: Time array
            current: Current signal array
            slice_ratio: Size of slice as ratio of original length

        Returns:
            Tuple of (sliced_time, sliced_current)
        """
        n_samples = len(current)
        slice_size = int(n_samples * slice_ratio)

        # Random start position
        max_start = n_samples - slice_size
        start_idx = np.random.randint(0, max_start + 1)
        end_idx = start_idx + slice_size

        sliced_time = time[start_idx:end_idx]
        sliced_current = current[start_idx:end_idx]

        return sliced_time, sliced_current

    # ========== Frequency Domain Augmentations ==========

    @staticmethod
    def frequency_mask(time: np.ndarray,
                      current: np.ndarray,
                      mask_ratio: float = 0.1,
                      mask_count: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mask random frequency bands (SpecAugment-style).

        Args:
            time: Time array
            current: Current signal array
            mask_ratio: Ratio of frequencies to mask
            mask_count: Number of frequency bands to mask

        Returns:
            Tuple of (time, masked_current)
        """
        # FFT
        fft_data = np.fft.rfft(current)
        n_freqs = len(fft_data)

        mask_size = int(n_freqs * mask_ratio)

        # Apply multiple masks
        for _ in range(mask_count):
            start_idx = np.random.randint(0, n_freqs - mask_size)
            fft_data[start_idx:start_idx + mask_size] = 0

        # Inverse FFT
        masked_current = np.fft.irfft(fft_data, n=len(current))

        return time, masked_current

    # ========== Combined Augmentation ==========

    def random_augment(self,
                      time: np.ndarray,
                      current: np.ndarray,
                      augmentation_list: Optional[list] = None,
                      prob: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random augmentation from a list.

        Args:
            time: Time array
            current: Current signal array
            augmentation_list: List of augmentation names to choose from
                             If None, uses all available augmentations
            prob: Probability of applying augmentation

        Returns:
            Tuple of (augmented_time, augmented_current)
        """
        if np.random.rand() > prob:
            return time, current

        if augmentation_list is None:
            augmentation_list = [
                'time_warp',
                'time_shift',
                'magnitude_scale',
                'magnitude_shift',
                'gaussian_noise',
                'spike_noise'
            ]

        # Choose random augmentation
        aug_name = np.random.choice(augmentation_list)

        # Apply augmentation
        if aug_name == 'time_warp':
            return self.time_warp(time, current)
        elif aug_name == 'time_shift':
            return self.time_shift(time, current)
        elif aug_name == 'magnitude_scale':
            return self.magnitude_scale(time, current)
        elif aug_name == 'magnitude_shift':
            return self.magnitude_shift(time, current)
        elif aug_name == 'gaussian_noise':
            return self.add_gaussian_noise(time, current)
        elif aug_name == 'spike_noise':
            return self.add_spike_noise(time, current)
        elif aug_name == 'window_warp':
            return self.window_warp(time, current)
        elif aug_name == 'frequency_mask':
            return self.frequency_mask(time, current)
        else:
            return time, current

    def compose_augmentations(self,
                            time: np.ndarray,
                            current: np.ndarray,
                            augmentations: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply multiple augmentations in sequence.

        Args:
            time: Time array
            current: Current signal array
            augmentations: List of (augmentation_name, params_dict) tuples

        Returns:
            Tuple of (augmented_time, augmented_current)
        """
        aug_time = time.copy()
        aug_current = current.copy()

        for aug_name, params in augmentations:
            if aug_name == 'time_warp':
                aug_time, aug_current = self.time_warp(aug_time, aug_current, **params)
            elif aug_name == 'time_shift':
                aug_time, aug_current = self.time_shift(aug_time, aug_current, **params)
            elif aug_name == 'magnitude_scale':
                aug_time, aug_current = self.magnitude_scale(aug_time, aug_current, **params)
            elif aug_name == 'magnitude_shift':
                aug_time, aug_current = self.magnitude_shift(aug_time, aug_current, **params)
            elif aug_name == 'gaussian_noise':
                aug_time, aug_current = self.add_gaussian_noise(aug_time, aug_current, **params)
            elif aug_name == 'spike_noise':
                aug_time, aug_current = self.add_spike_noise(aug_time, aug_current, **params)
            elif aug_name == 'window_warp':
                aug_time, aug_current = self.window_warp(aug_time, aug_current, **params)
            elif aug_name == 'frequency_mask':
                aug_time, aug_current = self.frequency_mask(aug_time, aug_current, **params)

        return aug_time, aug_current


if __name__ == "__main__":
    # Example usage
    print("Testing time-series augmentation...")

    # Create synthetic signal
    t = np.linspace(0, 10, 1000)
    signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 3 * t)

    augmenter = TimeSeriesAugmenter(seed=42)

    # Test different augmentations
    print("\n1. Time Warp:")
    _, aug1 = augmenter.time_warp(t, signal)
    print(f"   Original shape: {signal.shape}, Augmented shape: {aug1.shape}")

    print("\n2. Magnitude Scale:")
    _, aug2 = augmenter.magnitude_scale(t, signal, sigma=0.2)
    print(f"   Original mean: {np.mean(signal):.4f}, Augmented mean: {np.mean(aug2):.4f}")

    print("\n3. Gaussian Noise (SNR=20dB):")
    _, aug3 = augmenter.add_gaussian_noise(t, signal, snr_db=20)
    print(f"   Original std: {np.std(signal):.4f}, Noisy std: {np.std(aug3):.4f}")

    print("\n4. Random Augmentation:")
    _, aug4 = augmenter.random_augment(t, signal, prob=1.0)
    print(f"   Applied random augmentation")

    print("\n5. Composed Augmentations:")
    augmentations = [
        ('magnitude_scale', {'sigma': 0.1}),
        ('gaussian_noise', {'snr_db': 25}),
        ('time_shift', {'max_shift_ratio': 0.05})
    ]
    _, aug5 = augmenter.compose_augmentations(t, signal, augmentations)
    print(f"   Applied 3 augmentations in sequence")

    print("\nAugmentation testing complete!")
