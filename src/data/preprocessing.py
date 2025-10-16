"""
Data Preprocessing Utilities
Signal processing, normalization, and length standardization for sensor data.
"""

import numpy as np
from typing import Tuple, Optional, Literal
from scipy import signal as scipy_signal


class SignalPreprocessor:
    """Preprocessing utilities for electrochemical sensor signals."""

    def __init__(self):
        """Initialize the preprocessor."""
        pass

    # ========== Normalization Methods ==========

    @staticmethod
    def normalize_zscore(data: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, dict]:
        """
        Z-score normalization (standardization).

        Args:
            data: Input signal array
            eps: Small constant to prevent division by zero

        Returns:
            Tuple of (normalized_data, normalization_params)
        """
        mean = np.mean(data)
        std = np.std(data)

        normalized = (data - mean) / (std + eps)

        params = {
            'method': 'zscore',
            'mean': float(mean),
            'std': float(std),
            'eps': eps
        }

        return normalized, params

    @staticmethod
    def normalize_minmax(data: np.ndarray,
                        feature_range: Tuple[float, float] = (0, 1),
                        eps: float = 1e-8) -> Tuple[np.ndarray, dict]:
        """
        Min-max normalization to specified range.

        Args:
            data: Input signal array
            feature_range: Target range (min, max)
            eps: Small constant to prevent division by zero

        Returns:
            Tuple of (normalized_data, normalization_params)
        """
        min_val = np.min(data)
        max_val = np.max(data)

        # Scale to [0, 1]
        scaled = (data - min_val) / (max_val - min_val + eps)

        # Scale to feature_range
        range_min, range_max = feature_range
        normalized = scaled * (range_max - range_min) + range_min

        params = {
            'method': 'minmax',
            'min': float(min_val),
            'max': float(max_val),
            'feature_range': feature_range,
            'eps': eps
        }

        return normalized, params

    @staticmethod
    def normalize_robust(data: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, dict]:
        """
        Robust normalization using median and IQR (resistant to outliers).

        Args:
            data: Input signal array
            eps: Small constant to prevent division by zero

        Returns:
            Tuple of (normalized_data, normalization_params)
        """
        median = np.median(data)
        q75 = np.percentile(data, 75)
        q25 = np.percentile(data, 25)
        iqr = q75 - q25

        normalized = (data - median) / (iqr + eps)

        params = {
            'method': 'robust',
            'median': float(median),
            'q25': float(q25),
            'q75': float(q75),
            'iqr': float(iqr),
            'eps': eps
        }

        return normalized, params

    @staticmethod
    def denormalize(data: np.ndarray, params: dict) -> np.ndarray:
        """
        Reverse normalization using stored parameters.

        Args:
            data: Normalized data
            params: Normalization parameters from normalize_* methods

        Returns:
            Original scale data
        """
        method = params['method']

        if method == 'zscore':
            return data * params['std'] + params['mean']

        elif method == 'minmax':
            range_min, range_max = params['feature_range']
            # Reverse feature_range scaling
            scaled = (data - range_min) / (range_max - range_min)
            # Reverse [0, 1] scaling
            return scaled * (params['max'] - params['min']) + params['min']

        elif method == 'robust':
            return data * params['iqr'] + params['median']

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    # ========== Length Normalization ==========

    @staticmethod
    def pad_sequence(data: np.ndarray,
                     target_length: int,
                     mode: Literal['constant', 'edge', 'reflect', 'wrap'] = 'constant',
                     constant_value: float = 0.0) -> np.ndarray:
        """
        Pad sequence to target length.

        Args:
            data: Input signal array
            target_length: Desired length
            mode: Padding mode ('constant', 'edge', 'reflect', 'wrap')
            constant_value: Value for constant padding

        Returns:
            Padded array
        """
        current_length = len(data)

        if current_length >= target_length:
            return data[:target_length]

        pad_width = target_length - current_length

        if mode == 'constant':
            return np.pad(data, (0, pad_width), mode='constant', constant_values=constant_value)
        else:
            return np.pad(data, (0, pad_width), mode=mode)

    @staticmethod
    def truncate_sequence(data: np.ndarray,
                         target_length: int,
                         strategy: Literal['center', 'random', 'start', 'end'] = 'center') -> np.ndarray:
        """
        Truncate sequence to target length.

        Args:
            data: Input signal array
            target_length: Desired length
            strategy: Truncation strategy
                - 'center': Take middle portion
                - 'random': Random crop
                - 'start': Take first portion
                - 'end': Take last portion

        Returns:
            Truncated array
        """
        current_length = len(data)

        if current_length <= target_length:
            return data

        if strategy == 'center':
            start_idx = (current_length - target_length) // 2
            return data[start_idx:start_idx + target_length]

        elif strategy == 'random':
            start_idx = np.random.randint(0, current_length - target_length + 1)
            return data[start_idx:start_idx + target_length]

        elif strategy == 'start':
            return data[:target_length]

        elif strategy == 'end':
            return data[-target_length:]

        else:
            raise ValueError(f"Unknown truncation strategy: {strategy}")

    @staticmethod
    def normalize_length(data: np.ndarray,
                        target_length: int,
                        pad_mode: str = 'constant',
                        truncate_strategy: str = 'center',
                        constant_value: float = 0.0) -> np.ndarray:
        """
        Normalize sequence to target length (pad or truncate as needed).

        Args:
            data: Input signal array
            target_length: Desired length
            pad_mode: Padding mode if sequence is too short
            truncate_strategy: Truncation strategy if sequence is too long
            constant_value: Value for constant padding

        Returns:
            Length-normalized array
        """
        current_length = len(data)

        if current_length < target_length:
            return SignalPreprocessor.pad_sequence(
                data, target_length, mode=pad_mode, constant_value=constant_value
            )
        elif current_length > target_length:
            return SignalPreprocessor.truncate_sequence(
                data, target_length, strategy=truncate_strategy
            )
        else:
            return data

    # ========== Filtering ==========

    @staticmethod
    def lowpass_filter(data: np.ndarray,
                      cutoff_freq: float,
                      sampling_rate: float,
                      order: int = 4) -> np.ndarray:
        """
        Apply Butterworth lowpass filter.

        Args:
            data: Input signal array
            cutoff_freq: Cutoff frequency (Hz)
            sampling_rate: Sampling rate (Hz)
            order: Filter order

        Returns:
            Filtered signal
        """
        nyquist = sampling_rate / 2
        normal_cutoff = cutoff_freq / nyquist

        b, a = scipy_signal.butter(order, normal_cutoff, btype='low', analog=False)
        filtered = scipy_signal.filtfilt(b, a, data)

        return filtered

    @staticmethod
    def highpass_filter(data: np.ndarray,
                       cutoff_freq: float,
                       sampling_rate: float,
                       order: int = 4) -> np.ndarray:
        """
        Apply Butterworth highpass filter.

        Args:
            data: Input signal array
            cutoff_freq: Cutoff frequency (Hz)
            sampling_rate: Sampling rate (Hz)
            order: Filter order

        Returns:
            Filtered signal
        """
        nyquist = sampling_rate / 2
        normal_cutoff = cutoff_freq / nyquist

        b, a = scipy_signal.butter(order, normal_cutoff, btype='high', analog=False)
        filtered = scipy_signal.filtfilt(b, a, data)

        return filtered

    @staticmethod
    def bandpass_filter(data: np.ndarray,
                       low_freq: float,
                       high_freq: float,
                       sampling_rate: float,
                       order: int = 4) -> np.ndarray:
        """
        Apply Butterworth bandpass filter.

        Args:
            data: Input signal array
            low_freq: Low cutoff frequency (Hz)
            high_freq: High cutoff frequency (Hz)
            sampling_rate: Sampling rate (Hz)
            order: Filter order

        Returns:
            Filtered signal
        """
        nyquist = sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist

        b, a = scipy_signal.butter(order, [low, high], btype='band', analog=False)
        filtered = scipy_signal.filtfilt(b, a, data)

        return filtered

    @staticmethod
    def moving_average_filter(data: np.ndarray, window_size: int) -> np.ndarray:
        """
        Apply moving average filter.

        Args:
            data: Input signal array
            window_size: Size of moving average window

        Returns:
            Smoothed signal
        """
        window = np.ones(window_size) / window_size
        filtered = np.convolve(data, window, mode='same')
        return filtered

    # ========== Outlier Removal ==========

    @staticmethod
    def remove_outliers_iqr(data: np.ndarray,
                           multiplier: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outliers using IQR method.

        Args:
            data: Input signal array
            multiplier: IQR multiplier (1.5 = moderate, 3.0 = extreme)

        Returns:
            Tuple of (cleaned_data, outlier_mask)
        """
        q75 = np.percentile(data, 75)
        q25 = np.percentile(data, 25)
        iqr = q75 - q25

        lower_bound = q25 - (multiplier * iqr)
        upper_bound = q75 + (multiplier * iqr)

        outlier_mask = (data < lower_bound) | (data > upper_bound)

        # Replace outliers with median
        cleaned = data.copy()
        median = np.median(data[~outlier_mask])
        cleaned[outlier_mask] = median

        return cleaned, outlier_mask

    @staticmethod
    def remove_outliers_zscore(data: np.ndarray,
                               threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outliers using z-score method.

        Args:
            data: Input signal array
            threshold: Z-score threshold (typically 2-3)

        Returns:
            Tuple of (cleaned_data, outlier_mask)
        """
        mean = np.mean(data)
        std = np.std(data)

        z_scores = np.abs((data - mean) / std)
        outlier_mask = z_scores > threshold

        # Replace outliers with mean
        cleaned = data.copy()
        cleaned[outlier_mask] = mean

        return cleaned, outlier_mask


def create_preprocessing_pipeline(normalize_method: str = 'zscore',
                                  target_length: Optional[int] = None,
                                  filter_type: Optional[str] = None,
                                  **kwargs) -> callable:
    """
    Create a preprocessing pipeline function.

    Args:
        normalize_method: Normalization method ('zscore', 'minmax', 'robust', or None)
        target_length: Target sequence length (None = no length normalization)
        filter_type: Filter type ('lowpass', 'highpass', 'bandpass', 'ma', or None)
        **kwargs: Additional arguments for specific preprocessing steps

    Returns:
        Pipeline function that takes (time, current) and returns processed data
    """
    preprocessor = SignalPreprocessor()

    def pipeline(time: np.ndarray, current: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Apply preprocessing pipeline.

        Returns:
            Tuple of (time, current, preprocessing_params)
        """
        params = {}

        # 1. Filtering (if specified)
        if filter_type is not None:
            sampling_rate = 1000.0 / (time[1] - time[0])  # Hz

            if filter_type == 'lowpass':
                current = preprocessor.lowpass_filter(
                    current,
                    kwargs.get('cutoff_freq', 100.0),
                    sampling_rate,
                    kwargs.get('order', 4)
                )
                params['filter'] = {'type': 'lowpass', 'cutoff_freq': kwargs.get('cutoff_freq', 100.0)}

            elif filter_type == 'highpass':
                current = preprocessor.highpass_filter(
                    current,
                    kwargs.get('cutoff_freq', 1.0),
                    sampling_rate,
                    kwargs.get('order', 4)
                )
                params['filter'] = {'type': 'highpass', 'cutoff_freq': kwargs.get('cutoff_freq', 1.0)}

            elif filter_type == 'bandpass':
                current = preprocessor.bandpass_filter(
                    current,
                    kwargs.get('low_freq', 1.0),
                    kwargs.get('high_freq', 100.0),
                    sampling_rate,
                    kwargs.get('order', 4)
                )
                params['filter'] = {
                    'type': 'bandpass',
                    'low_freq': kwargs.get('low_freq', 1.0),
                    'high_freq': kwargs.get('high_freq', 100.0)
                }

            elif filter_type == 'ma':
                current = preprocessor.moving_average_filter(
                    current,
                    kwargs.get('window_size', 10)
                )
                params['filter'] = {'type': 'moving_average', 'window_size': kwargs.get('window_size', 10)}

        # 2. Length normalization (if specified)
        if target_length is not None:
            time = preprocessor.normalize_length(
                time,
                target_length,
                pad_mode=kwargs.get('pad_mode', 'constant'),
                truncate_strategy=kwargs.get('truncate_strategy', 'center')
            )
            current = preprocessor.normalize_length(
                current,
                target_length,
                pad_mode=kwargs.get('pad_mode', 'constant'),
                truncate_strategy=kwargs.get('truncate_strategy', 'center')
            )
            params['length_norm'] = {
                'target_length': target_length,
                'pad_mode': kwargs.get('pad_mode', 'constant'),
                'truncate_strategy': kwargs.get('truncate_strategy', 'center')
            }

        # 3. Value normalization (if specified)
        if normalize_method is not None:
            if normalize_method == 'zscore':
                current, norm_params = preprocessor.normalize_zscore(current)
            elif normalize_method == 'minmax':
                current, norm_params = preprocessor.normalize_minmax(
                    current,
                    feature_range=kwargs.get('feature_range', (0, 1))
                )
            elif normalize_method == 'robust':
                current, norm_params = preprocessor.normalize_robust(current)
            else:
                raise ValueError(f"Unknown normalization method: {normalize_method}")

            params['normalization'] = norm_params

        return time, current, params

    return pipeline


if __name__ == "__main__":
    # Example usage
    preprocessor = SignalPreprocessor()

    # Test data
    data = np.random.randn(1000) + 5.0

    # Test normalization
    normalized, params = preprocessor.normalize_zscore(data)
    print(f"Z-score normalization: mean={np.mean(normalized):.4f}, std={np.std(normalized):.4f}")

    # Test denormalization
    denormalized = preprocessor.denormalize(normalized, params)
    print(f"Denormalization error: {np.mean(np.abs(data - denormalized)):.6f}")

    # Test length normalization
    short_data = np.random.randn(500)
    padded = preprocessor.normalize_length(short_data, 1000)
    print(f"Padded length: {len(padded)}")

    long_data = np.random.randn(1500)
    truncated = preprocessor.normalize_length(long_data, 1000)
    print(f"Truncated length: {len(truncated)}")
