"""
CNN Model Architectures for Time-Series Classification
1D CNN models for electrochemical sensor signal classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class SimpleCNN1D(nn.Module):
    """
    Simple baseline 1D CNN for time-series classification.

    Architecture:
    - 4 convolutional blocks (conv → bn → relu → pool)
    - Global average pooling
    - Fully connected classifier
    """

    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 3,
                 base_filters: int = 32,
                 dropout: float = 0.5):
        """
        Initialize SimpleCNN1D.

        Args:
            in_channels: Number of input channels (1 for single sensor)
            num_classes: Number of output classes
            base_filters: Number of filters in first conv layer (doubled each block)
            dropout: Dropout probability
        """
        super(SimpleCNN1D, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Convolutional blocks
        self.conv1 = self._make_conv_block(in_channels, base_filters, kernel_size=7)
        self.conv2 = self._make_conv_block(base_filters, base_filters * 2, kernel_size=5)
        self.conv3 = self._make_conv_block(base_filters * 2, base_filters * 4, kernel_size=3)
        self.conv4 = self._make_conv_block(base_filters * 4, base_filters * 8, kernel_size=3)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_filters * 8, num_classes)

    def _make_conv_block(self, in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
        """Create a convolutional block."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, in_channels, sequence_length)

        Returns:
            Output logits (batch_size, num_classes)
        """
        # Convolutional blocks
        x = self.conv1(x)  # (B, 32, L/2)
        x = self.conv2(x)  # (B, 64, L/4)
        x = self.conv3(x)  # (B, 128, L/8)
        x = self.conv4(x)  # (B, 256, L/16)

        # Global pooling
        x = self.global_pool(x)  # (B, 256, 1)
        x = x.squeeze(-1)  # (B, 256)

        # Classifier
        x = self.dropout(x)
        x = self.fc(x)  # (B, num_classes)

        return x


class ResidualBlock1D(nn.Module):
    """Residual block for 1D CNN."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: bool = False):
        """
        Initialize residual block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for convolution
            downsample: Whether to downsample the skip connection
        """
        super(ResidualBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection
        self.downsample = None
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    """
    ResNet-inspired 1D CNN with skip connections.

    More robust to vanishing gradients and better feature learning.
    """

    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 3,
                 base_filters: int = 32,
                 num_blocks: List[int] = [2, 2, 2, 2],
                 dropout: float = 0.5):
        """
        Initialize ResNet1D.

        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            base_filters: Number of filters in first layer
            num_blocks: Number of residual blocks in each layer
            dropout: Dropout probability
        """
        super(ResNet1D, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Initial convolution
        self.conv1 = nn.Conv1d(in_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(base_filters, base_filters, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(base_filters, base_filters * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(base_filters * 2, base_filters * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(base_filters * 4, base_filters * 8, num_blocks[3], stride=2)

        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_filters * 8, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Create a layer of residual blocks."""
        layers = []

        # First block (may downsample)
        layers.append(ResidualBlock1D(in_channels, out_channels, stride, downsample=(stride != 1)))

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling and classifier
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class MultiScaleCNN1D(nn.Module):
    """
    Multi-scale 1D CNN with parallel convolutional paths.

    Captures features at different temporal scales simultaneously.
    """

    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 3,
                 base_filters: int = 32,
                 dropout: float = 0.5):
        """
        Initialize MultiScaleCNN1D.

        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            base_filters: Number of filters per scale
            dropout: Dropout probability
        """
        super(MultiScaleCNN1D, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Multi-scale convolutional paths (different kernel sizes)
        self.scale1 = self._make_scale_path(in_channels, base_filters, kernel_size=3)
        self.scale2 = self._make_scale_path(in_channels, base_filters, kernel_size=5)
        self.scale3 = self._make_scale_path(in_channels, base_filters, kernel_size=7)
        self.scale4 = self._make_scale_path(in_channels, base_filters, kernel_size=9)

        # Combine scales
        combined_filters = base_filters * 4

        # Additional convolutional blocks
        self.conv1 = self._make_conv_block(combined_filters, combined_filters * 2, kernel_size=3)
        self.conv2 = self._make_conv_block(combined_filters * 2, combined_filters * 4, kernel_size=3)
        self.conv3 = self._make_conv_block(combined_filters * 4, combined_filters * 8, kernel_size=3)

        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(combined_filters * 8, num_classes)

    def _make_scale_path(self, in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
        """Create a single-scale convolutional path."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def _make_conv_block(self, in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
        """Create a convolutional block."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Multi-scale feature extraction (parallel paths)
        x1 = self.scale1(x)  # kernel_size=3
        x2 = self.scale2(x)  # kernel_size=5
        x3 = self.scale3(x)  # kernel_size=7
        x4 = self.scale4(x)  # kernel_size=9

        # Concatenate multi-scale features
        x = torch.cat([x1, x2, x3, x4], dim=1)

        # Further processing
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Global pooling and classifier
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module, input_shape: Tuple[int, int, int]) -> str:
    """
    Get a summary of the model architecture.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch, channels, length)

    Returns:
        Summary string
    """
    from torchinfo import summary

    summary_str = str(summary(
        model,
        input_size=input_shape,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        verbose=0
    ))

    return summary_str


def create_model(model_name: str,
                in_channels: int = 1,
                num_classes: int = 3,
                **kwargs) -> nn.Module:
    """
    Factory function to create models by name.

    Args:
        model_name: Name of model ('simple', 'resnet', 'multiscale')
        in_channels: Number of input channels
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments

    Returns:
        Initialized model
    """
    model_name = model_name.lower()

    if model_name == 'simple':
        return SimpleCNN1D(in_channels, num_classes, **kwargs)
    elif model_name == 'resnet':
        return ResNet1D(in_channels, num_classes, **kwargs)
    elif model_name == 'multiscale':
        return MultiScaleCNN1D(in_channels, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":
    # Test models
    print("Testing CNN Models...")
    print("=" * 80)

    batch_size = 8
    in_channels = 1
    seq_length = 10000
    num_classes = 3

    # Create dummy input
    x = torch.randn(batch_size, in_channels, seq_length)

    # Test SimpleCNN1D
    print("\n1. SimpleCNN1D")
    print("-" * 80)
    model1 = SimpleCNN1D(in_channels=in_channels, num_classes=num_classes)
    output1 = model1(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output1.shape}")
    print(f"Parameters: {count_parameters(model1):,}")

    # Test ResNet1D
    print("\n2. ResNet1D")
    print("-" * 80)
    model2 = ResNet1D(in_channels=in_channels, num_classes=num_classes)
    output2 = model2(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output2.shape}")
    print(f"Parameters: {count_parameters(model2):,}")

    # Test MultiScaleCNN1D
    print("\n3. MultiScaleCNN1D")
    print("-" * 80)
    model3 = MultiScaleCNN1D(in_channels=in_channels, num_classes=num_classes)
    output3 = model3(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output3.shape}")
    print(f"Parameters: {count_parameters(model3):,}")

    # Model comparison
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(f"{'Model':<20} {'Parameters':<15} {'Output Shape'}")
    print("-" * 80)
    print(f"{'SimpleCNN1D':<20} {count_parameters(model1):<15,} {str(output1.shape)}")
    print(f"{'ResNet1D':<20} {count_parameters(model2):<15,} {str(output2.shape)}")
    print(f"{'MultiScaleCNN1D':<20} {count_parameters(model3):<15,} {str(output3.shape)}")

    print("\n" + "=" * 80)
    print("ALL MODEL TESTS PASSED!")
    print("=" * 80)
