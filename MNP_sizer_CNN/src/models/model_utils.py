"""
Model Utilities
Save, load, and analyze PyTorch models.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple
import json


def save_model(model: nn.Module,
              save_path: str,
              optimizer: Optional[torch.optim.Optimizer] = None,
              epoch: Optional[int] = None,
              metrics: Optional[Dict] = None):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        save_path: Path to save checkpoint
        optimizer: Optimizer state (optional)
        epoch: Current epoch (optional)
        metrics: Training metrics (optional)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    if metrics is not None:
        checkpoint['metrics'] = metrics

    torch.save(checkpoint, save_path)
    print(f"Model saved to: {save_path}")


def load_model(model: nn.Module,
              load_path: str,
              optimizer: Optional[torch.optim.Optimizer] = None,
              device: str = 'cpu') -> Tuple[nn.Module, Optional[torch.optim.Optimizer], Dict]:
    """
    Load model checkpoint.

    Args:
        model: Model instance (architecture should match saved model)
        load_path: Path to checkpoint
        optimizer: Optimizer to load state into (optional)
        device: Device to load model to

    Returns:
        Tuple of (model, optimizer, checkpoint_info)
    """
    load_path = Path(load_path)

    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    checkpoint = torch.load(load_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Extract checkpoint info
    info = {
        'epoch': checkpoint.get('epoch', None),
        'metrics': checkpoint.get('metrics', None),
        'model_class': checkpoint.get('model_class', 'Unknown')
    }

    print(f"Model loaded from: {load_path}")
    if info['epoch'] is not None:
        print(f"Epoch: {info['epoch']}")
    if info['metrics'] is not None:
        print(f"Metrics: {info['metrics']}")

    return model, optimizer, info


def save_model_config(config: Dict, save_path: str):
    """
    Save model configuration to JSON.

    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Config saved to: {save_path}")


def load_model_config(load_path: str) -> Dict:
    """
    Load model configuration from JSON.

    Args:
        load_path: Path to config file

    Returns:
        Configuration dictionary
    """
    load_path = Path(load_path)

    if not load_path.exists():
        raise FileNotFoundError(f"Config not found: {load_path}")

    with open(load_path, 'r') as f:
        config = json.load(f)

    print(f"Config loaded from: {load_path}")
    return config


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.

    Args:
        model: PyTorch model
        trainable_only: Count only trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_summary(model: nn.Module, input_shape: Tuple[int, ...]):
    """
    Print detailed model summary.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch, channels, length)
    """
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)

    print(f"\nModel: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model, trainable_only=False):,}")
    print(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")

    # Try to get detailed summary with torchinfo
    try:
        from torchinfo import summary
        print("\n" + "-" * 80)
        summary(
            model,
            input_size=input_shape,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            row_settings=["var_names"]
        )
    except ImportError:
        print("\nInstall torchinfo for detailed summary: pip install torchinfo")

    print("=" * 80)


def get_layer_names(model: nn.Module) -> list:
    """
    Get list of all layer names in model.

    Args:
        model: PyTorch model

    Returns:
        List of layer names
    """
    return [name for name, _ in model.named_modules() if name]


def freeze_layers(model: nn.Module, layer_names: list):
    """
    Freeze specified layers (set requires_grad=False).

    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if name.startswith(layer_name):
                param.requires_grad = False

    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Frozen {frozen_params:,} / {total_params:,} parameters")


def unfreeze_all_layers(model: nn.Module):
    """
    Unfreeze all layers (set requires_grad=True).

    Args:
        model: PyTorch model
    """
    for param in model.parameters():
        param.requires_grad = True

    print("All layers unfrozen")


def export_to_onnx(model: nn.Module,
                  save_path: str,
                  input_shape: Tuple[int, ...],
                  input_names: list = ['input'],
                  output_names: list = ['output']):
    """
    Export model to ONNX format.

    Args:
        model: PyTorch model
        save_path: Path to save ONNX model
        input_shape: Input tensor shape
        input_names: Input tensor names
        output_names: Output tensor names
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Create dummy input
    dummy_input = torch.randn(*input_shape)

    # Export
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        str(save_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            input_names[0]: {0: 'batch_size'},
            output_names[0]: {0: 'batch_size'}
        },
        opset_version=11
    )

    print(f"ONNX model saved to: {save_path}")


def test_model_inference(model: nn.Module,
                        input_shape: Tuple[int, ...],
                        device: str = 'cpu') -> float:
    """
    Test model inference speed.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device to run on

    Returns:
        Average inference time (ms)
    """
    import time

    model = model.to(device)
    model.eval()

    # Warm-up
    dummy_input = torch.randn(*input_shape).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Benchmark
    num_iterations = 100
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)

    end_time = time.time()
    avg_time_ms = ((end_time - start_time) / num_iterations) * 1000

    print(f"Average inference time: {avg_time_ms:.2f} ms")

    return avg_time_ms


if __name__ == "__main__":
    # Example usage
    print("Testing model utilities...")

    from cnn_models import SimpleCNN1D

    # Create model
    model = SimpleCNN1D(in_channels=1, num_classes=3)

    print("\n1. Model Summary:")
    print_model_summary(model, (8, 1, 10000))

    print("\n2. Save/Load Model:")
    save_model(model, "test_model.pth", epoch=10, metrics={'acc': 0.95})
    loaded_model, _, info = load_model(SimpleCNN1D(1, 3), "test_model.pth")
    print(f"Loaded model info: {info}")

    print("\n3. Inference Speed Test:")
    test_model_inference(model, (1, 1, 10000))

    print("\nModel utilities test complete!")
