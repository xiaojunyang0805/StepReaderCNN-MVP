"""
Quick visualization of synthetic signals to verify step patterns.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from data.synthetic_generator import SyntheticSignalGenerator


def main():
    """Generate and visualize synthetic signals."""
    print("Generating synthetic signals with collision steps...")

    generator = SyntheticSignalGenerator(seed=42)

    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Synthetic Electrochemical Collision Signals', fontsize=14, fontweight='bold')

    for idx, class_name in enumerate(['1um', '2um', '3um']):
        print(f"\nGenerating {class_name} signal...")

        # Generate signal with specific number of steps for clarity
        time_data, current_data = generator.generate_signal(
            class_name,
            num_steps=5,  # Fixed number for visualization
            noise_level=0.5,  # Lower noise for clearer steps
            add_drift=False,
            add_spikes=False
        )

        # Plot full signal
        ax_full = axes[idx, 0]
        ax_full.plot(time_data / 1000, current_data, linewidth=0.5)
        ax_full.set_xlabel('Time (s)')
        ax_full.set_ylabel('Current')
        ax_full.set_title(f'{class_name} - Full Signal ({len(current_data):,} points)')
        ax_full.grid(True, alpha=0.3)

        # Plot zoomed view (first 20% to see steps clearly)
        zoom_length = int(len(current_data) * 0.2)
        ax_zoom = axes[idx, 1]
        ax_zoom.plot(time_data[:zoom_length] / 1000, current_data[:zoom_length], linewidth=0.8)
        ax_zoom.set_xlabel('Time (s)')
        ax_zoom.set_ylabel('Current')
        ax_zoom.set_title(f'{class_name} - Zoomed View (showing collision steps)')
        ax_zoom.grid(True, alpha=0.3)

        print(f"  Mean: {current_data.mean():.4f}")
        print(f"  Std: {current_data.std():.4f}")
        print(f"  Range: [{current_data.min():.4f}, {current_data.max():.4f}]")

    plt.tight_layout()

    # Save figure
    output_path = 'synthetic_signals_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
