"""
Visualize real signals to understand step patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from data.synthetic_generator import SyntheticSignalGenerator

# Load one real signal from each class
data_dir = Path('TestData')

fig, axes = plt.subplots(3, 3, figsize=(16, 10))
fig.suptitle('Real vs Synthetic Signals Comparison', fontsize=14, fontweight='bold')

generator = SyntheticSignalGenerator(seed=42)

for idx, class_name in enumerate(['1um', '2um', '3um']):
    # Load real signal
    real_file = list(data_dir.glob(f'PS {class_name}*.csv'))[0]
    df = pd.read_csv(real_file)
    real_time = df.iloc[:, 0].values
    real_current = df.iloc[:, 1].values

    # Generate synthetic signal with reduced noise to see steps clearly
    synth_time, synth_current = generator.generate_signal(
        class_name,
        num_steps=5,
        noise_level=0.3,  # Lower noise to see steps
        add_drift=False,
        add_spikes=False
    )

    # Plot 1: Real signal - Full
    ax1 = axes[idx, 0]
    ax1.plot(real_time / 1000, real_current, linewidth=0.5, color='blue')
    ax1.set_title(f'{class_name.upper()} - Real (Full)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Current')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Real signal - Zoomed (first 10 seconds)
    ax2 = axes[idx, 1]
    mask = real_time <= 10000  # First 10 seconds
    ax2.plot(real_time[mask] / 1000, real_current[mask], linewidth=0.8, color='blue')
    ax2.set_title(f'{class_name.upper()} - Real (First 10s)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Current')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Synthetic signal - Zoomed (first 10 seconds)
    ax3 = axes[idx, 2]
    mask_synth = synth_time <= 10000
    ax3.plot(synth_time[mask_synth] / 1000, synth_current[mask_synth], linewidth=0.8, color='red')
    ax3.set_title(f'{class_name.upper()} - Synthetic (First 10s)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Current')
    ax3.grid(True, alpha=0.3)

    print(f"\n{class_name.upper()}:")
    print(f"  Real: mean={real_current.mean():.4f}, std={real_current.std():.4f}, range={real_current.max()-real_current.min():.4f}")
    print(f"  Synth: mean={synth_current.mean():.4f}, std={synth_current.std():.4f}, range={synth_current.max()-synth_current.min():.4f}")

plt.tight_layout()
plt.savefig('real_vs_synthetic_comparison.png', dpi=150, bbox_inches='tight')
print('\nSaved to: real_vs_synthetic_comparison.png')
