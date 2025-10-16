"""
Analyze real signal step characteristics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load real signals
data_dir = Path('TestData')
files = {
    '1um': list(data_dir.glob('PS 1um*.csv'))[0],
    '2um': list(data_dir.glob('PS 2um*.csv'))[0],
    '3um': list(data_dir.glob('PS 3um*.csv'))[0]
}

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle('Real Signal Analysis - Step Patterns', fontsize=14, fontweight='bold')

for idx, (class_name, file_path) in enumerate(files.items()):
    df = pd.read_csv(file_path)
    time = df.iloc[:, 0].values
    current = df.iloc[:, 1].values

    # Full signal
    ax_full = axes[idx, 0]
    ax_full.plot(time / 1000, current, linewidth=0.5)
    ax_full.set_xlabel('Time (s)')
    ax_full.set_ylabel('Current')
    ax_full.set_title(f'{class_name} - Full Signal')
    ax_full.grid(True, alpha=0.3)

    # Zoomed view (first 20%)
    zoom_length = int(len(current) * 0.2)
    ax_zoom = axes[idx, 1]
    ax_zoom.plot(time[:zoom_length] / 1000, current[:zoom_length], linewidth=0.8)
    ax_zoom.set_xlabel('Time (s)')
    ax_zoom.set_ylabel('Current')
    ax_zoom.set_title(f'{class_name} - Zoomed (first 20%)')
    ax_zoom.grid(True, alpha=0.3)

    print(f'\n{class_name}: {file_path.name}')
    print(f'  Mean: {current.mean():.4f}')
    print(f'  Std: {current.std():.4f}')
    print(f'  Range: {current.max() - current.min():.4f}')

plt.tight_layout()
plt.savefig('real_signals_analysis.png', dpi=150, bbox_inches='tight')
print('\nSaved to: real_signals_analysis.png')
