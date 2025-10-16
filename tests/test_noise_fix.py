"""
Quick test to verify noise reduction fix
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.synthetic_generator import SyntheticSignalGenerator
import matplotlib.pyplot as plt

# Generate 3um signal with default settings
generator = SyntheticSignalGenerator(seed=42)
time_data, current_data = generator.generate_signal('3um', noise_level=1.0)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
ax.plot(time_data / 1000, current_data, linewidth=0.5, color='blue')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Current (A)')
ax.set_title('3um Synthetic Signal - After Noise Reduction Fix')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test_noise_fix_3um.png', dpi=150, bbox_inches='tight')
print("Saved test_noise_fix_3um.png")

# Print statistics
print(f"\nSignal Statistics:")
print(f"  Length: {len(current_data):,} points")
print(f"  Duration: {time_data[-1]/1000:.1f} seconds")
print(f"  Mean: {current_data.mean():.4f}")
print(f"  Std: {current_data.std():.4f}")
print(f"  Min: {current_data.min():.4f}")
print(f"  Max: {current_data.max():.4f}")
