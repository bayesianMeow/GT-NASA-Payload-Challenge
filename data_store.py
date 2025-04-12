import numpy as np
import matplotlib.pyplot as plt

# Parameters
sampling_rate = 1000000  # 1 MHz
duration = 0.01  # 10 ms of data
num_samples = int(sampling_rate * duration)
t = np.linspace(0, duration, num_samples, endpoint=False)

# Create synthetic signal
signal_freq1 = 50000  # 50 kHz
signal_freq2 = 120000  # 120 kHz
interference_freq = 450000  # 450 kHz

signal = (
    1.0 * np.sin(2 * np.pi * signal_freq1 * t) +
    0.7 * np.sin(2 * np.pi * signal_freq2 * t) +
    0.5 * np.random.randn(num_samples) +
    0.3 * np.sin(2 * np.pi * interference_freq * t)
)

# Save to binary file (for FFT processing)
signal_float32 = signal.astype(np.float32)
with open('raw_data.bin', 'wb') as f:
    signal_float32.tofile(f)

# Save to text file (human-readable)
with open('raw_data.txt', 'w') as f:
    for time, value in zip(t, signal):
        f.write(f"{time:.6f}, {value:.6f}\n")  # Time (s), Amplitude

print(f"Generated files: 'raw_data.bin' (binary) and 'raw_data.txt' (text)")