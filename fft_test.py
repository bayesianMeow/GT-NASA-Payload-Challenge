import numpy as np
import matplotlib.pyplot as plt



# Parameters
sampling_rate = 1000000  # 1 MHz sampling rate
file_path = 'raw_data.txt'  # Path to the text file containing raw data

# Read raw data from the SD card (simulated as a binary file)
def read_raw_data(file_path):
    with open(file_path, 'rb') as f:
        raw_data = np.fromfile(f, dtype=np.float32)  # Assuming 32-bit float data
    return raw_data

# Apply FFT to the raw data
def apply_fft(data, sampling_rate):
    n = len(data)
    fft_result = np.fft.fft(data)
    fft_freqs = np.fft.fftfreq(n, 1/sampling_rate)
    return fft_result, fft_freqs

# Decompose noise and signal
def decompose_noise(fft_result, fft_freqs, noise_threshold):
    magnitude = np.abs(fft_result)
    noise_mask = magnitude < noise_threshold
    signal_mask = ~noise_mask

    noise_fft = fft_result * noise_mask
    signal_fft = fft_result * signal_mask

    return noise_fft, signal_fft

# Reconstruct time-domain signal from FFT
def inverse_fft(fft_result):
    return np.real(np.fft.ifft(fft_result))  # Take the real part for real-valued signals

# Plot the frequency spectrum
def plot_spectrum(fft_freqs, fft_result, title):
    plt.figure(figsize=(10, 6))
    plt.plot(fft_freqs[:len(fft_freqs)//2], np.abs(fft_result[:len(fft_result)//2]))
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

# Main processing function
def process_data(file_path, sampling_rate, noise_threshold):
    # Step 1: Read raw data
    raw_data = read_raw_data(file_path)

    # Step 2: Apply FFT
    fft_result, fft_freqs = apply_fft(raw_data, sampling_rate)

    # Step 3: Decompose noise and signal
    noise_fft, signal_fft = decompose_noise(fft_result, fft_freqs, noise_threshold)

    # Step 4: Reconstruct time-domain signals
    noise_signal = inverse_fft(noise_fft)
    clean_signal = inverse_fft(signal_fft)

    # Step 5: Plot the results
    plot_spectrum(fft_freqs, fft_result, "Original Frequency Spectrum")
    plot_spectrum(fft_freqs, noise_fft, "Noise Frequency Spectrum")
    plot_spectrum(fft_freqs, signal_fft, "Signal Frequency Spectrum")

    return clean_signal, noise_signal

# Example usage
noise_threshold = 100  # Adjust this threshold based on your data
clean_signal, noise_signal = process_data(file_path, sampling_rate, noise_threshold)