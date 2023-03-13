import csv
import wave
import numpy as np
import matplotlib.pyplot as plt
# bruh:w
# Load the main file
with open('main_file.csv', newline='') as f:
    reader = csv.reader(f)
    data = np.array([[float(val) for val in row if val != ''] for row in reader])

# Load the background noise file and calculate its average
subtract_noise = input("Do you want to subtract background noise? (y/n): ").lower() == "y"
if subtract_noise:
    with open('background_noise.csv', newline='') as f:
        reader = csv.reader(f)
        noise_data = np.array([[float(val) for val in row if val != ''] for row in reader])
        noise_avg = np.mean(noise_data, axis=0)

        # Subtract the noise average from the main file
        clean_data = data - noise_avg
else:
    clean_data = data

# Normalize the data
max_val = np.abs(clean_data).max()
if max_val > 1.0:
    clean_data /= max_val

# Set the audio parameters
num_channels = 1
sample_rate = 50000

# Write the audio files
with wave.open('original_audio.wav', 'w') as file:
    file.setparams((num_channels, 3, sample_rate, 0, 'NONE', 'Uncompressed'))
    file.writeframes(data.tobytes())

with wave.open('clean_audio.wav', 'w') as file:
    file.setparams((num_channels, 3, sample_rate, 0, 'NONE', 'Uncompressed'))
    file.writeframes(clean_data.tobytes())

# Compute and plot the spectrograms
plt.figure(figsize=(12, 6))

# Original spectrogram
plt.subplot(1, 2, 1)
plt.specgram(data.flatten(), NFFT=4096*128, Fs=sample_rate)
plt.ylim(1200, 1500)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Original Spectrogram')

# Cleaned spectrogram
plt.subplot(1, 2, 2)
plt.specgram(clean_data.flatten(),NFFT=4096*128,  Fs=sample_rate)
plt.ylim(1200, 1500)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Cleaned Spectrogram')

plt.tight_layout()
plt.show()
