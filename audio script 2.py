import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

n_perseg = 256
n_frequencies = 32

# Define function to read CSV file
def read_csv(filename):
    df = pd.read_csv(filename, header=None)
    return df.to_numpy()

# Define function to preprocess data
def preprocess_data(data):
    data = np.squeeze(data)
    data = np.asarray(data, dtype=np.float32)
    data = np.nan_to_num(data)
    return data

# Define function to generate spectrogram data
def generate_spectrogram_data(data, sampling_rate):
    f, t, spectrogram_data = signal.spectrogram(data, fs=sampling_rate, nperseg=n_perseg, noverlap=int(n_perseg/2), scaling='spectrum')
    # spectrogram_data = 20*np.log10(np.abs(spectrogram_data) + 1e-10)
    return f, t, spectrogram_data

# Ask user for input files
"""
main_file = input("Enter the filename for the main data: ")
noise_file = input("Enter the filename for the background noise: ")
subtract_noise = input("Do you want to subtract the background noise? (y/n): ")
"""
main_file = "Data/Drone1_Flight1/Array_D1F1.csv"
noise_file = "Data/Background/Array_Background.csv"
subtract_noise = 'n'
# Read data from files
main_data = read_csv(main_file)
noise_data = read_csv(noise_file)

# Preprocess data
main_data = preprocess_data(main_data)
noise_data = preprocess_data(noise_data)
noise_data = noise_data[:, 16][:main_data.shape[0]]
freq = 1/50000
t = np.arange(0, 15, 1/50000)
main_data = x = 3*np.cos(2*np.pi*freq*t)

# Compute spectrogram data for noise
f_noise, t_noise, spectrogram_noise = generate_spectrogram_data(noise_data, 50000)

# Compute average noise level
avg_noise_level = np.mean(spectrogram_noise[:, :], axis=1)

# Compute spectrogram data for main data
f_main, t_main, spectrogram_main = generate_spectrogram_data(main_data, 50000)

# Subtract noise from main data
if subtract_noise.lower() == 'y':
    spectrogram_main = spectrogram_main - avg_noise_level.reshape(-1, 1)

# Crop spectrogram data
print(t_main.shape)
print(f_main.shape)
print(spectrogram_main.shape)
N_transforms = int((len(noise_data)-n_perseg/2)*2/n_perseg)
spectrogram_noise = spectrogram_noise[:n_frequencies, :N_transforms-1]
spectrogram_main = spectrogram_main[:n_frequencies, :N_transforms-1]
print(spectrogram_main.shape)

# Plot spectrogram
fig, axs = plt.subplots(1, 2, figsize=(16, 8))
if subtract_noise.lower() == 'y':
    axs[0].pcolormesh(t_main[:N_transforms], f_main[:n_frequencies+1], spectrogram_main, cmap='plasma', shading='auto')
    axs[0].set_title('Main Data (Noise-Free)')
else:
    axs[0].pcolormesh(t_noise[:N_transforms], f_noise[:n_frequencies+1], spectrogram_noise, cmap='plasma', shading='auto')
    axs[0].set_title('Background Noise')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Frequency (Hz)')
axs[1].pcolormesh(t_main[:N_transforms], f_main[:n_frequencies+1], spectrogram_main, cmap='plasma', shading='auto')
axs[1].set_title('Main Data')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Frequency (Hz)')
plt.axis(ymin=0, ymax=1000)
plt.show()
print('done')