import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

n_perseg = 1024*8*4
n_frequencies = 1024*64

flight_number = str(5)

def read_csv(filename):
    df = pd.read_csv(filename, header=None)
    return df.to_numpy()

def preprocess_data(data):
    data = np.squeeze(data)
    data = np.asarray(data, dtype=np.float32)
    data = np.nan_to_num(data)
    return data

def preprocess_data(data):
    data = np.squeeze(data)
    data = np.asarray(data, dtype=np.float32)
    data = np.nan_to_num(data)
    return data

main_file = f"data/Drone{flight_number}_Flight1/Array_D{flight_number}F1.csv"
# noise_file = "Data/Background/Array_Background.csv"
subtract_noise = 'n'
# Read data from files
main_data = read_csv(main_file)
# noise_data = read_csv(noise_file)
main_data = preprocess_data(main_data)

# noise_data = preprocess_data(noise_data)
# take microphone 16
# noise_data = noise_data[:, 16][:main_data.shape[0]]

f, t, Sxx = signal.spectrogram(main_data, fs=50000, nperseg=n_perseg, nfft=int(n_perseg*16), noverlap=int(n_perseg*0.8))
Sxx = 10 * np.log10(Sxx)
Sxx[Sxx < -125] = -125
plt.pcolormesh(t, f[:n_frequencies], Sxx[:n_frequencies, :], cmap='jet')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.yscale("log")
plt.colorbar(label="Power/Frequency (dB/Hz)", orientation="vertical")
plt.axis(ymin=10, ymax=500)
plt.savefig(f'flight{flight_number}spectrogram_v3')
plt.show()

