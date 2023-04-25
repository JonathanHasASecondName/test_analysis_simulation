import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from sklearn.decomposition import PCA

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

main_file = f"data/Drone{flight_number}_Flight1/Array_D{flight_number}F1.csv"
subtract_noise = 'n'
# Read data from files
main_data = read_csv(main_file)
main_data = preprocess_data(main_data)

f, t, Sxx = signal.spectrogram(main_data, fs=50000, nperseg=n_perseg, nfft=int(n_perseg*16), noverlap=int(n_perseg*0.8))

f = f[:n_frequencies]

Sxx = 10 * np.log10(Sxx)
Sxx[Sxx < -125] = -125
Sxx = Sxx[:n_frequencies, :]

pca = PCA(1)
red = pca.fit_transform(Sxx.T)[:,0]


print(red, red.shape)

plt.subplot(2,1,1).minorticks_on()
plt.plot(t, red)
plt.grid(linestyle='-', which='major', linewidth=0.9)
plt.grid(linestyle=':', which='minor',linewidth=0.5)

plt.subplot(2,1,2)
plt.pcolormesh(t, f[:n_frequencies], Sxx[:n_frequencies, :], cmap='jet')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.yscale("log")
#plt.colorbar(label="Power/Frequency (dB/Hz)", orientation="vertical")
plt.axis(ymin=10, ymax=500)
plt.savefig(f'flight{flight_number}spectrogram_v3')
plt.show()