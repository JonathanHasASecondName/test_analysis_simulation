import numpy as np
import csv
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import soundfile as sf

volume = 10000

with open("data/Array_D1F1.csv", 'r') as f:
    data = list(csv.reader(f, delimiter=","))

data = np.array(data, float)
data = data[0,:]
data = data[:len(data)-1]*volume

print("Data OK")

t = np.arange(0,15,1/50000)

print("Time OK")

# compute sample rate, assuming times are in seconds
sample_rate_hz = 50000
sf.write('recording.wav', data, sample_rate_hz)
print("Output OK")

# load audio file
sample_rate, data = wavfile.read('recording.wav')

# create spectrogram
frequencies, times, spectrogram = signal.spectrogram(data, fs=sample_rate)

# plot spectrogram
plt.subplot(2,1,1)
plt.title("Pressures")
plt.plot(t,data,"b",linewidth=0.05)
plt.ylabel('Pressure [Pa]')
plt.xlabel('Time [sec]')

plt.subplot(2,1,2)
plt.title("Spectrogram")
plt.ylim([0,1000])
plt.pcolormesh(times, frequencies, spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.tight_layout()
plt.show()

print("Plot OK")

