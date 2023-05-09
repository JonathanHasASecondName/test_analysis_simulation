import numpy as np
import csv
import soundfile as sf
import pywt

volume = 10000

with open("data/Drone5_Flight1/Array_D5F1.csv", 'r') as f:
    data = list(csv.reader(f, delimiter=","))

data = np.array(data, float)
data = data[:,0]
data = data[:len(data)-1]*volume

# Apply wavelet denoising
threshold = np.std(data) * np.sqrt(2 * np.log(len(data)))
coeffs = pywt.wavedec(data, 'db4', mode='per')
coeffs[1:] = (pywt.threshold(i, value=0.5*threshold, mode='soft') for i in coeffs[1:])
denoised_data = pywt.waverec(coeffs, 'db4', mode='per')

print("Data OK")

t = np.arange(0,15,1/50000)

print("Time OK")

# compute sample rate, assuming times are in seconds
sample_rate_hz = 50000
sf.write('recording_drone5_denoised.wav', denoised_data, sample_rate_hz)
print("Output OK")

"""
print("Data OK")
t = np.arange(0,15,1/50000)
print("Time OK")
# compute sample rate, assuming times are in seconds
sample_rate_hz = 50000
sf.write('recording_drone2.wav', data, sample_rate_hz)
print("Output OK")
# load audio file
sample_rate, data = wavfile.read('recording_drone2.wav')
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
plt.ylim([0,2500])
plt.pcolormesh(times, frequencies, spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.tight_layout()
plt.show()
print("Plot OK")
"""