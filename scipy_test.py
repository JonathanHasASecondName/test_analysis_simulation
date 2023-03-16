import numpy as np
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
rng = np.random.default_rng()
fs = 10e3
N = 1e5
amp = 2 * np.sqrt(2)
noise_power = 0.01 * fs / 2
time = np.arange(N) / float(fs)
carrier = amp * np.sin(2*np.pi*3*time)
x = carrier

n_perseg = 2048*4
f, t, Sxx = signal.spectrogram(x, fs, nperseg=n_perseg)
print(f.shape)
print(Sxx.shape)
print(t.shape)
print(Sxx[:30, :].shape)
plt.pcolormesh(t, f[:30], Sxx[:30, :], shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

