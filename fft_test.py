import matplotlib.pyplot as plt
import numpy as np
from numpy import fft


plt.style.use('seaborn-poster')
# sampling rate
sr = 100
# sampling interval
ts = 1.0/sr
t = np.arange(0,2,ts)

freq = 3
x = 3*np.cos(2*np.pi*freq*t)



plt.show()


X = fft.fft(x)
N = len(X)
n = np.arange(N)
T = N/sr
freq = n/T

plt.figure(figsize=(12, 6))
plt.subplot(121)

plt.stem(freq, np.abs(X), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.xlim(0, 10)

plt.subplot(122)
plt.plot(t, fft.ifft(X), 'r')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()
