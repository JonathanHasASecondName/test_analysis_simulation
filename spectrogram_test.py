import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
import scipy

# sampling rate
sr = 2000
# sampling interval
ts = 1.0/sr
t = np.arange(0,1,ts)

freq = 5
x = 3*np.cos(2*np.pi*freq*t)

"""
plt.subplot(122)
plt.plot(t, x, 'r')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
"""
if True:
    f, t, Sxx = scipy.signal.spectrogram(
        x=x,
        fs=sr,
        window=('tukey', 0.25),
        
    )
    plt.figure(figsize=(8, 10))
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

plt.show()
