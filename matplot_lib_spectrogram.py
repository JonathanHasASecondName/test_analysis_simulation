import matplotlib.pyplot as plt
import numpy as np

sample_time = 1000
sr = 2048
# sampling interval
ts = 1.0/sr
t = np.arange(0,sample_time,ts)
NFFT = 1024*16

freq = 25
x = 3*np.cos(2*np.pi*freq*t)
x[int(len(t)/2):] = 3*np.cos(2*np.pi*freq*2*t[int(len(t)/2):])

fig, (ax1, ax2) = plt.subplots(nrows=2)
ax1.plot(t, x)
Pxx, freqs, bins, im = ax2.specgram(x, NFFT=NFFT, Fs=sr)
plt.axis(ymin=0, ymax=100)
plt.show()
