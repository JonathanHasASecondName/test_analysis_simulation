import matplotlib.pyplot as plt
import numpy as np

sr = 2000
# sampling interval
ts = 1.0/sr
t = np.arange(0,1,ts)
NFFT = 1024

freq = 5
scaling_factor = 10  # allows more detailed transforms in frequency
x = 3*np.cos(2*np.pi*freq*t*scaling_factor)
x[:1000] = 3*np.cos(2*np.pi*freq*t[:1000]*scaling_factor)
x[1000:] = 3*np.cos(2*np.pi*freq*2*t[1000:]*scaling_factor)

fig, (ax1, ax2) = plt.subplots(nrows=2)
ax1.plot(t, x)
Pxx, freqs, bins, im = ax2.specgram(x, NFFT=NFFT, noverlap=1, Fs=sr/scaling_factor)
plt.axis(ymin=0, ymax=10)
plt.show()
