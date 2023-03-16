import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf

with open("data/Array_D1F1.csv", 'r') as f:
    data = list(csv.reader(f, delimiter=","))

data = np.array(data, float)
data = data[0,:]
data = data[:len(data)-1]

t = np.arange(0,15,1/50000)

plt.plot(t,data,"b")
plt.show()

# compute sample rate, assuming times are in seconds
times = df['time'].values
n_measurements = len(times)
timespan_seconds = times[-1] - times[0]
sample_rate_hz = 50000

# write data
data = df['value'].values
sf.write('recording.wav', data, sample_rate_hz)

print(len(data))