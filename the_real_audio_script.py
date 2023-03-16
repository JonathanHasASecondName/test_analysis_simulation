import numpy as np
import csv
import matplotlib.pyplot as plt
import soundfile as sf

with open("data/Array_D1F1.csv", 'r') as f:
    data = list(csv.reader(f, delimiter=","))

data = np.array(data, float)
data = data[0,:]
data = data[:len(data)-1]

print("Data OK")

t = np.arange(0,15,1/50000)

print("Time OK")

plt.plot(t,data,"b")
plt.show()

print("Plot OK")

# compute sample rate, assuming times are in seconds
sample_rate_hz = 50000
sf.write('recording.wav', data, sample_rate_hz)

print("Output OK")