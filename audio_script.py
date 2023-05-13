"""
-------------------------------------
Audio Script
-------------------------------------
by Gerard Mendoza Ferrandis

The following code generates an audio file from the raw data of the drone flights.
There is an option for de-noising.
"""

import numpy as np
import csv
import soundfile as sf
import pywt

volume = 10000

with open("data/Drone5_Flight1/Array_D5F1.csv", 'r') as f:
    data = list(csv.reader(f, delimiter=","))

data = np.array(data, float)
data = data[:, 0]
data = data[:len(data) - 1] * volume

# Apply wavelet denoising
threshold = np.std(data) * np.sqrt(2 * np.log(len(data)))
coeffs = pywt.wavedec(data, 'db4', mode='per')
coeffs[1:] = (pywt.threshold(i, value=0.5 * threshold, mode='soft') for i in coeffs[1:])
denoised_data = pywt.waverec(coeffs, 'db4', mode='per')

print("Data OK")

t = np.arange(0, 15, 1 / 50000)

print("Time OK")

# compute sample rate, assuming times are in seconds
sample_rate_hz = 50000
sf.write('recording_drone5_denoised.wav', denoised_data, sample_rate_hz)
print("Output OK")
