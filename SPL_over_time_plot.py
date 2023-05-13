"""
-------------------------------------
SPL Plot Script
-------------------------------------
by Roeland Oosterveld

The following code produces the sound pressure level (SPL) graphs for every drone flight.
"""

import os
from matplotlib import pyplot as plt
import numpy as np
import csv
import pywt

import pandas as pd

drone_number = str(5)

# open csv file
direc = f'C:\\Users\\ooste\\.spyder-py3'
os.chdir(direc)
with open(f"Array_D{drone_number}F1_mic_12.csv", newline='') as f:
    reader = csv.reader(f)
    raw_data = np.array([[float(val) for val in row if val != ''] for row in reader])
data = raw_data.T
data = data[0]

# setting up some important variables and truncating the data accoringly
Fs = 50000
dt = 1 / Fs
starting_point = 0
length_sample = 15
end_point = (starting_point + length_sample)
data = data[int(starting_point * Fs):int(Fs * end_point)]

# creating time array
n = 5000
time = np.arange(starting_point, end_point, dt * n)
p0 = 0.00002


# devides data in chuncks with n number of data points
def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


data = data[:int(Fs * length_sample)]
chunked_data = list(divide_chunks(data, n))

# get the root mean square of each chunk
rms_data = []
for i in range(len(chunked_data)):
    x = np.mean(chunked_data[i] ** 2)
    rms_data.append(np.sqrt(x))

# calculating the sound pressure level for each chunk
SPL = []
for i in rms_data:
    if i != 0:
        x = 20 * np.log10(i / p0)
    else:
        x = -60
    SPL.append(x)

# setting up regression polynomial of the SPL data
p = 8  # degree of regression polynomial
df = pd.DataFrame({'x': time, 'y': SPL})
model = model = np.poly1d(np.polyfit(df.x, df.y, p))
reg_line = time

fit_line = model(reg_line)
max_osp = max(fit_line)
ind_max = np.where(fit_line == max_osp)
time_max_osp = time[ind_max]
print(time_max_osp)

# plotting both the data and the regression polynomial
plt.subplot(1, 1, 1).minorticks_on()
plt.plot(time, SPL, 'royalblue', label='actual SPL', )
plt.plot(reg_line, model(reg_line), 'crimson', label='regression of SPL', linewidth=0.9)
plt.xlabel('Time [$s$]')
plt.ylabel('Sound pressure level [$dB$]')
plt.grid(linestyle='-', which='major', linewidth=0.95)
plt.grid(linestyle=':', which='minor', linewidth=0.5)
plt.legend()
plt.show()
print(max(SPL))
