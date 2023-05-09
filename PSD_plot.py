# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:30:19 2023

@author: ooste
"""

from matplotlib import pyplot as plt
import numpy as np
import csv
import sys
from scipy import signal
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
import os

drone_number=str(5)
starting_points=[14,8,7.8,5.8,5]
#open csv file
direc = f'C:\\Users\\ooste\\.spyder-py3\\Drone{drone_number}_Flight1'
os.chdir(direc)
with open(f"Array_D{drone_number}F1.csv", newline='') as f:
    reader=csv.reader(f)
    raw_data=np.array([[float(val) for val in row if val != ''] for row in reader])

data=raw_data.T
data=data[0]

Fs=50000

starting_point=starting_points[(int(drone_number)-1)]
print(starting_point)
length_sample=1
end_point=(starting_point+length_sample)
data=data[int(starting_point*Fs-1):int(Fs*end_point)]


nfft=32
p0=0.00002

Pxx, freq=plt.psd(data,Fs=Fs, NFFT=256*nfft, scale_by_freq=True, color='royalblue')
max_value=max(Pxx[24:100])
index_max=np.where(Pxx == max_value)
base_freq=freq[index_max]
print(base_freq)

harmonics=[]
for i in range(1,13):
    tone=i*base_freq
    harmonics.append(tone)
    
    
for i in harmonics:
    plt.axvline(x=i, color='crimson', linewidth=0.8)
  
    
plt.subplot(1,1,1).minorticks_on()
plt.xlabel('Frequency [$Hz$]')
plt.ylabel("Power spectral density [$dB/Hz$]")
plt.xlim([0,2000])
plt.ylim([-140,-85])
plt.grid(linestyle='-', which='major', linewidth=0.95)
plt.grid(linestyle=':', which='minor', linewidth=0.5)
plt.show()

"""
freq, Pxx=signal.welch(data, Fs, nperseg=8*1024)
pxx_decibel=[]
for i in range(len(Pxx)):
    decibel=20*np.log10(Pxx[i]/p0)
    pxx_decibel.append(decibel)
plt.plot(freq, pxx_decibel)
plt.xlabel('frequency [Hz]')
plt.ylabel("Power spectral density [Db/Hz]")
plt.xlim([0,2000])
plt.ylim([-200,-85])
plt.show()

nfft=32
(f,S)=signal.periodogram(data, Fs, scaling='density', nfft=nfft*256)
S_deci=[]
for i in range(len(S)):
    s=20*np.log10(S[i]/p0)
    S_deci.append(s)
plt.plot(f, S_deci)
plt.xlabel('frequency [Hz]')
plt.ylabel("Power spectral density [Db/Hz]")
plt.xlim([0,2000])
plt.ylim([-200,-70])
plt.show()
"""