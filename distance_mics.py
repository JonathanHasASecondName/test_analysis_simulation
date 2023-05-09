import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from sklearn.decomposition import PCA
import numpy.random as rd
import pywt
from scipy.stats import linregress

# Constants
p0 = 20*(10**(-6))

# Function Definitions
def read_csv(filename):
    df = pd.read_csv(filename, header=None)
    return df.to_numpy()

def preprocess_data(data):
    data = np.squeeze(data)
    data = np.asarray(data, dtype=np.float32)
    data = np.nan_to_num(data)
    return data

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

def denoising(S,k=0.5):
    threshold = np.std(S) * np.sqrt(2 * np.log(len(S)))
    coeffs = pywt.wavedec(S, 'db4', mode='per')
    coeffs[1:] = (pywt.threshold(i, value=k * threshold, mode='soft') for i in coeffs[1:])
    return pywt.waverec(coeffs, 'db4', mode='per')

def remove_outliers(data, k):
    mean = np.mean(data)
    stdev = np.std(data)
    threshold = k * stdev
    filtered_data = [x for x in data if abs(x - mean) <= threshold]
    return filtered_data

def data(flight_number,mic,batch=5000):
    if mic == 12:
        main_file = f"newdata/Drone{flight_number}_Flight1/Array_D{flight_number}F1.csv"
        main_data = read_csv(main_file)
        main_data = preprocess_data(main_data)
        main_data = list(divide_chunks(main_data, batch))[0]
        main_data = denoising(main_data)
        rms_data = []
        for i in range(len(main_data)):
            x = np.mean(main_data[i] ** 2)
            rms_data.append(np.sqrt(x))
        return rms_data
    else:
        main_file = f"data/Drone{flight_number}_Flight1/Array_D{flight_number}F1.csv"
        main_data = read_csv(main_file)
        main_data = preprocess_data(main_data)
        main_data = list(divide_chunks(main_data, batch))[0]
        main_data = denoising(main_data)
        rms_data = []
        for i in range(len(main_data)):
            x = np.mean(main_data[i] ** 2)
            rms_data.append(np.sqrt(x))
        return rms_data

def ctz(lst):
    closest = lst[0]
    for i in range(1, len(lst)):
        if abs(lst[i]) < abs(closest):
            closest = lst[i]
    return closest

def good_log(data):
    mic = []
    c = ctz(data)
    for i in data:
        if i != 0:
            x = 20 * np.log10(i / p0)
        else:
            x = 20 * np.log10(c / p0)
        mic.append(x)
    return mic

diffs = []
for i in range(1,6):

    b = 100

    main_file_16 = data(i,16,b)
    main_file_12 = data(i,12,b)

    mic16 = good_log(main_file_16)
    mic12 = good_log(main_file_12)

    diff = np.array(mic12)-np.array(mic16)
    diff_n = remove_outliers(diff,1)

    #diffs.append(diff)
    diffs.append(diff_n)

fig, ax = plt.subplots(1, 1)
ax.minorticks_on()
ax.grid(linestyle='-', which='major', linewidth=0.9)
ax.grid(linestyle=':', which='minor', linewidth=0.5)
ax.axhspan(0, 20, alpha=0.05, color='blue')
ax.axhspan(-20, 0, alpha=0.05, color='red')


colors = ["red","blue","green","#A52A2A","magenta"]

for i in range(0,5):
    t = np.linspace(0,15,len(diffs[i]))
    slope, intercept, r_value, p_value, std_err = linregress(t, diffs[i])
    r = slope*t+intercept
    plt.plot(t,diffs[i],linewidth=1,label=f"Drone {i+1}",color=colors[i],alpha=0.5)
    plt.plot(t, r, linewidth=0.8,linestyle='--',color=colors[i])

plt.text(0.2, 17, 'Mic 12', fontsize=15, color='blue')
plt.text(0.2, -18.5, 'Mic 16', fontsize=15, color='red')
plt.hlines(0,0,15,"k",linewidth=0.7)
plt.xlabel("Time [s]")
plt.ylabel("Loudness Difference [dB]")
plt.xlim((0,15))
plt.ylim((-20,20))
plt.xticks(range(0, 16, 3))
plt.legend(fontsize=7)
plt.savefig(fname=f"Drone Distance Analysis",dpi=900)
plt.show()




