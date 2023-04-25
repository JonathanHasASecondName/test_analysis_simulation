import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from sklearn.decomposition import PCA

# Inputs
n_perseg = 1024*8*4
n_frequencies = 1024*64
flight_number = str(5)

# Function Definitions
def read_csv(filename):
    df = pd.read_csv(filename, header=None)
    return df.to_numpy()

def preprocess_data(data):
    data = np.squeeze(data)
    data = np.asarray(data, dtype=np.float32)
    data = np.nan_to_num(data)
    return data

# Read Data from Files
main_file = f"data/Drone{flight_number}_Flight1/Array_D{flight_number}F1.csv"
main_data = read_csv(main_file)
main_data = preprocess_data(main_data)

# Produce Spectrogram Data
f, t, Sxx = signal.spectrogram(main_data, fs=50000, nperseg=n_perseg, nfft=int(n_perseg*16), noverlap=int(n_perseg*0.8))

# Truncate & Process Data
f = f[:n_frequencies]
Sxx = 10 * np.log10(Sxx)
Sxx[Sxx < -125] = -125
Sxx = Sxx[:n_frequencies, :]
print("Frequencies",f,len(f))
print("Pressures",Sxx,len(Sxx),Sxx.shape)

# Perform PCA on Pressure Levels (Sxx)
pca = PCA(1)
red = pca.fit_transform(Sxx.T)[:,0]
weights = pca.components_.reshape(-1)
variance = pca.explained_variance_ratio_
print("PCAd Pressures",red,len(red))
print("Weights",weights,len(weights))
print("Variance",variance)

# Obtain Main Frequencies
top = np.argsort(weights.T)[:100]
fund_w = [weights[i] for i in top]
fund_f = np.asarray([f[i] for i in top])
fund_f_round = np.round(fund_f,0)
print(f"Top {len(top)} Positions",top)
print(f"Top {len(top)} Weights",fund_w)
print(f"Top {len(top)} Frequencies",fund_f)

f_set = set(fund_f_round)
#print(sorted(f_set, reverse=True))

f_set = np.asarray(list(f_set))
#print (f_set)

plt.subplot(1,1,1).minorticks_on()
plt.scatter(fund_w, fund_f)
plt.title("Frequencies vs Weigths")
plt.grid(linestyle='-', which='major', linewidth=0.9)
plt.grid(linestyle=':', which='minor',linewidth=0.5)
plt.show()

plt.subplot(1,1,1).minorticks_on()
plt.scatter(fund_w * fund_f, fund_f)
plt.title("Weighted Frequencies vs Weigths")
plt.grid(linestyle='-', which='major', linewidth=0.9)
plt.grid(linestyle=':', which='minor',linewidth=0.5)
plt.show()

# Plot Spectrogram and PWOSPL
fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 3]})

fig.suptitle(f"Drone {flight_number} - Var: {round(float(variance)*100,1)}%")

ax[0].minorticks_on()
ax[0].plot(t, red)
ax[0].set_ylabel('PWOSPL [dB]')
ax[0].set_xlabel('Time [sec]')
ax[0].set_xlim(t[0], t[-1])
ax[0].grid(linestyle='-', which='major', linewidth=0.9)
ax[0].grid(linestyle=':', which='minor',linewidth=0.5)

ax[1].minorticks_on()
ax[1].pcolormesh(t, f, Sxx, cmap='jet')
ax[1].set_ylabel('Frequency [Hz]')
ax[1].set_xlabel('Time [sec]')
ax[1].set_xlim(t[0], t[-1])
#plt.colorbar(label="Power/Frequency [dB/Hz]", orientation="horizontal",location='bottom')
ax[1].set_yscale("log")
ax[1].axis(ymin=10, ymax=500)
plt.subplots_adjust(hspace=0.3)
plt.tight_layout()
plt.savefig(fname=f"Drone {flight_number} PCA",dpi=900)
plt.show()