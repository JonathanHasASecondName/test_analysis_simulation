import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from sklearn.decomposition import PCA
import numpy.random as rd

"""
Inputs & Functions
"""
# Inputs
n_perseg = 1024*8*4 # number of FFTs
n_frequencies = 1024*(2**6) # frequencies to show on spectrogram
flight_number = str(5) # flight number
target_freqs = 5 # number of "top" frequencies desired
hear = 20*(10**(-6)) # hearing threshold

# Function Definitions
def read_csv(filename):
    df = pd.read_csv(filename, header=None)
    return df.to_numpy()

def preprocess_data(data):
    data = np.squeeze(data)
    data = np.asarray(data, dtype=np.float32)
    data = np.nan_to_num(data)
    return data

def masking(freqs,tol=0.06):
    mask = []
    for i in range(len(freqs)-1):
        r = abs(round((freqs[i+1]-freqs[i])/freqs[i+1],3))
        if r <=tol:
            mask.append(0)
        if r > tol and (i==0 or mask[i-1] != 0):
            mask.append(2)
        if r > tol and mask[i-1] == 0:
            mask.append(1)
    return mask

def cropping(freqs):
    mask = masking(freqs)
    new = []
    for i in range(len(mask)):
        if mask[i] == 0: #append average
            new.append((freqs[i]+freqs[i+1])/2)
        if mask[i] == 1:
            if i==len(mask)-1:
                new.append(freqs[i + 1])
        if mask[i] == 2: #append itself
            new.append(freqs[i])
            if i==len(mask)-1:
                new.append(freqs[i+1])
    return new

"""
Read, Produce & Truncate Data
"""

for flight_number in range(1, 6):


    # Read Raw Data
    main_file = f"data/Drone{flight_number}_Flight1/Array_D{flight_number}F1.csv"
    main_data = read_csv(main_file)
    main_data = preprocess_data(main_data)

    # Produce Spectrogram Data
    f, t, Sxx = signal.spectrogram(main_data, fs=50000, nperseg=n_perseg, nfft=int(n_perseg*16), noverlap=int(n_perseg*0.8))
    # Truncate Data

    f = f[:n_frequencies]
    Sxx_legacy = Sxx
    Sxx = 10 * np.log10(Sxx_legacy)
    Sxx[Sxx < -125] = -125
    Sxx = Sxx[:n_frequencies, :]

    """
    Perform PCA
    -Sxx: Pressure Levels in [dB]
    -Sxx_legacy: Pressure Levels in [Pa]
    """
    # Reduce
    pca = PCA(1)
    red = np.array(pca.fit_transform(Sxx_legacy.T)[:,0])
    weights = pca.components_.reshape(-1)
    variance = pca.explained_variance_ratio_

    # Post-Processing
    min = abs(np.min(red))
    red += min
    red = 10 * np.log10(red) #comment if [Pa]
    preinf = np.min(red[np.isfinite(red)]) #comment if [Pa]
    red[np.isneginf(red)] = preinf #comment if [Pa]

    """
    Obtain Main Frequencies
    """

    probe = rd.randint(70,150)
    counter = 0
    while True:
        top = np.argsort(-weights.T)[:probe]
        fund_w = [weights[i] for i in top]
        fund_f = np.asarray([f[i] for i in top])
        fund_f_round = np.round(fund_f,0)
        f_set = set(fund_f_round)
        f_set = np.asarray(list(f_set))
        a = f_set
        mask = masking(a)
        while True:
            a = cropping(a)
            mask = masking(a)
            if len(mask)*2 == sum(mask):
                for i in range(len(a)):
                    a[i] = round(a[i],1)
                    a = np.array(a)
                break
        if len(a) < target_freqs:
            probe += rd.randint(5,20)
            counter +=1
        if len(a) > target_freqs:
            probe -= rd.randint(5,20)
            counter += 1
        if len(a) == target_freqs:
            break
        if counter >= 100:
            target_freqs += 1
            counter = 0

    #text_f = []
    #for i in range(len(a)):
    #    text_f.append(str(a[i]))

    print("Cleaned Top Frequencies",a)

    """
    Plot Graphs
    """
    fig, ax = plt.subplots(3, 1, gridspec_kw={'height_ratios': [3, 4, 1]})

    fig.suptitle(f"Drone {flight_number} - Var: {round(float(variance)*100,1)}%")

    ax[0].minorticks_on()
    ax[0].plot(t, red)
    ax[0].set_ylabel('Eigenloudness [dB]')
    ax[0].set_xlabel('Time [sec]')
    ax[0].set_xlim(t[0], t[-1])
    ax[0].grid(linestyle='-', which='major', linewidth=0.9)
    ax[0].grid(linestyle=':', which='minor',linewidth=0.5)

    ax[1].minorticks_on()
    ax[1].pcolormesh(t, f, Sxx, cmap='jet')
    ax[1].set_ylabel('Frequency [Hz]')
    ax[1].set_xlabel('Time [sec]')
    ax[1].set_xlim(t[0], t[-1])
    mappable = ax[1].pcolormesh(t, f, Sxx, cmap='jet')
    fig.colorbar(mappable, ax=ax[1], label="Power/Frequency [dB/Hz]", orientation="vertical")
    ax[1].set_yscale("log")
    ax[1].axis(ymin=10, ymax=500)

    ax[2].axis('off')
    ax[2].table(cellText=[a], colLabels=None, cellLoc='center', loc='center')
    ax[2].set_title(f"Top {target_freqs} Characteristic Frequencies [Hz]", y=0.7,fontsize=12)

    plt.subplots_adjust(hspace=0.25)
    plt.tight_layout()
    plt.savefig(fname=f"Drone {flight_number} PCA New (dB)",dpi=900)

    plt.show()

"""
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
"""