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

    ### ---- MICROPHONE 12 ---- ###

    # Read Raw Data
    main_file = f"newdata/Drone{flight_number}_Flight1/Array_D{flight_number}F1.csv"
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
    weighter = np.sum(weights)
    variance = pca.explained_variance_ratio_
    red /= weighter

    # Post-Processing
    min = abs(np.min(red))
    red += min
    red = 10 * np.log10(red) #comment if [Pa]
    preinf = np.min(red[np.isfinite(red)]) #comment if [Pa]
    red[np.isneginf(red)] = preinf #comment if [Pa]
    maxpoint = np.where(red == np.amax(red))
    timeof = t[maxpoint]

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

    print("Cleaned Top Frequencies (Mic 12)",a)

    mic12_freqs = a
    mic12_eigenloudness = red
    mic12_specgram = Sxx
    mic12_t = t
    mic12_f = f
    mic12_close = timeof

    ### ---- MICROPHONE 16 ---- ###

    # Read Raw Data
    main_file = f"data/Drone{flight_number}_Flight1/Array_D{flight_number}F1.csv"
    main_data = read_csv(main_file)
    main_data = preprocess_data(main_data)

    # Produce Spectrogram Data
    f, t, Sxx = signal.spectrogram(main_data, fs=50000, nperseg=n_perseg, nfft=int(n_perseg * 16),
                                   noverlap=int(n_perseg * 0.8))

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
    red = np.array(pca.fit_transform(Sxx_legacy.T)[:, 0])
    weights = pca.components_.reshape(-1)
    weighter = np.sum(weights)
    variance = pca.explained_variance_ratio_
    red /= weighter

    # Post-Processing
    min = abs(np.min(red))
    red += min
    red = 10 * np.log10(red)  # comment if [Pa]
    preinf = np.min(red[np.isfinite(red)])  # comment if [Pa]
    red[np.isneginf(red)] = preinf  # comment if [Pa]
    maxpoint = np.where(red == np.amax(red))
    timeof = t[maxpoint]

    """
    Obtain Main Frequencies
    """

    probe = rd.randint(70, 150)
    counter = 0
    while True:
        top = np.argsort(-weights.T)[:probe]
        fund_w = [weights[i] for i in top]
        fund_f = np.asarray([f[i] for i in top])
        fund_f_round = np.round(fund_f, 0)
        f_set = set(fund_f_round)
        f_set = np.asarray(list(f_set))
        a = f_set
        mask = masking(a)
        while True:
            a = cropping(a)
            mask = masking(a)
            if len(mask) * 2 == sum(mask):
                for i in range(len(a)):
                    a[i] = round(a[i], 1)
                    a = np.array(a)
                break
        if len(a) < target_freqs:
            probe += rd.randint(5, 20)
            counter += 1
        if len(a) > target_freqs:
            probe -= rd.randint(5, 20)
            counter += 1
        if len(a) == target_freqs:
            break
        if counter >= 100:
            target_freqs += 1
            counter = 0

    print("Cleaned Top Frequencies (Mic 16)", a)

    mic16_freqs = a
    mic16_eigenloudness = red
    mic16_specgram = Sxx
    mic16_t = t
    mic16_f = f
    mic16_close = timeof

    """
    Plot Graphs
    """

    img = plt.imread("drone.png")

    fig, ax = plt.subplots(3, 1, gridspec_kw={'height_ratios': [3, 4, 1]})

    fig.suptitle(f"Drone {flight_number} - Var: {round(float(variance)*100,1)}%")

    ax[0].minorticks_on()
    ax[0].plot(mic16_t, mic16_eigenloudness,color="blue")
    ax[0].plot(mic12_t, mic12_eigenloudness, color="red")
    ax[0].set_ylabel('Eigenloudness [dB]')
    #ax[0].imshow(img,extent=[timeof,timeof+img.shape[1],red[maxpoint],red[maxpoint]+img.shape[0]])
    ax[0].set_xlabel('Time [sec]')
    ax[0].vlines(timeof, np.min(mic16_eigenloudness) * 1.05, np.max(mic16_eigenloudness) * 0.95,
                 label=f"Closest Approach: t= {round(float(mic16_close), 1)} s", colors="blue")
    ax[0].vlines(timeof, np.min(mic16_eigenloudness) * 1.05, np.max(mic16_eigenloudness) * 0.95,
                 label=f"Closest Approach: t= {round(float(mic12_close), 1)} s", colors="red")
    ax[0].set_ylim(np.min(mic16_eigenloudness) * 1.05 , np.max(mic16_eigenloudness) * 0.95)
    ax[0].set_xlim(t[0], t[-1])
    ax[0].legend(fontsize="6")
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
    plt.savefig(fname=f"Drone {flight_number} PCA New (Mic 12)",dpi=900)

    plt.show()