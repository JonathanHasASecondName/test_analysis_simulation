"""
-------------------------------------
The PCA-Kmeans "Eigenloudness" & Frequency Analysis Method
-------------------------------------
by Gerard Mendoza Ferrandis

The following code produces the eigenloudness plots for every drone flight and every microphone.
The characteristic frequencies are obtained by clustering the frequencies that were deemed
more relevant by the PCA analysis. The frequency table breaks down what are the most important
frequencies according to each microphone, and the common ground using information from the cluster
of microphones (here only two microphones).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from sklearn.decomposition import PCA
import pywt
from sklearn.cluster import KMeans

"""
Inputs & Functions
"""
# Inputs
n_perseg = 1024 * 8 * 4  # number of FFTs
n_frequencies = 1024 * (2 ** 6)  # frequencies to show on spectrogram
flight_number = str(5)  # flight number
target_freqs = 5  # number of "top" frequencies desired
hear = 20 * (10 ** (-6))  # hearing threshold


# Function Definitions
def read_csv(filename):
    df = pd.read_csv(filename, header=None)
    return df.to_numpy()


def preprocess_data(data):
    data = np.squeeze(data)
    data = np.asarray(data, dtype=np.float32)
    data = np.nan_to_num(data)
    return data


def masking(freqs, tol=0.06):
    mask = []
    for i in range(len(freqs) - 1):
        r = abs(round((freqs[i + 1] - freqs[i]) / freqs[i + 1], 3))
        if r <= tol:
            mask.append(0)
        if r > tol and (i == 0 or mask[i - 1] != 0):
            mask.append(2)
        if r > tol and mask[i - 1] == 0:
            mask.append(1)
    return mask


def cropping(freqs):
    mask = masking(freqs)
    new = []
    for i in range(len(mask)):
        if mask[i] == 0:  # append average
            new.append((freqs[i] + freqs[i + 1]) / 2)
        if mask[i] == 1:
            if i == len(mask) - 1:
                new.append(freqs[i + 1])
        if mask[i] == 2:  # append itself
            new.append(freqs[i])
            if i == len(mask) - 1:
                new.append(freqs[i + 1])
    return new


def obtain(mic, flight_number):
    if mic == 12:
        main_file = f"newdata/Drone{flight_number}_Flight1/Array_D{flight_number}F1.csv"
    if mic == 16:
        main_file = f"data/Drone{flight_number}_Flight1/Array_D{flight_number}F1.csv"
    main_data = read_csv(main_file)
    main_data = preprocess_data(main_data)

    f, t, Sxx = signal.spectrogram(main_data, fs=50000, nperseg=n_perseg, nfft=int(n_perseg * 16),
                                   noverlap=int(n_perseg * 0.8))
    f = f[:n_frequencies]
    Sxx_legacy = Sxx

    Sxx = 10 * np.log10(Sxx_legacy)
    Sxx[Sxx < -125] = -125
    Sxx = Sxx[:n_frequencies, :]

    return f, t, Sxx_legacy, Sxx


def denoise(file, k):
    threshold = np.std(file) * np.sqrt(2 * np.log(len(file)))
    coeffs = pywt.wavedec(file, 'db4', mode='per')
    coeffs[1:] = (pywt.threshold(i, value=k * threshold, mode='soft') for i in coeffs[1:])
    return pywt.waverec(coeffs, 'db4', mode='per')


def pca_data(file, n=1):
    pca = PCA(n)
    red = np.array(pca.fit_transform(file.T)[:, 0])
    weights = pca.components_.reshape(-1)
    weighter = np.sum(weights)
    variance = pca.explained_variance_ratio_
    red /= weighter
    return red, weights, variance


def postprocessing(file):
    min = abs(np.min(file))
    file += min
    red = 10 * np.log10(file)  # comment if [Pa]
    preinf = np.min(red[np.isfinite(red)])  # comment if [Pa]
    red[np.isneginf(red)] = preinf  # comment if [Pa]
    maxpoint = np.where(red == np.amax(red))
    timeof = t[maxpoint]
    return red, maxpoint, timeof


def main_freqs(f, weights, target_freqs, probe=200):
    top = np.argsort(-weights.T)[:probe]
    fund_w = [weights[i] for i in top]
    fund_f = np.asarray([f[i] for i in top]).reshape(-1, 1)
    kmeans = KMeans(n_clusters=int(target_freqs))
    kmeans.fit(fund_f)
    a = np.sort(np.around(np.asarray(kmeans.cluster_centers_.T[0]), 1))
    return a, fund_f, fund_w


def reconcile(f1, f2, target_freqs):
    f_both = np.concatenate((f1, f2)).reshape(-1, 1)
    kmeans = KMeans(n_clusters=int(target_freqs))
    kmeans.fit(f_both)
    a = np.sort(np.around(np.asarray(kmeans.cluster_centers_.T[0]), 1))
    return a


"""
Read, Produce & Truncate Data
"""

s = 150  # sampling constant

for flight_number in range(1, 6):
    ### ---- MICROPHONE 12 ---- ###

    f, t, Sxx_legacy, Sxx = obtain(12, flight_number)

    red, weights, variance = pca_data(Sxx_legacy)

    red, maxpoint, timeof = postprocessing(red)

    a, fund_f, fund_w = main_freqs(f, weights, target_freqs, s)

    mic12_freqs = a
    mic12_eigenloudness = red
    mic12_specgram = Sxx
    mic12_t = t
    mic12_f = f
    mic12_close = timeof
    mic12_var = variance

    ### ---- MICROPHONE 16 ---- ###

    f, t, Sxx_legacy, Sxx = obtain(16, flight_number)

    red, weights, variance = pca_data(Sxx_legacy)

    red, maxpoint, timeof = postprocessing(red)

    a, fund_f, fund_w = main_freqs(f, weights, target_freqs, s)

    mic16_freqs = a
    mic16_eigenloudness = red
    mic16_specgram = Sxx
    mic16_t = t
    mic16_f = f
    mic16_close = timeof
    mic16_var = variance

    f_rec = reconcile(mic16_freqs, mic12_freqs, target_freqs)
    print(f"Drone {flight_number}\n Mic 12: {mic12_freqs}\n Mic 16: {mic16_freqs}\n Comp: {f_rec}")

    """
    Plot Graphs
    """

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 2]})

    fig.suptitle(f"Drone {flight_number} Eigenloudness vs Time")

    ax[0].minorticks_on()
    ax[0].plot(mic16_t, mic16_eigenloudness, color="blue", linewidth=0.5)
    ax[0].plot(mic12_t, mic12_eigenloudness, color="red", linewidth=0.5)
    ax[0].set_ylabel('Eigenloudness [dB]', fontsize=10)
    ax[0].set_xlabel('Time [sec]', fontsize=10)
    ax[0].vlines(mic16_close, np.min(mic16_eigenloudness) * 1.05, np.max(mic16_eigenloudness) * 0.95,
                 label="Mic 16: $t_{ca}$=" + f"{round(float(mic16_close), 1)} s", colors="blue", linewidth=0.5)
    ax[0].vlines(mic12_close, np.min(mic16_eigenloudness) * 1.05, np.max(mic16_eigenloudness) * 0.95,
                 label="Mic 12: $t_{ca}$=" + f"{round(float(mic12_close), 1)} s", colors="red", linewidth=0.5)
    ax[0].set_ylim(np.min(mic16_eigenloudness) * 1.05, np.max(mic16_eigenloudness) * 0.95)
    ax[0].set_xlim(t[0], t[-1])
    ax[0].legend(fontsize="8")
    ax[0].grid(linestyle='-', which='major', linewidth=0.9)
    ax[0].grid(linestyle=':', which='minor', linewidth=0.5)

    var1 = np.asarray([round(float(mic16_var) * 100, 1)])
    var2 = np.asarray([round(float(mic12_var) * 100, 1)])
    var3 = np.asarray(["-"])
    data1 = np.concatenate((mic16_freqs, var1))
    data2 = np.concatenate((mic12_freqs, var2))
    data3 = np.concatenate((f_rec, var3))
    table_data = [data1, data2, data3]
    row_labels = ["Mic 16", "Mic 12", "Cluster"]
    col_labels = [f"$f_{i + 1}$ [Hz]" for i in range(len(mic12_freqs))] + ["Var [%]"]
    ax[1].axis("off")
    table = ax[1].table(cellText=table_data,
                        cellLoc='center',
                        rowLabels=row_labels,
                        colLabels=col_labels,
                        loc="center")
    table.auto_set_font_size(True)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax[1].set_title(f"Drone {flight_number} Characteristic Frequency Analysis", y=0.9)

    plt.subplots_adjust(hspace=0.1)
    plt.tight_layout()
    plt.savefig(fname=f"Drone {flight_number} PCA + K-Means Combined (s={s})", dpi=900)

    plt.show()
