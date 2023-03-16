import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gps_coords import haversine

def plot_mic_array():
    darray = np.loadtxt("data/config.txt")

    x = -1*darray[:, -2]
    y = darray[:, -1]

    plt.scatter(x,y)
    plt.ylabel("y")
    plt.xlabel("x")
    plt.show()
    return

def plot_mic_array_corrected():
    darray = np.loadtxt("data/config.txt")

    x = -1*darray[:, -2]
    y = darray[:, -1]
    plt.scatter(x,y)
    plt.scatter(x[0, 5, 6, 16, 20, 31, 40, 45, 53, 62, 63], y[0, 5, 6, 16, 20, 31, 40, 45, 53, 62, 63], "red")
    plt.ylabel("y")
    plt.xlabel("x")
    plt.show()
    return

def plot_height_drone():
    dp = pd.read_csv("data/GPS_D1F1.csv")

    y = dp.iloc[:, 3]
    x = dp.iloc[:, 0]

    plt.plot(x,y)
    plt.xlabel("Time(ms)")
    plt.ylabel("Drone altitude(m)")
    plt.show()
    return




plot_mic_array()