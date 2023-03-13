import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gps_coords import haversine

def plot_mic_array():
    darray = np.loadtxt("data/config.txt")

    x = darray[:, -2]
    y = darray[:, -1]

    plt.scatter(x,y)
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

plot_height_drone()

