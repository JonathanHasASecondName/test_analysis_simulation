import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def plot_noise_mics():

    for i in range(1, 6):
        dp = pd.read_csv("data/Array_D" + str(i) + "F1.csv")
        print(dp)

        x = np.linspace(0,15000, len(dp)+1)

        plt.plot(x, dp)
        plt.title("Drone" + str(i))
        plt.show()
    return



#plot_height_drone()
plot_noise_mics()

