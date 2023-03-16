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
    broken=([0,40,48,2,34,59,5,45,46,55,63])
    plt.scatter(x[broken], y[broken], c="red")
    plt.scatter(x[57], y[57], c="green")
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

def convert_coords(filename):
    data = np.genfromtxt(filename, delimiter=',')
    converted = np.copy(data[:, :3])
    for index, i in enumerate(converted[:, 1:3]):
        converted[index, 1:3] = haversine(i)

    x, y = converted[:,1], converted[:, 2]

    return x, y

def plot_all_flights():
    """ 
    BAD FUNCTION 
    only works with specific filenames and directory structure
    but it gets the job done ¯\_(ツ)_/¯
    """

    fig, axs = plt.subplots(3, 2)
    for i in range(5):
        x, y = convert_coords("data/GPS_D{}F1.csv".format(i+1))            
        axs[i // 2, i % 2].plot(x, y)
    plt.show()

plot_all_flights()
plot_mic_array_corrected()
