import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from gps_coords import haversine

time_difference = [-1, 106451, 59364, 139195, 88821, 87895]

def plot_mic_array():

    darray = np.loadtxt("data/config.txt")

    x = -1 * darray[:, -2]
    y = darray[:, -1]

    plt.scatter(x, y)
    plt.ylabel("y")
    plt.xlabel("x")

    plt.show()
    return

def plot_mic_array_corrected():
    darray = np.loadtxt("data/config.txt")

    x = -1 * darray[:, -2]
    y = darray[:, -1]
    plt.scatter(x, y)
    broken = ([0, 40, 48, 2, 34, 59, 5, 45, 46, 55, 63])
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

    plt.plot(x, y)
    plt.xlabel("Time(ms)")
    plt.ylabel("Drone altitude(m)")
    plt.show()
    return

def convert_coords(filename):
    data = np.genfromtxt(filename, delimiter=',')
    converted = np.copy(data[:, :3])
    for index, i in enumerate(converted[:, 1:3]):
        converted[index, 1:3] = haversine(i)

    x, y = converted[:, 1], converted[:, 2]

    return x, y

def update(num, x, y, line):
    line.set_data(x[:num], y[:num])
    # line.axes.axis([0, 10, 0, 1])
    return line,

def plot_one_flight(flightnum):
    x, y = convert_coords("data/GPS_D{}F1.csv".format(flightnum))
    plt.plot(x, y, "r")
    plt.show()

def plot_all_flights():
    """
    BAD FUNCTION
    only works with specific filenames and directory structure
    but it gets the job done ¯\_(ツ)_/¯
    """

    fig, axs = plt.subplots(3, 2)
    for i in range(5):
        x, y = convert_coords("data/Drone{0}_Flight1/GPS_D{0}F1.csv".format(i + 1))
        index = i // 2, i % 2
        axs[index].plot(x, y)
        axs[index].set_xlim([-300, 300])
        axs[index].set_ylim([-300, 300])
    plt.show()


def animate_flight(filename):
    x, y = convert_coords(filename)

    fig, ax = plt.subplots()
    line, = ax.plot(x, y, color='k')
    ax.set_xlim([-300, 300])
    ax.set_ylim([-300, 300])

    ani = animation.FuncAnimation(fig, update, len(x), fargs=[x, y, line], interval=20,
                                  blit=True)  # Freely inspired from StackOverflow
    plt.show()


# plot_one_flight(1)
# plot_all_flights()
# plot_mic_array_corrected()
# animate_flight("data/Drone1_Flight1/GPS_D1F1.csv")

def closest_point(flightnum):
    x, y = convert_coords("data/Drone{0}_Flight1/GPS_D{0}F1.csv".format(flightnum))
    data = np.genfromtxt("data/Drone{0}_Flight1/GPS_D{0}F1.csv".format(flightnum),delimiter=",")

    #data = np.delete(data, 0, axis=0)

    data[0,0]=0
    sub_x = x - (-0.0279)
    sub_y = y - (-1.6998)
    sub_x[0] = 0
    sub_y[0] = 0
    # sub_x = np.isnan(sub_x)
    # sub_y = np.isnan(sub_y)
    dist = np.sqrt(np.square(sub_x) + np.square(sub_y))
    plt.figure()
    print((dist))
    print((data))

    plt.plot(data[:, 0] - time_difference[flightnum],np.transpose(dist))
    plt.axvline(x=0, color='b', label='Start')
    plt.axvline(x=15000, color='r', label='End')
    plt.title("Drone" + str(flightnum))
    plt.show()
    print(min(dist))


for i in range(1, 6):
    closest_point(i)
