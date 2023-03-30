import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from gps_coords import haversine

time_difference = [-1, 61451, -636, 84195, 28821, 17895]
limit_down = [-1, 300, ] #where data from the excel file starts to matter
limit_up = [-1, 650,  ] #where data from the excel file is not relevant anymore

start_time = [-1, 45000, 60000, 35000, 60000, 70000]

expected_closest_point_time = [-1, 14800, 7500, 7200, 9000, 9000]
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

    broken = ([0, 40, 48, 2, 34, 59, 5, 45, 46, 55, 63])
    plt.figure()

    plt.subplot(111).minorticks_on()
    plt.scatter(x, y)
    plt.scatter(x[broken], y[broken], c="red")
    plt.scatter(x[57], y[57], c="green")
    plt.ylabel("y")
    plt.xlabel("x")
    plt.grid(linestyle='-', which='major', linewidth=0.9)
    plt.grid(linestyle=':', which='minor',linewidth = 0.5)
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

    x, y = converted[:, 1], converted[:, 2] #

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


def drone_speed(flightnum):

    x, y = convert_coords("data/Drone{0}_Flight1/GPS_D{0}F1.csv".format(flightnum))
    dp = pd.read_csv("data/Drone{0}_Flight1/GPS_D{0}F1.csv".format(flightnum))
    time = dp.iloc[:, 0].values

    x = x[1:] - (-0.0279)
    print(x)
    y = y[1:] - (-1.6998)
    #print(time)
    speed = []

    if len(x) == len(time):

        for i in range(1, len(time)):
            distx = (x[i] - x[i-1])
            disty = (y[i] - y[i-1])
            dist_stamp = np.sqrt(distx **2 + disty **2)
            #print()
            #print(dist_stamp)
            #print(time[i], time[i-1])
            time_stamp = time[(i)] - time[i-1]
            #print(time_stamp)
            time_stamp = time_stamp*0.001  # convert to s
            #print(time_stamp)
            speed_stamp = dist_stamp / time_stamp

            speed.append(speed_stamp)
    else:
        print("tf bro")
    #print(speed)
    speed = np.asarray(speed)
    print("SPEED ", flightnum)
    print(np.mean(speed[100:-100]))
    #print(np.sort(speed)[-200:])
    #print(max(speed))
    #print(speed_stamp)
    return speed







# plot_one_flight(1)
# plot_all_flights()
# plot_mic_array_corrected()
# animate_flight("data/Drone1_Flight1/GPS_D1F1.csv")

def closest_point(flightnum):


    x, y = convert_coords("data/Drone{0}_Flight1/GPS_D{0}F1.csv".format(flightnum))
    data = np.genfromtxt("data/Drone{0}_Flight1/GPS_D{0}F1.csv".format(flightnum),delimiter=",")

    dp = pd.read_csv("data/Drone{0}_Flight1/GPS_D{0}F1.csv".format(flightnum))
    time = dp.iloc[:, 0]

    #data = np.delete(data, 0, axis=0)
    #x = np.delete(x, 0, axis=0)
    #y = np.delete(y, 0, axis=0)

    sub_x = x - (-0.0279)
    sub_y = y - (-1.6998)
    #sub_x = np.isnan(sub_x)
    #sub_y = np.isnan(sub_y)
    dist = np.sqrt(np.square(sub_x) + np.square(sub_y))
    print(dist)
    minimum = np.nanargmin(dist)
    print(minimum)
    min_time =time[minimum]
    print("min time")
    print(min_time)
    plt.figure()
    print((dist))
    print((time))

    clock_difference = np.abs(min_time - time_difference[flightnum] - start_time[flightnum] - expected_closest_point_time[flightnum])
    plt.plot(data[:, 0] - time_difference[flightnum],np.transpose(dist))
    plt.axvline(x=start_time[flightnum], color='b', label='Start')
    plt.axvline(x=start_time[flightnum] + 15000, color='r', label='End')
    plt.axvline(x=start_time[flightnum] + expected_closest_point_time[flightnum], color ='black', label = 'Theo')
    plt.axvline(min_time - time_difference[flightnum], color ='y', label = 'PASSBY')

    plt.xlabel("GPS time (ms)")
    plt.ylabel("Geometrical distance from microphone (m)")

    plt.title("Drone" + str(flightnum) + " - " + str(clock_difference) + "(clock difference milisec)")
    plt.legend()
    plt.show()
    print(min(dist))


for i in range(1, 6):
    closest_point(i)
    drone_speed(i)



