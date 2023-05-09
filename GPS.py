import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import signal

from gps_coords import haversine

time_difference = [-1, 61451, -636, 84195, 28821, 17895]

start_time = [-1, 45000, 60000, 35000, 60000, 70000]

expected_closest_point_time = [-1, 14900, 8400, 8500, 6250, 6000]
expected_closest_point_time_12 = [-1, 14900, 8300, 8250, 5750, 3000]
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

    x = x[1:] - (-0.1538)
    print(x)
    y = y[1:] - (0.4457)
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
    dist = np.sqrt(np.square(sub_x) + np.square(sub_y))

    # finding the minimum distance
    minimum = np.nanargmin(dist[:1500])
    dist[0] = dist[1]

    sub_x = x - (0.0279)
    sub_y = y - (1.6998)
    dist_12 = np.sqrt(np.square(sub_x) + np.square(sub_y))


    minimum_12 = np.nanargmin(dist_12[:1500])
    dist_12[0] = dist_12[1]
    #min1 = dist[np.argsort(dist)[0]]
    #min2 = dist[np.argsort(dist)[1]]

    ### NEW CODEEE ###
    print("Relative minimum point")
    points = signal.argrelmin(dist, order=100)[0]
    values = dist[points]

    # determine both peaks in the graphs
    min1, min2 = time[points[np.argpartition(values, 1)[:2]]]
    min1 -= time_difference[flightnum]
    min2 -= time_difference[flightnum]

    # choose the one that matches data
    closestmin = min1 if abs(min1 - start_time[flightnum] + 7500) < abs(min2 - start_time[flightnum] + 7500) else min2

    ### MICROPHONE 12 ###
    print("Relative minimum point")
    points = signal.argrelmin(dist_12, order=100)[0]
    values_12 = dist_12[points]

    '''
    # determine both peaks in the graphs
    min1, min2 = time[points[np.argpartition(values, 1)[:2]]]
    min1 -= time_difference[flightnum]
    min2 -= time_difference[flightnum]

    # choose the one that matches data
    closestmin_12 = min1 if abs(min1 - start_time[flightnum] + 7500) < abs(min2 - start_time[flightnum] + 7500) else min2
    '''
    # check values
    print ("Drone", flightnum, closestmin - expected_closest_point_time[flightnum], closestmin )


    min_time =time[minimum]

    fig, ax = plt.subplots(1)
    ax.minorticks_on()
    #plt.figure()
    ax.axvline(x=closestmin, color='g', label='GPS closest approach')

    clock_difference = np.abs(
        min_time - time_difference[flightnum] - start_time[flightnum] - expected_closest_point_time[flightnum])
    '''
    # CLOCK DIFFERENCE ESTIMATE
    clock_difference = np.abs(min_time - time_difference[flightnum] - start_time[flightnum] - expected_closest_point_time[flightnum])
    clock_difference_12 = np.abs(
        closestmin_12 - time_difference[flightnum] - start_time[flightnum] - expected_closest_point_time_12[flightnum])
    avg_clock_difference = np.average(clock_difference, clock_difference_12)
    '''

    ax.plot(data[:, 0] - time_difference[flightnum],np.transpose(dist), linewidth=0.8)
    ax.plot(data[:, 0] - time_difference[flightnum], np.transpose(dist_12), linewidth=0.8)
    ax.axvline(x=start_time[flightnum], color='black')
    ax.axvline(x=start_time[flightnum] + 15000, color='black')
    ax.axvspan(start_time[flightnum], start_time[flightnum] + 15000, facecolor="none", hatch="//", edgecolor="b", alpha=0.5, label = 'interval', linewidth=0.3)
    ax.axvline(x=start_time[flightnum] + expected_closest_point_time[flightnum], color ='orange', label = 'Theo 16', linewidth=1)
    ax.axvline(x=start_time[flightnum] + expected_closest_point_time_12[flightnum], color='red', label='Theo 12', linewidth=1)
    # plt.axvline(min_time - time_difference[flightnum], color ='y', label = 'PASSBY')


    #Scatter point
    ax.scatter (closestmin, dist[int((closestmin + time_difference[flightnum])/100)], marker="*", color='green')

    ax.grid(True)
    ax.set_xlabel("GPS time (ms)")
    ax.set_ylabel("Geometrical distance from microphone (m)")

    ax.set_xlim(40000, expected_closest_point_time[flightnum] + 80000)

    ax.set_title("Drone" + str(flightnum) + " - " + str(clock_difference) + "(clock difference milisec)")
    ax.legend()
    plt.show()


for i in range(1, 6):
    closest_point(i)
    # drone_speed(i)



