import numpy as np
import csv
import matplotlib.pyplot as plt

with open("data/Array_D1F1.csv", 'r') as f:
    data = list(csv.reader(f, delimiter=","))

data = np.array(data, float)
data = data[0,:]
data = data[:len(data)-1]

t = np.arange(0,15,1/50000)

plt.plot(t,data,"b")
plt.show()

print(len(data))