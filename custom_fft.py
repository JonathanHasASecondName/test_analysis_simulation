import numpy as np


def calculate_term(x: list[int], index: int):
    sum = 0
    for q in range(len(x)):
        sum+=x[q]*np.exp(-2j*np.pi*q*index/len(x))
    return sum

def dft(x):
    transform = []
    for i, sample in enumerate(x):
        transform.append(calculate_term(x, i))
    return transform

# sampling rate :)
sr = 1
# sampling interval
ts = 1.0/sr
t = np.arange(0,1,ts)

freq = 3
x = 3*np.sin(2*np.pi*freq*t)
x = [abs(z) for z in (dft(x))]
for i, value in enumerate(x):
    if value>0.01:
        print(i, value)
    else:
        print(i, 0)
