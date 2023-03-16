import numpy as np
import csv


with open('Array_D1F1.csv', 'r') as f:
    data = list(csv.reader(f, delimiter=","))

data = np.array(data)