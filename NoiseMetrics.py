import numpy as np

def ospl(pressures):

    p_rms = np.square(np.sum(mp.square(pressures[:, 0]))) #