import numpy as np


# source: https://stackoverflow.com/a/13849249
def unit_vector(vector: np.array):
    return vector / np.linalg.norm(vector) if np.count_nonzero(vector) else vector


def angle_between(v1: np.array, v2: np.array):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))