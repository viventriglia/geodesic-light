import numpy as np


def compute_alpha(n, n_max=2.0):
    norm_n = n / n_max
    return 0.1 + 0.9 / (1 + np.exp(-10 * (norm_n - 0.6)))


def photon_ring_radius(M, a):
    return 2 * M * (1 + np.cos((2 / 3) * np.arccos(-a / M)))
