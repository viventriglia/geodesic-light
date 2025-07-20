from typing import Union

import numpy as np


def compute_alpha(
    n: Union[float, np.ndarray], n_max: float = 2.0
) -> Union[float, np.ndarray]:
    """
    Compute the transparency alpha value based on the winding number n

    The function uses a sigmoid curve to smoothly transition between transparency values,
    with most of the transition happening around n/n_max ≈ 0.6

    Args:
        n: Winding number (can be scalar or array)
        n_max: Maximum normalization value (default 2.0)

    Returns:
        Alpha value(s) between 0.1 and 1.0
    """
    norm_n = n / n_max
    sigmoid = 1 / (1 + np.exp(-10 * (norm_n - 0.6)))
    return 0.1 + 0.9 * sigmoid


def photon_ring_radius(M: float, a: float) -> float:
    """
    Calculate the photon sphere radius for a Kerr black hole

    Args:
        M: Mass of the black hole (must be positive)
        a: Spin parameter of the black hole (-M ≤ a ≤ M)

    Returns:
        Radius of the photon sphere
    """
    return 2 * M * (1 + np.cos((2 / 3) * np.arccos(-a / M)))
