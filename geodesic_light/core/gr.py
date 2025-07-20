import numpy as np
from numba import njit


def initial_conditions(
    alpha: float,
    beta: float,
    r0: float,
    theta0: float,
    phi0: float,
    a: float,
) -> dict[str, np.ndarray | float]:
    """
    Compute initial conditions for null geodesic ray-tracing in Kerr spacetime

    Args:
        alpha: Image plane x-coordinate
        beta: Image plane y-coordinate
        r0: Observer radial coordinate
        theta0: Observer inclination
        phi0: Observer azimuth
        a: Black hole spin parameter

    Returns:
        Dictionary with:
            - y0: Initial state vector [r, theta, phi, t, p_r, p_theta]
            - E: Conserved energy
            - L: Conserved angular momentum
            - kappa: Carter constant
    """

    # Avoid singularities at small angles
    if abs(theta0) < 1e-8:
        theta0 = np.sign(theta0) * 1e-8
    if abs(phi0) < 1e-8:
        phi0 = np.sign(phi0) * 1e-8

    a2 = a**2
    r2 = r0**2
    sin_theta0 = np.sin(theta0)
    cos_theta0 = np.cos(theta0)
    sin_phi0 = np.sin(phi0)
    cos_phi0 = np.cos(phi0)

    # Transformation from observer to BH frame
    D = np.sqrt(r2 + a2) * sin_theta0 - beta * cos_theta0

    x = D * cos_phi0 - alpha * sin_phi0
    y = D * sin_phi0 + alpha * cos_phi0
    z = r0 * cos_theta0 + beta * sin_theta0

    w = x**2 + y**2 + z**2 - a2

    # Initial 6D-vector (r, theta, phi, t, p_r, p_theta)
    y0 = np.zeros(6)

    # Convert from Cartesian to Boyer-Lindquist coordinates
    y0[0] = np.sqrt((w + np.sqrt(w * w + (2.0 * a * z) * (2.0 * a * z))) / 2.0)
    y0[1] = np.arccos(z / y0[0])
    y0[2] = np.arctan2(y, x)
    y0[3] = 0

    r = y0[0]
    theta = y0[1]
    phi = y0[2]
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    sigma = r**2 + (a * cos_theta) ** 2
    R = np.sqrt(a2 + r**2)
    v = -sin_theta0 * np.cos(phi)  # reflect velocity: inverse ray-tracing
    zdot = -1.0

    # Initial velocities in Boyer-Lindquist coordinates
    rdot0 = zdot * (-R * R * cos_theta0 * cos_theta + r * R * v * sin_theta) / sigma
    thetadot0 = zdot * (cos_theta0 * r * sin_theta + R * v * cos_theta) / sigma
    phidot0 = zdot * sin_theta0 * np.sin(phi) / (R * sin_theta)

    r2 = r**2
    delta = r2 - 2.0 * r + a2
    s1 = sigma - 2.0 * r

    # Conserved energy
    E = np.sqrt(
        s1 * (rdot0**2 / delta + thetadot0**2) + delta * (sin_theta**2) * phidot0**2
    )

    # Conserved 4-momentum (rescaled by energy)
    y0[4] = (rdot0 * sigma / delta) / E
    y0[5] = (thetadot0 * sigma) / E

    # Angular momentum
    L = ((sigma * delta * phidot0 - 2.0 * a * r * E) * (sin_theta**2) / s1) / E

    # kappa
    kappa = y0[5] ** 2 + a2 * (sin_theta**2) + L**2 / (sin_theta**2)

    return {"y0": y0, "E": E, "L": L, "kappa": kappa}


@njit()
def ode_ray(
    t: float,
    y: np.ndarray,
    a: float,
    kappa: float,
    E: float,
    L: float,
) -> np.ndarray:
    """
    Right-hand side of null geodesic equations in Kerr spacetime (affine parametrization).

    Args:
        t: Time or affine parameter
        y: 6D state vector [r, theta, phi, t, p_r, p_theta]
        a: Black hole spin parameter
        kappa: Carter constant
        E: Conserved energy (for each ray)
        L: Conserved angular momentum (for each ray)

    Returns:
        Derivatives dy/dt as a 6D array
    """
    r, theta, p_r, p_th = y[0], y[1], y[4], y[5]
    r2, a2 = r**2, a**2
    sin_th, cos_th = np.sin(theta), np.cos(theta)
    sigma = r2 + a2 * (cos_th**2)
    delta = r2 - 2.0 * r + a2

    if np.abs(sin_th) < 1e-8:
        sin_th = np.sign(sin_th) * 1e-8

    # Set equation array for derivatives
    eqs = np.zeros(6)

    # Geodesic equations
    eqs[0] = -p_r * delta / sigma
    eqs[1] = -p_th / sigma
    eqs[2] = -(2.0 * r * a + (sigma - 2.0 * r) * L / (sin_th**2)) / (sigma * delta)
    eqs[3] = -(1.0 + (2.0 * r * (r2 + a2) - 2.0 * r * a * L) / (sigma * delta))
    eqs[4] = -(
        ((r - 1.0) * (-kappa) + 2.0 * r * (r2 + a2) - 2.0 * a * L) / (sigma * delta)
        - 2.0 * (p_r**2) * (r - 1.0) / sigma
    )
    eqs[5] = -sin_th * cos_th * (L**2 / sin_th**4 - a2) / sigma

    return eqs
