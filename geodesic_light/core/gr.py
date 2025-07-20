import numpy as np
from numba import njit
from scipy.integrate import odeint, solve_ivp


def initial_conditions(alpha, beta, r0, theta0, phi0, a):

    # alpha  =  x-coordinate in the image plane
    # beta   =  y-coordinate in the image plane
    # r0     =  distance to the observer
    # theta0 =  inclination of the observer
    # phi0   =  azimuthal angle of the observer
    # a      =  spin of the black hole

    # correct for small angles in order to avoid zeros
    if abs(theta0) < 1e-8:
        theta0 = np.sign(theta0) * 1e-8
    if abs(phi0) < 1e-8:
        phi0 = np.sign(phi0) * 1e-8

    # some shorthands
    a2 = a**2
    r2 = r0**2
    sin_theta0 = np.sin(theta0)
    cos_theta0 = np.cos(theta0)
    sin_phi0 = np.sin(phi0)
    cos_phi0 = np.cos(phi0)

    # transformation from observer (with gamma=0) to BH frame
    D = np.sqrt(r2 + a2) * sin_theta0 - beta * cos_theta0

    x = D * cos_phi0 - alpha * sin_phi0
    y = D * sin_phi0 + alpha * cos_phi0
    z = r0 * cos_theta0 + beta * sin_theta0

    w = x**2 + y**2 + z**2 - a2

    # initial 6D-vector (r, theta, phi, t, p_r, p_theta)
    y0 = np.zeros(6)

    # convert cartesian to Boyer-Lindquist coords
    y0[0] = np.sqrt((w + np.sqrt(w * w + (2.0 * a * z) * (2.0 * a * z))) / 2.0)
    y0[1] = np.arccos(z / y0[0])
    y0[2] = np.arctan2(y, x)
    y0[3] = 0

    ########
    # calculate initial velocities
    ########

    r = y0[0]
    theta = y0[1]
    phi = y0[2]
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # auxiliary variables
    sigma = r**2 + (a * cos_theta) ** 2
    R = np.sqrt(a2 + r**2)
    v = -sin_theta0 * np.cos(phi)  # reflect velocity: inverse ray-tracing
    zdot = -1.0

    # initial velocities in Boyer-Lindquist coords
    rdot0 = zdot * (-R * R * cos_theta0 * cos_theta + r * R * v * sin_theta) / sigma
    thetadot0 = zdot * (cos_theta0 * r * sin_theta + R * v * cos_theta) / sigma
    phidot0 = zdot * sin_theta0 * np.sin(phi) / (R * sin_theta)

    # additional variables
    r2 = r**2
    delta = r2 - 2.0 * r + a2
    s1 = sigma - 2.0 * r

    # conserved energy (see later on)
    E = np.sqrt(
        s1 * (rdot0**2 / delta + thetadot0**2) + delta * (sin_theta**2) * phidot0**2
    )

    # conserved 4-momentum (rescaled by energy; see later on)
    y0[4] = (rdot0 * sigma / delta) / E
    y0[5] = (thetadot0 * sigma) / E

    # compute angular momentum (see later on)
    L = ((sigma * delta * phidot0 - 2.0 * a * r * E) * (sin_theta**2) / s1) / E

    # compute kappa (see later on)
    kappa = y0[5] ** 2 + a2 * (sin_theta**2) + L**2 / (sin_theta**2)

    # return initial-condition vector, energy, angular momentum and kappa as a dictonary
    return {"y0": y0, "E": E, "L": L, "kappa": kappa}


@njit()
def ode_ray(t, y, a, kappa, E, L):
    # t      =  t or affine parameter
    # y      =  6D state vector (r, theta, phi, t, p_r, p_theta)
    # a      =  black hole spin
    # kappa  =  kappa
    # E      =  conserved energy for each ray
    # L      =  conserved angular momentum for each ray

    # get variables to make programming clearer to read
    r = y[0]
    theta = y[1]
    p_r = y[4]
    p_th = y[5]

    # auxiliary variables
    r2, a2 = r**2, a**2
    sin_th, cos_th = np.sin(theta), np.cos(theta)
    sigma = r2 + a2 * (cos_th**2)
    delta = r2 - 2.0 * r + a2

    # avoid small numbers
    if np.abs(sin_th) < 1e-8:
        sin_th = np.sign(sin_th) * 1e-8

    # set equation array for derivatives
    eqs = np.zeros(6)

    # geodesic equations
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
