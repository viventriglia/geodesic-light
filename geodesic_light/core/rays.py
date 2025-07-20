import numpy as np
from scipy.integrate import solve_ivp

from geodesic_light.core.gr import ode_ray, initial_conditions


def make_hit_r_EH(M, a):
    def hit_r_EH(t, y):
        # Stop rays at the horizon
        r_EH = M + np.sqrt(M**2 - a**2)
        return y[0] - abs(r_EH + 1e-2)

    hit_r_EH.terminal = True
    return hit_r_EH


def leavimg(t, y):

    # stop rays after some distance
    return 15 - np.abs(y[0])


def ray(M, a, r0, alpha, beta, theta0, phi0, tmin, tmax):

    init_cond = initial_conditions(alpha, beta, r0, theta0, phi0, a)
    kappa, E, L = init_cond["kappa"], init_cond["E"], init_cond["L"]

    # stop rays which hit the horizon
    hit_event = make_hit_r_EH(M, a)

    # stop rays which travelled to r greater than a certain value to be chosen
    leavimg.terminal = True

    # call ODE solver
    solver = solve_ivp(
        lambda t, y: ode_ray(t, y, a, kappa, E, L),
        t_span=[tmin, tmax],
        y0=init_cond["y0"],
        method="Radau",
        events=(hit_event, leavimg),
        dense_output=True,
        atol=1e-8,
        rtol=1e-6,
    )

    # compute x, y, z in cartesian coords
    x = solver.y[0, :] * np.cos(solver.y[2, :]) * np.sin(solver.y[1, :])
    y = solver.y[0, :] * np.sin(solver.y[2, :]) * np.sin(solver.y[1, :])
    z = solver.y[0, :] * np.cos(solver.y[1, :])

    return {"solver": solver, "x": x, "y": y, "z": z}


def multiray(M, a, r0, theta0, phi0, alphas, betas, tmin, tmax):

    # alphas = list of x-positions on the image plane
    # betas  = list of y-positions on the image plane

    result = []
    for alpha in alphas:
        for beta in betas:
            geodesic = ray(M, a, r0, alpha, beta, theta0, phi0, tmin, tmax)
            result.append(geodesic)

    return result
