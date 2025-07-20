import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Callable
from geodesic_light.core.gr import ode_ray, initial_conditions


def _create_horizon_crossing_event(M: float, a: float) -> Callable:
    """
    Create an event function to stop integration when ray crosses the event horizon

    Args:
        M: Black hole mass
        a: Black hole spin parameter

    Returns:
        Event function for ODE solver
    """

    def horizon_event(t: float, y: np.ndarray) -> float:
        """Event function detecting horizon crossing"""
        r_EH = M + np.sqrt(M**2 - a**2)
        return y[0] - abs(r_EH + 1e-2)  # Small offset to avoid numerical issues

    horizon_event.terminal = True
    return horizon_event


def _create_escape_event(max_radius: float = 15.0) -> Callable:
    """
    Create an event function to stop integration when ray escapes beyond specified radius

    Args:
        max_radius: Maximum radius before stopping integration

    Returns:
        Event function for ODE solver
    """

    def escape_event(t: float, y: np.ndarray) -> float:
        """Event function detecting ray escape"""
        return max_radius - np.abs(y[0])

    escape_event.terminal = True
    return escape_event


def _compute_cartesian_coordinates(
    solver: solve_ivp,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert spherical to Cartesian coordinates

    Args:
        solver: ODE solution object

    Returns:
        Tuple of (x, y, z) coordinate arrays
    """
    r, θ, ϕ = solver.y[0, :], solver.y[1, :], solver.y[2, :]
    x = r * np.cos(ϕ) * np.sin(θ)
    y = r * np.sin(ϕ) * np.sin(θ)
    z = r * np.cos(θ)
    return x, y, z


def _trace_single_ray(
    M: float,
    a: float,
    r0: float,
    alpha: float,
    beta: float,
    theta0: float,
    phi0: float,
    tmin: float = 0.0,
    tmax: float = 100.0,
    max_radius: float = 15.0,
) -> dict:
    """
    Trace a single light ray in Kerr spacetime.

    Args:
        M: Black hole mass
        a: Black hole spin parameter
        r0: Initial radial coordinate
        alpha: Image plane x-coordinate
        beta: Image plane y-coordinate
        theta0: Initial polar angle
        phi0: Initial azimuthal angle
        tmin: Minimum integration time, default is 0.0
        tmax: Maximum integration time, default is 100.0
        max_radius: Maximum radius before stopping integration, default is 15.0

    Returns:
        Dictionary containing solver results and Cartesian coordinates
    """
    # Set initial conditions
    init_cond = initial_conditions(alpha, beta, r0, theta0, phi0, a)
    kappa, E, L = init_cond["kappa"], init_cond["E"], init_cond["L"]

    # Set up event detection
    horizon_event = _create_horizon_crossing_event(M, a)
    escape_event = _create_escape_event(max_radius)

    # Solve geodesic equations
    solver = solve_ivp(
        fun=lambda t, y: ode_ray(t, y, a, kappa, E, L),
        t_span=[tmin, tmax],
        y0=init_cond["y0"],
        method="Radau",
        events=(horizon_event, escape_event),
        dense_output=True,
        atol=1e-8,
        rtol=1e-6,
    )

    # Convert to Cartesian coordinates
    x, y, z = _compute_cartesian_coordinates(solver)

    return {"solver": solver, "x": x, "y": y, "z": z, "alpha": alpha, "beta": beta}


def raytracing(
    M: float,
    a: float,
    r0: float,
    theta0: float,
    phi0: float,
    alphas: np.ndarray,
    betas: np.ndarray,
    tmin: float = 0.0,
    tmax: float = 100.0,
    max_radius: float = 15.0,
) -> list[dict]:
    """
    Trace multiple light rays from an image plane grid

    Args:
        M: Black hole mass
        a: Black hole spin parameter
        r0: Initial radial coordinate
        theta0: Initial polar angle
        phi0: Initial azimuthal angle
        alphas: Array of image plane x-coordinates
        betas: Array of image plane y-coordinates
        tmin: Minimum integration time
        tmax: Maximum integration time
        max_radius: Maximum radius before stopping integration

    Returns:
        List of ray tracing results
    """
    return [
        _trace_single_ray(M, a, r0, alpha, beta, theta0, phi0, tmin, tmax, max_radius)
        for alpha in alphas
        for beta in betas
    ]
