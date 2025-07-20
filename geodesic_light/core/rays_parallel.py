from functools import partial
from typing import Optional
from multiprocessing import Pool, cpu_count

import numpy as np

from geodesic_light.core.rays import _trace_single_ray


def _worker_wrapper(
    alpha: float,
    beta: float,
    M: float,
    a: float,
    r0: float,
    theta0: float,
    phi0: float,
    tmin: float,
    tmax: float,
    max_radius: float,
) -> dict:
    return _trace_single_ray(
        M, a, r0, alpha, beta, theta0, phi0, tmin, tmax, max_radius
    )


def raytracing_parallel(
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
    n_processes: Optional[int] = None,
    chunksize: Optional[int] = None,
) -> list[dict]:
    """
    Trace multiple light rays in parallel using multiprocessing

    For small tasks or when running with few cores, the overhead might outweigh benefits;
    recommended for large grids on local machines with many cores, otherwise use `raytracing`

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
        n_processes: Number of worker processes; default: min(cpu_count, num_tasks)
        chunksize: Number of tasks sent to each worker at a time; default is None, meaning
            the pool will decide based on the number of tasks and processes

    Returns:
        List of ray tracing results
    """
    # Convert inputs to numpy arrays and create task grid
    alphas = np.asarray(alphas)
    betas = np.asarray(betas)

    alpha_grid, beta_grid = np.meshgrid(alphas, betas)
    tasks = [
        (float(alpha), float(beta))
        for alpha, beta in zip(alpha_grid.ravel(), beta_grid.ravel())
    ]

    num_tasks = len(tasks)
    if n_processes is None:
        n_processes = min(cpu_count(), num_tasks)
    n_processes = max(1, min(n_processes, num_tasks))

    # Partialise the tasks for parallel processing
    worker_func = partial(
        _worker_wrapper,
        M=M,
        a=a,
        r0=r0,
        theta0=theta0,
        phi0=phi0,
        tmin=tmin,
        tmax=tmax,
        max_radius=max_radius,
    )

    with Pool(processes=n_processes) as pool:
        results = list(pool.starmap(worker_func, tasks, chunksize=chunksize))

    return results
