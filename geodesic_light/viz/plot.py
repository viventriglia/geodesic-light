import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from geodesic_light.viz.utils import photon_ring_radius, compute_alpha


def save_plot(fig: Figure, filename: str, dpi: int = 300) -> None:
    """
    Save the figure to a file

    Args:
        fig: Matplotlib figure object to save
        filename: Output filename
        dpi: Resolution in dots per inch, defaults to 300
    """
    fig.savefig(
        filename,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
        facecolor=fig.get_facecolor(),
    )


def _create_figure(
    xlim: tuple[float, float], ylim: tuple[float, float]
) -> tuple[Figure, Axes]:
    """Create and configure the figure and axes"""
    fig = plt.figure(figsize=(12, 12), facecolor="#020a17")
    ax = fig.add_subplot(111, facecolor="#020a17")
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def _plot_rays(ax: Axes, rays: list[dict], lw: float) -> None:
    """Plot individual rays with color based on their physical (winding) properties"""
    for ray in rays:
        phi = abs(ray["solver"].y[2, -1])
        n = phi / (2.0 * np.pi)
        jitter = np.random.normal(loc=0.0, scale=0.02)
        alpha = np.clip(compute_alpha(n) + jitter, 0, 1)
        norm_n = min(n, 2.0) / 2.0
        cmap = plt.cm.inferno
        rgb = cmap(norm_n)
        color = (*rgb[:3], alpha)
        ax.plot(ray["x"], ray["y"], marker="None", color=color, ls="-", lw=lw)


def _plot_black_hole(ax: Axes, M: float, a: float) -> None:
    """Plot black hole shadow and accretion glow"""
    r_EH = M + np.sqrt(M**2 - a**2)

    # Black Hole Shadow
    shadow = plt.Circle((0, 0), r_EH, color="k", zorder=10)
    ax.add_artist(shadow)

    # Accretion glow
    for r in np.linspace(r_EH, r_EH + 0.1, 12):
        glow = plt.Circle((0, 0), r, color=(1, 0.6, 0.1, 0.04), zorder=5)
        ax.add_artist(glow)


def _plot_photon_rings(ax: Axes, M: float, a: float) -> None:
    """Plot photon rings, if they exist for the given mass and spin parameters"""
    try:
        r_ph = photon_ring_radius(M, a)
        n_rings = 5
        for i in range(n_rings):
            alpha = 0.6 / (i + 1)
            lw = 0.1 + 0.3 * i
            ring = plt.Circle(
                (0, 0),
                r_ph,
                color=(1, 1, 1, alpha),
                lw=lw,
                fill=False,
                ls="-",
                zorder=15 + i,
            )
            ax.add_artist(ring)
    except ValueError:
        pass


def plot_ray_tracing(
    rays: list[dict],
    M: float,
    a: float,
    xlim: tuple[float, float] = (-10, 10),
    ylim: tuple[float, float] = (-10, 10),
    lw: float = 0.1,
) -> Figure:
    """
    Generate a visualization of light rays around a rotating black hole

    Args:
        rays: List of ray dictionaries containing solver and trajectory data
        M: Black hole mass
        a: Black hole spin parameter
        xlim: Tuple of (xmin, xmax) plot limits
        ylim: Tuple of (ymin, ymax) plot limits
        lw: Line width for ray trajectories

    Returns:
        Matplotlib figure object containing the visualization
    """
    fig, ax = _create_figure(xlim, ylim)
    _plot_rays(ax, rays, lw)
    _plot_black_hole(ax, M, a)
    _plot_photon_rings(ax, M, a)

    return fig
