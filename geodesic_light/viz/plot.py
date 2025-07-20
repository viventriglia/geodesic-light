import numpy as np
import matplotlib.pyplot as plt

from geodesic_light.viz.utils import photon_ring_radius, compute_alpha


def ray_analysis(rays, M, a, dim=[-10, 10, -10, 10, -10, 10], lw=0.1):

    # rays = list of rays
    # dim  = plot range

    fig = plt.figure(figsize=(12, 12), facecolor="#020a17")
    ax = fig.add_subplot(111, facecolor="#020a17")
    ax.set_aspect("equal")

    for i in rays:
        phi = abs(i["solver"].y[2, -1])
        n = phi / (2.0 * np.pi)
        jitter = np.random.normal(loc=0.0, scale=0.02)
        alpha = np.clip(compute_alpha(n) + jitter, 0, 1)
        norm_n = min(n, 2.0) / 2.0
        cmap = plt.cm.inferno  # inferno, plasma
        rgb = cmap(norm_n)
        color = (*rgb[:3], alpha)

        plt.plot(i["x"], i["y"], marker="None", color=color, ls="-", lw=lw)

    ax.set_xlim(dim[0], dim[1])
    ax.set_ylim(dim[2], dim[3])
    ax.axis("off")

    r_EH = M + np.sqrt(M**2 - a**2)

    # Black Hole Shadow
    shadow = plt.Circle((0, 0), r_EH, color="k", zorder=10)
    ax.add_artist(shadow)

    for r in np.linspace(r_EH, r_EH + 0.1, 12):
        glow = plt.Circle((0, 0), r, color=(1, 0.6, 0.1, 0.04), zorder=5)
        ax.add_artist(glow)

    # Photon Rings
    try:
        r_ph = photon_ring_radius(M, a)
        n_rings = 5
        for i in range(n_rings):
            alpha = 0.6 / (i + 1)
            lw_ = 0.1 + 0.3 * i
            ring = plt.Circle(
                (0, 0),
                r_ph,
                color=(1, 1, 1, alpha),
                lw=lw_,
                fill=False,
                ls="-",
                zorder=15 + i,
            )
            ax.add_artist(ring)
    except ValueError as e:
        pass

    # fig.savefig(
    #     f'spin_{a}_photons_{len(rays)}_lw_{lw}.png',
    #     dpi=600,
    #     bbox_inches='tight',
    #     pad_inches=0,
    #     facecolor=fig.get_facecolor(),
    #     )
    plt.show()
