import numpy as np
import matplotlib.pyplot as plt
import batman


def simulate_trojan_swarm(
    P: float = 5.0,
    t0: float = 1325.0,
    a_rs: float = 12.0,
    inc: float = 88.5,
    u1: float = 0.3,
    u2: float = 0.2,
    rp_p: float = 0.08,
    exp_time: float = 600.0 / 86400.0,
    supersample: int = 9,
    # swarm controls
    N_swarm: int = 200,
    frac_L4: float = 0.5,
    sigma_deg: float = 8.0,
    rp_med: float = 0.0010,
    rp_scatter: float = 0.35,
    rp_min: float = 0.0002,
    # time window & resolution
    window_factor: float = 1.5,   # half-width in P (total span = 2 * window_factor * P)
    n_time: int = 40000,
    # plotting
    phase_xlim: tuple[float, float] = (-100.0, 100.0),
    ax: plt.Axes | None = None,
    seed: int | None = None,
):
    """
    Simulate a planet transit plus a swarm of Trojans near L4/L5 and plot
    the combined light curve versus orbital phase in degrees.

    Returns
    -------
    phase_deg : ndarray
        Orbital phase in degrees for the full time grid.
    flux : ndarray
        Combined (planet * Trojans) relative flux for the full time grid.
    ax : matplotlib.axes.Axes
        The Axes on which the zoomed [-100°, 100°] region is plotted.
    """
    if seed is not None:
        np.random.seed(seed)

    # --- Time grid ---
    t = np.linspace(t0 - window_factor * P,
                    t0 + window_factor * P,
                    n_time)

    # --- Helper for TransitParams ---
    def mk_params(t0_here, rp_here):
        p = batman.TransitParams()
        p.t0 = float(t0_here)
        p.per = float(P)
        p.rp = float(rp_here)
        p.a = float(a_rs)
        p.inc = float(inc)
        p.ecc = 0.0
        p.w = 90.0
        p.limb_dark = "quadratic"
        p.u = [u1, u2]
        return p

    # --- Planet only ---
    pp = mk_params(t0, rp_p)
    mp = batman.TransitModel(pp, t,
                             supersample_factor=supersample,
                             exp_time=exp_time)
    flux = mp.light_curve(pp)

    # --- Swarm setup ---
    n_L4 = int(round(N_swarm * frac_L4))
    n_L5 = N_swarm - n_L4

    def trojan_angles(n, center_deg, sigma_deg):
        return np.random.normal(loc=center_deg, scale=sigma_deg, size=n)

    def trojan_radii(n, rp_med, rp_scatter, rp_min):
        logsig = rp_scatter
        rp = rp_med * 10**(np.random.normal(0.0, logsig, size=n))
        return np.clip(rp, rp_min, None)

    angles_deg = np.concatenate([
        trojan_angles(n_L4, +60.0, sigma_deg),
        trojan_angles(n_L5, -60.0, sigma_deg),
    ])
    rps_T = trojan_radii(N_swarm, rp_med, rp_scatter, rp_min)

    # --- Map angles to transit times and multiply in Trojans ---
    dt_days = (angles_deg / 360.0) * P
    t0_Ts = t0 + dt_days

    for t0_T, rp_T in zip(t0_Ts, rps_T):
        pT = mk_params(t0_T, rp_T)
        mT = batman.TransitModel(pT, t,
                                 supersample_factor=supersample,
                                 exp_time=exp_time)
        flux *= mT.light_curve(pT)

    # --- Convert to phase and make a zoomed plot ---
    phase_cycles = (t - t0) / P
    phase_deg = phase_cycles * 360.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.4, 3.2))
    else:
        fig = ax.figure

    mask = (phase_deg >= phase_xlim[0]) & (phase_deg <= phase_xlim[1])
    ax.plot(phase_deg[mask], flux[mask], lw=1)
    ax.set_xlabel("Phase (degrees, 0 = planet mid-transit)")
    ax.set_ylabel("Relative flux")
    ax.set_xlim(*phase_xlim)
    ax.set_title(
        f"Planet + Trojan swarm near L4/L5 (N={N_swarm}, σ={sigma_deg}°)"
    )
    fig.tight_layout()

    return phase_deg, flux, ax