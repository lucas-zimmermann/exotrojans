import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk


def plot_stacked_tess_lightcurves(
    csv_path: str = "nexsci.csv",
    num: int = 10,
    *,
    skiprows: int = 69,
    bins: int = 10_000,
    xlim: tuple[float, float] = (-100, 100),
    ylim: tuple[float, float] = (0.995, 1.005),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Read a NExScI-style CSV, grab the first `num` valid TESS targets,
    fold and bin their lightcurves, and scatter-plot them on a shared axis.

    Returns the matplotlib Axes object.
    """
    # --- Load and clean parameter table ---
    df = pd.read_csv(csv_path, skiprows=skiprows)
    tics = np.array(df["tid"])
    pers = np.array(df["pl_orbper"])
    t0s = np.array(df["pl_tranmid"])

    mask = (np.isnan(pers)) | (np.isnan(t0s))
    tics = tics[~mask]
    pers = pers[~mask]
    t0s_bjd = t0s[~mask]
    t0s = t0s_bjd - 2457000.0  # convert BJD to BTJD for TESS

    # --- Set up figure/axes ---
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    # --- Loop over targets and plot ---
    for i in range(min(num, len(tics))):
        tic = tics[i]
        per = pers[i]
        t0 = t0s[i]

        sr = lk.search_lightcurve(f"TIC {int(tic)}", author="TESS-SPOC")
        if len(sr) == 0:
            continue

        lc = sr[0].download()
        if lc is None:
            continue

        folded = (
            lc.normalize()
              .fold(period=per, epoch_time=t0, wrap_phase=0.5)
              .remove_outliers()
        )
        binned = folded.bin(bins=bins)

        # Convert phase to degrees for plotting
        phase_deg = (binned.phase.value / per) * 360.0
        ax.scatter(phase_deg, binned.flux.value, s=1)

    # --- Final plot cosmetics ---
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axvline(60, color="gray", linestyle="--")
    ax.axvline(-60, color="gray", linestyle="--")

    ax.set_xlabel("Phase (°)")
    ax.set_ylabel("Normalized Flux")
    ax.set_title(f"Stacked Light Curves of {num} Exoplanets")

    fig.tight_layout()
    return ax