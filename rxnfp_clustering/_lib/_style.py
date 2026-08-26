"""Shared figure style — PROJECT RULE: all figure text is >= 18pt.

Lives in ``_lib/`` (shared code). Import at the top of any pipeline script,
after importing matplotlib — pipeline scripts live at ``pipeline/<stage>/x.py``,
so ``_lib`` is three directories up:

    import sys, os
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_lib"))
    import _style            # sets rcParams to 18pt / dpi 300

Any inline ``fontsize=`` you pass still overrides these rcParams, so keep every
inline value >= _style.FS (18). Prefer enlarging the figure / moving legends
outside the axes to avoid overlap rather than shrinking text below 18.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 18  # the floor for every text element

plt.rcParams.update({
    "font.size":        FS,
    "axes.titlesize":   FS,
    "axes.labelsize":   FS,
    "xtick.labelsize":  FS,
    "ytick.labelsize":  FS,
    "legend.fontsize":  FS,
    "figure.titlesize": FS,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})
