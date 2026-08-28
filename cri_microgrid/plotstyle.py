"""Shared IEEE-style figure formatting and a single validated color system
used by every notebook, so all eight paper figures read as one visual
family. Colors are taken unmodified from a pre-validated categorical /
status / sequential / diverging palette (CVD-safe adjacent pairs, checked
contrast) -- see the dataviz palette this project's figures were built
against; nothing here re-derives or eyeballs a color.
"""
from typing import Optional, Sequence
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

IEEE_COL_WIDTH_IN = 3.45
IEEE_PAGE_WIDTH_IN = 7.16

FIGSIZE_1COL = (IEEE_COL_WIDTH_IN, 4.6)
FIGSIZE_1COL_SHORT = (IEEE_COL_WIDTH_IN, 3.0)
FIGSIZE_1COL_TALL = (IEEE_COL_WIDTH_IN, 6.0)
FIGSIZE_1COL_STACK3 = (IEEE_COL_WIDTH_IN, 5.4)     # 3 stacked (a)/(b)/(c) panels
FIGSIZE_DASHBOARD = (IEEE_PAGE_WIDTH_IN, 7.6)      # full-width, many stacked panels
FIGSIZE_2COL_SHORT = (IEEE_PAGE_WIDTH_IN, 3.2)
FIGSIZE_2x2 = (IEEE_COL_WIDTH_IN * 1.55, IEEE_COL_WIDTH_IN * 1.4)

BASE_RC = {
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "lines.linewidth": 1.3,
    "axes.grid": True,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "grid.color": "#b7b6ae",
    "axes.edgecolor": "#c3c2b7",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def apply_style():
    plt.rcParams.update(BASE_RC)


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")


# --- Categorical palette (fixed order; never cycled/reassigned) -----------
CAT = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948",
}

# The five readiness indicators, in the fixed order they're introduced in
# the paper (RI, CI, SMI, AI, NCI); CRI itself gets a visually distinct,
# unused-elsewhere hue since it is the headline aggregate, not a sixth peer.
INDICATOR_COLORS = {
    "RI": CAT["blue"], "CI": CAT["orange"], "SMI": CAT["aqua"],
    "AI": CAT["yellow"], "NCI": CAT["magenta"], "CRI": CAT["violet"],
}

# Strategy-comparison studies (Direct / Threshold / CRI), fixed order.
STRATEGY_COLORS = {"direct": CAT["blue"], "threshold": CAT["orange"], "cri": CAT["aqua"]}
STRATEGY_LABELS = {"direct": "Direct activation", "threshold": "Threshold-based", "cri": "CRI-based"}

# --- Status palette: readiness classification regions (fixed, never themed) ---
STATUS = {"Activate": "#0ca30c", "Monitor": "#fab219", "Delay": "#ec835a", "Reject": "#d03b3b"}
STATUS_ORDER = ["Reject", "Delay", "Monitor", "Activate"]
CLASS_CODE_TO_NAME = {3: "Activate", 2: "Monitor", 1: "Delay", 0: "Reject"}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"

# --- Sequential ramp (single hue, light -> dark blue) ----------------------
_SEQ_BLUE_STEPS = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("seq_blue", _SEQ_BLUE_STEPS)

# --- Diverging pair (blue <-> red, neutral gray midpoint) -------------------
_DIV_STEPS = ["#0d366b", "#2a78d6", "#f0efec", "#e34948", "#7a1414"]
DIVERGING_CMAP = LinearSegmentedColormap.from_list("div_blue_red", _DIV_STEPS)


def shade_classification_bg(ax, t: np.ndarray, cls_code: np.ndarray, alpha: float = 0.12):
    """Shade the axes background by readiness classification (status
    colors), so the Activate/Monitor/Delay/Reject region is visible behind
    any time-series drawn on top."""
    codes = np.asarray(cls_code)
    if len(t) < 2:
        return
    change_idx = np.where(np.diff(codes) != 0)[0]
    starts = np.concatenate(([0], change_idx + 1))
    ends = np.concatenate((change_idx + 1, [len(t) - 1]))
    for s_idx, e_idx in zip(starts, ends):
        name = CLASS_CODE_TO_NAME.get(int(codes[s_idx]), "Reject")
        ax.axvspan(t[s_idx], t[min(e_idx, len(t) - 1)], color=STATUS[name], alpha=alpha, lw=0)


def classification_legend_handles():
    from matplotlib.patches import Patch
    return [Patch(facecolor=STATUS[name], alpha=0.35, label=name) for name in STATUS_ORDER[::-1]]
