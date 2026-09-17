"""Shared figure style for the paper's static figures (print + arXiv).

Palette and marks follow the dataviz reference palette: categorical slots blue #2a78d6 / orange #eb6834, a sequential
blue ramp, neutral grays for context, hairline grids, text in ink tokens and never in a series colour.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
S1, S2 = "#2a78d6", "#eb6834"
NEUTRAL_1, NEUTRAL_2 = "#f0efec", "#e6e5e0"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
# Categorical order for model families; assigned in fixed order, never cycled (>8 series fold into "other").
CAT = ["#2a78d6", "#eb6834", "#1b9e77", "#7b5bb8", "#b8860b", "#c2185b", "#00838f", "#6d6c66"]
OUT, DATA = "reports/figures", "reports/figures/data"
MONTHS = list("JFMAMJJASOND")

RC = {
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlesize": 9.5, "axes.titleweight": "semibold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.labelcolor": INK2,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "pdf.fonttype": 42,
}
plt.rcParams.update(RC)


def save(fig, name: str):
    """Write <name>.pdf and <name>.png into reports/figures."""
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
