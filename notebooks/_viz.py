"""
notebooks/_viz.py — One chart style shared by all three notebooks.

WHY THIS FILE EXISTS:
Three notebooks, one visual language: the same surface, ink, gridlines and
colour order everywhere, so a reader learns the encoding once. Keeping it
here (not copied into each notebook) means a colour changes in one place.

DESIGN DECISIONS:
- Opaque light surface (#fcfcfb). PNGs embedded in .ipynb are shown as-is by
  GitHub and VS Code in BOTH themes. A transparent PNG with dark ink vanishes
  on a dark theme; an opaque light card stays readable in either.
- Colours are the reference data-viz palette in its validated order. Charts
  where every colour can sit next to every other (lines, box overlays) use at
  most the first 3 slots — the documented all-pairs-safe subset for colour
  vision deficiency. Anything beyond that is grey plus a text label.
- Emphasis over rainbow: when one variant is the point (the shipped config),
  it gets SERIES[0] and the rest are grey.
"""

from __future__ import annotations

import matplotlib as mpl             # matplotlib: static PNGs that render inside the
                                     # .ipynb on GitHub without a live kernel —
                                     # interactive libraries (plotly) need JS that
                                     # GitHub's notebook viewer strips.

SURFACE = "#fcfcfb"                  # chart surface
INK = "#0b0b0b"                      # primary text
INK_2 = "#52514e"                    # secondary text (axis titles)
MUTED = "#898781"                    # tick labels
GRID = "#e1e0d9"                     # hairline grid
AXIS = "#c3c2b7"                     # baseline / axis
DEEMPH = "#c3c2b7"                   # de-emphasised marks

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]      # blue, orange, aqua — slots 1-3
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply() -> None:
    """Set matplotlib defaults: light surface, hairline grid, thin marks, left titles."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 110, "savefig.bbox": "tight",
        "font.size": 10, "text.color": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 10, "axes.labelcolor": INK_2, "axes.labelsize": 9.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2, "xtick.major.size": 0, "ytick.major.size": 0,
        "lines.linewidth": 2.0, "lines.markersize": 7,
        "legend.frameon": False, "legend.fontsize": 9,
    })


def hbar(ax, labels: list[str], values: list[float], highlight: str | None = None,
         fmt: str = "{:.2f}", xmax: float | None = None) -> None:
    """
    Horizontal bars, one colour (or one highlighted bar + grey), values at bar ends.

    Args:
        ax:        Matplotlib axes.
        labels:    Category names (drawn top to bottom in the given order).
        values:    Bar lengths.
        highlight: Label to colour SERIES[0]; the rest grey. None = all SERIES[0].
        fmt:       Format for the end-of-bar value labels.
        xmax:      Right limit of the value axis (None = automatic).
    """
    colors = [SERIES[0] if highlight in (None, lab) else DEEMPH for lab in labels]
    y = range(len(labels))[::-1]
    ax.barh(list(y), values, height=0.62, color=colors, edgecolor=SURFACE, linewidth=1.5)
    ax.set_yticks(list(y), labels)
    ax.grid(axis="y", visible=False)
    right = xmax or max(values) * 1.15
    ax.set_xlim(0, right)
    for yi, v in zip(y, values):
        ax.text(v + right * 0.01, yi, fmt.format(v), va="center", fontsize=8.5, color=INK_2)
