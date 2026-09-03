from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]

FIGURES_ROOT = REPO_ROOT / "figures"


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def require_file(*parts: str) -> Path:
    path = repo_path(*parts)
    if not path.is_file():
        raise FileNotFoundError(
            f"required input artefact is missing: {path.relative_to(REPO_ROOT)} "
            f"(resolved to {path})"
        )
    return path


TEXTHEIGHT_PT = 702.783
LINEWIDTH_PT = 455.244
PT_PER_INCH = 72.27

TEXTHEIGHT_IN = TEXTHEIGHT_PT / PT_PER_INCH
LINEWIDTH_IN = LINEWIDTH_PT / PT_PER_INCH

CANVAS: Mapping[str, tuple[float, float]] = {
    "fig1": (6.30, 1.56),
    "fig2": (6.30, 2.53),
    "fig3": (6.30, 1.85),
    "fig4": (6.30, 1.75),
}


FONT_BASE = 8.0
FONT_TICK = 8.0
FONT_PANEL_LABEL = 9.0
FONT_ANNOT = 8.0
FONT_SMALL = 7.5

INK = "#1a1a1a"
MID = "#5c5c5c"
LIGHT = "#9a9a9a"
SHADE = "#e2e2e2"
ACCENT = "#2f5d8a"
ACCENT_2 = "#8a4b2f"

SERIF_STACK: tuple[str, ...] = (
    "cmr10",
    "CMU Serif",
    "Latin Modern Roman",
    "STIX Two Text",
    "DejaVu Serif",
)


def apply_style() -> None:
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": list(SERIF_STACK),
            "font.size": FONT_BASE,
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "axes.formatter.use_mathtext": True,
            "text.usetex": False,
            "text.color": INK,
            "axes.labelsize": FONT_BASE,
            "axes.titlesize": FONT_BASE,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.axisbelow": True,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.dpi": 200,
            "xtick.labelsize": FONT_TICK,
            "ytick.labelsize": FONT_TICK,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "xtick.minor.size": 1.4,
            "ytick.minor.size": 1.4,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.major.pad": 2.0,
            "ytick.major.pad": 2.0,
            "xtick.color": INK,
            "ytick.color": INK,
            "lines.linewidth": 0.9,
            "lines.markersize": 3.0,
            "lines.markeredgewidth": 0.7,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0.6,
            "legend.fontsize": FONT_SMALL,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.handletextpad": 0.5,
            "legend.labelspacing": 0.25,
            "legend.columnspacing": 1.1,
            "legend.borderpad": 0.0,
            "legend.borderaxespad": 0.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.compression": 6,
            "svg.hashsalt": "twopm-thesis-figures",
            "figure.constrained_layout.use": False,
        }
    )


def new_figure(figure_key: str) -> plt.Figure:
    if figure_key not in CANVAS:
        raise KeyError(f"unknown figure key {figure_key!r}")
    width, height = CANVAS[figure_key]
    return plt.figure(figsize=(width, height))


def panel_label(ax: plt.Axes, text: str, *, x: float = 0.0, y: float = 1.0) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT_PANEL_LABEL,
        color=INK,
    )


def tidy(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color(INK)
    ax.tick_params(which="both", direction="out")


_PDF_METADATA = {
    "Title": None,
    "Author": None,
    "Subject": None,
    "Keywords": None,
    "Producer": None,
    "Creator": None,
    "CreationDate": None,
    "ModDate": None,
}


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    png_dpi: int = 200,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"

    fig.savefig(pdf_path, format="pdf", metadata=_PDF_METADATA)
    fig.savefig(png_path, format="png", dpi=png_dpi)

    width_in, height_in = (float(v) for v in fig.get_size_inches())
    return {
        "pdf": pdf_path,
        "png": png_path,
        "width_in": width_in,
        "height_in": height_in,
        "width_pt": width_in * PT_PER_INCH,
        "height_pt": height_in * PT_PER_INCH,
    }


def report_inputs(paths: Iterable[Path]) -> None:
    print("input artefacts consumed:")
    for path in paths:
        path = Path(path)
        try:
            shown = path.relative_to(REPO_ROOT)
        except ValueError:
            shown = path
        print(f"  {shown}")


def report_outputs(saved: Mapping[str, object], latex_target: str) -> None:
    pdf = Path(str(saved["pdf"]))
    png = Path(str(saved["png"]))
    print("outputs written:")
    print(f"  {pdf.relative_to(REPO_ROOT)}")
    print(f"  {png.relative_to(REPO_ROOT)}")
    print(
        "final PDF canvas: "
        f"{saved['width_in']:.4f} x {saved['height_in']:.4f} in "
        f"= {saved['width_pt']:.2f} x {saved['height_pt']:.2f} pt "
        "(TeX pt, 72.27/in)"
    )
    print(f"latex: \\includegraphics[width=\\linewidth]{{{latex_target}}}")


class CheckLog:

    def __init__(self, label: str) -> None:
        self.label = label
        self.failures: list[str] = []

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        status = "PASS" if passed else "FAIL"
        suffix = f"  [{detail}]" if detail else ""
        print(f"  {status}  {name}{suffix}")
        if not passed:
            self.failures.append(f"{name}{suffix}")
        return passed

    def close(self) -> None:
        if self.failures:
            joined = "\n  ".join(self.failures)
            raise SystemExit(
                f"{self.label} - PARSE MISMATCH, {len(self.failures)} check(s) "
                f"failed:\n  {joined}"
            )
        print(f"  all checks passed for {self.label}")
