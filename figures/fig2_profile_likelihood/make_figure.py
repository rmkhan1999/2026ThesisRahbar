from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import thesis_style as ts


FIGURE_KEY = "fig2"
STEM = "figure2_profile_likelihood"
OUTPUT_DIR = Path(__file__).resolve().parent

PANELS: tuple[tuple[str, str], ...] = (
    ("chi_wake", r"$\chi_w$ (h)"),
    ("chi_sleep", r"$\chi_s$ (h)"),
    ("threshold_gap", r"$\Delta H$"),
    ("amplitude", r"$a$"),
)

THRESHOLD = 2.0

WALL_LOW, WALL_HIGH = 0.066, 0.0775

LOCAL_OPTIMUM_AMPLITUDES = (0.089, 0.1005)

XTICKS = {
    "chi_wake": (10, 20, 30, 40),
    "chi_sleep": (2, 4, 6, 8, 10),
    "threshold_gap": (0.3, 0.4, 0.5, 0.6, 0.7),
    "amplitude": (0.05, 0.10, 0.15, 0.20, 0.25),
}


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def load_campaign() -> tuple[dict, Path]:
    path = ts.require_file("docs", "profile_campaign.json")
    with path.open("r", encoding="utf-8") as handle:
        campaign = json.load(handle)
    for key in (
        "campaign_best_negative_log_likelihood",
        "mle_negative_log_likelihood",
        "unrestricted_mle_underconverged",
        "mle_relative_excess_over_campaign_best",
        "max_fev",
        "grid_size",
        "profiles",
    ):
        if key not in campaign:
            raise KeyError(
                f"docs/profile_campaign.json is missing required field {key!r}"
            )
    for name, _ in PANELS:
        if name not in campaign["profiles"]:
            raise KeyError(f"profile block {name!r} is missing from the campaign")
    return campaign, path


def run_checks(campaign: dict) -> dict:
    log = ts.CheckLog("FIGURE 2")
    print("mandatory parse checks:")

    best = campaign["campaign_best_negative_log_likelihood"]
    mle = campaign["mle_negative_log_likelihood"]
    excess = campaign["mle_relative_excess_over_campaign_best"]
    max_fev = int(campaign["max_fev"])

    log.check(
        "campaign_best_negative_log_likelihood == 0.08768915426785154",
        best == 0.08768915426785154,
        f"read {best!r}",
    )
    log.check(
        "mle_negative_log_likelihood == 0.09765235201674635",
        mle == 0.09765235201674635,
        f"read {mle!r}",
    )
    log.check(
        "unrestricted_mle_underconverged is true",
        campaign["unrestricted_mle_underconverged"] is True,
        f"read {campaign['unrestricted_mle_underconverged']!r}",
    )
    log.check(
        "mle_relative_excess_over_campaign_best ~ 0.1136",
        abs(float(excess) - 0.1136) < 5e-5,
        f"read {excess!r}",
    )
    log.check("max_fev == 1000", max_fev == 1000, f"read {max_fev}")

    counts = {n: len(campaign["profiles"][n]["points"]) for n, _ in PANELS}
    total = sum(counts.values())
    log.check(
        "exactly 84 points across the four blocks, 21 per block",
        total == 84 and set(counts.values()) == {21},
        f"total {total}, per block {counts}",
    )

    exhausted = {
        name: [p for p in campaign["profiles"][name]["points"] if int(p["nfev"]) == max_fev]
        for name, _ in PANELS
    }
    dist = {n: len(v) for n, v in exhausted.items()}
    log.check(
        "13 budget-exhausted points, distributed chi_wake 3 / chi_sleep 6 / "
        "threshold_gap 0 / amplitude 4",
        dist == {"chi_wake": 3, "chi_sleep": 6, "threshold_gap": 0, "amplitude": 4},
        f"read {dist}",
    )
    flat = [p for v in exhausted.values() for p in v]
    below = sum(1 for p in flat if float(p["delta_nll_from_campaign_best"]) < THRESHOLD)
    log.check(
        "of those 13, exactly 9 below the threshold and 4 not",
        below == 9 and (len(flat) - below) == 4,
        f"below {below}, at or above {len(flat) - below}",
    )
    log.check(
        "every budget-exhausted point also has success false",
        all(p["success"] is False for p in flat),
        f"success flags {sorted({p['success'] for p in flat})}",
    )

    amplitude = {
        round(float(p["fixed_value"]), 6): p
        for p in campaign["profiles"]["amplitude"]["points"]
    }

    def amp(value: float) -> dict:
        key = round(float(value), 6)
        if key not in amplitude:
            raise KeyError(f"amplitude grid has no evaluated point at a = {value}")
        return amplitude[key]

    p066 = amp(0.066)
    log.check(
        "a = 0.066 has delta_nll_from_campaign_best ~ 10.74 with success true",
        abs(float(p066["delta_nll_from_campaign_best"]) - 10.74) < 0.01
        and p066["success"] is True,
        f"delta {p066['delta_nll_from_campaign_best']:.6f}, success {p066['success']}",
    )
    p0775 = amp(0.0775)
    log.check(
        "a = 0.0775 has delta_nll_from_campaign_best ~ 0.094 with success true",
        abs(float(p0775["delta_nll_from_campaign_best"]) - 0.094) < 0.001
        and p0775["success"] is True,
        f"delta {p0775['delta_nll_from_campaign_best']:.6f}, success {p0775['success']}",
    )
    for value in LOCAL_OPTIMUM_AMPLITUDES:
        point = amp(value)
        log.check(
            f"a = {value} has success true and nfev < 1000 "
            f"(NOT budget-exhausted)",
            point["success"] is True and int(point["nfev"]) < max_fev,
            f"success {point['success']}, nfev {point['nfev']}, "
            f"delta {point['delta_nll_from_campaign_best']:.6f}",
        )

    log.close()
    return {
        "campaign_best": best,
        "mle": mle,
        "excess": excess,
        "max_fev": max_fev,
        "budget_distribution": dist,
        "budget_below_threshold": below,
    }


def classify(name: str, point: dict, max_fev: int) -> str:
    if int(point["nfev"]) == max_fev:
        return "exhausted"
    if name == "amplitude" and any(
        _close(point["fixed_value"], v) for v in LOCAL_OPTIMUM_AMPLITUDES
    ):
        return "local_optimum"
    return "converged"


def build_figure(campaign: dict, max_fev: int) -> tuple[plt.Figure, list[str]]:
    fig = ts.new_figure(FIGURE_KEY)
    gs = fig.add_gridspec(
        1,
        4,
        left=0.088,
        right=0.995,
        bottom=0.175,
        top=0.830,
        wspace=0.13,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    shaded_ends: list[str] = []

    for index, ((name, xlabel), ax) in enumerate(zip(PANELS, axes)):
        block = campaign["profiles"][name]
        points = sorted(block["points"], key=lambda p: float(p["fixed_value"]))
        x = np.array([float(p["fixed_value"]) for p in points])
        y = np.array([float(p["delta_nll_from_campaign_best"]) for p in points])
        kinds = [classify(name, p, max_fev) for p in points]

        span = x.max() - x.min()
        pad = 0.075 * span
        ax.set_xlim(x.min() - pad, x.max() + pad)

        if y[0] < THRESHOLD:
            ax.axvspan(x.min() - pad, x.min(), color=ts.SHADE, lw=0, zorder=0)
            shaded_ends.append(f"{name} lower end ({x.min():g})")
        if y[-1] < THRESHOLD:
            ax.axvspan(x.max(), x.max() + pad, color=ts.SHADE, lw=0, zorder=0)
            shaded_ends.append(f"{name} upper end ({x.max():g})")

        ax.plot(x, y, color=ts.LIGHT, lw=0.6, zorder=2, solid_joinstyle="round")

        ax.axhline(
            THRESHOLD,
            color=ts.INK,
            lw=0.6,
            ls=(0, (3.5, 2.0)),
            zorder=3,
        )

        if name == "amplitude":
            ax.axvspan(
                WALL_LOW,
                WALL_HIGH,
                color=ts.ACCENT,
                alpha=0.16,
                lw=0,
                zorder=1,
            )
            ax.annotate(
                "wall",
                xy=(0.5 * (WALL_LOW + WALL_HIGH), 0.0),
                xytext=(0, 1.0),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=ts.FONT_SMALL,
                color=ts.ACCENT,
            )

        for kind, marker, facecolor, edgecolor, size, order in (
            ("converged", "o", ts.INK, ts.INK, 3.0, 5),
            ("exhausted", "o", "white", ts.INK, 3.4, 5),
            ("local_optimum", "^", ts.ACCENT, ts.ACCENT, 3.6, 6),
        ):
            mask = np.array([k == kind for k in kinds])
            if not mask.any():
                continue
            ax.plot(
                x[mask],
                y[mask],
                marker,
                ls="none",
                mfc=facecolor,
                mec=edgecolor,
                mew=0.7,
                ms=size,
                zorder=order,
            )

        ax.set_yscale("symlog", linthresh=0.01, linscale=0.35)
        ax.set_ylim(0.0, 500.0)
        ax.set_xlabel(xlabel, labelpad=1.5)
        ax.set_xticks(XTICKS[name])
        ts.tidy(ax)

        if index == 0:
            ax.set_yticks([0.0, 0.1, 1.0, 10.0, 100.0])
            ax.set_yticklabels(["0", "0.1", "1", "10", "100"])
            ax.set_ylabel(r"$\Delta\mathrm{NLL}$", labelpad=1.5)
            ax.text(
                0.60,
                THRESHOLD,
                r"$\Delta\mathrm{NLL}=2$",
                transform=ax.get_yaxis_transform(which="grid"),
                ha="center",
                va="bottom",
                fontsize=ts.FONT_SMALL,
                color=ts.INK,
            )
        else:
            ax.set_yticks([0.0, 0.1, 1.0, 10.0, 100.0])
            ax.set_yticklabels([])
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", which="both", length=0)

        ts.panel_label(ax, f"({'abcd'[index]})", x=-0.02, y=1.015)

    handles = [
        Line2D([], [], marker="o", ls="none", mfc=ts.INK, mec=ts.INK,
               ms=3.0, mew=0.7, label="converged within budget"),
        Line2D([], [], marker="o", ls="none", mfc="white", mec=ts.INK,
               ms=3.4, mew=0.7, label="evaluation budget exhausted"),
        Line2D([], [], marker="^", ls="none", mfc=ts.ACCENT, mec=ts.ACCENT,
               ms=3.6, mew=0.7, label="local optimum"),
        Patch(facecolor=ts.SHADE, edgecolor="none", label="outside tested grid"),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.088, 0.990),
        ncol=4,
        fontsize=ts.FONT_SMALL,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.05,
    )
    return fig, shaded_ends


def main() -> None:
    ts.apply_style()
    campaign, path = load_campaign()
    ts.report_inputs([path])
    print()
    parsed = run_checks(campaign)
    print()

    fig, shaded_ends = build_figure(campaign, parsed["max_fev"])
    print("grid ends shaded as untested (endpoint still below the threshold):")
    for end in shaded_ends:
        print(f"  {end}")
    print(
        "amplitude transition band drawn between "
        f"a = {WALL_LOW} and a = {WALL_HIGH} only; "
        "no boundary drawn at a = 0.112 or a = 0.124"
    )
    print(
        "y quantity plotted: delta_nll_from_campaign_best "
        f"(reference NLL {parsed['campaign_best']:.6f}); "
        "delta_nll_from_mle is NOT used"
    )
    print()

    saved = ts.save_figure(fig, OUTPUT_DIR, STEM)
    plt.close(fig)
    ts.report_outputs(
        saved,
        "figures/fig2_profile_likelihood/figure2_profile_likelihood.pdf",
    )


if __name__ == "__main__":
    main()
