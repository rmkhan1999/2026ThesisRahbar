from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import thesis_style as ts


FIGURE_KEY = "fig4"
STEM = "figure4_repeated_recovery"
OUTPUT_DIR = Path(__file__).resolve().parent

N_REPLICATES = 8
N_DRAWS = 100

LOWER_PERCENTILE, UPPER_PERCENTILE = 5.0, 95.0

HEADLINE_INDEX = 7
HEADLINE_TAU_TRUE = 23.17
HEADLINE_TAU_MEAN = 31.99
HEADLINE_TAU_SD = 0.24


def load_replicates() -> tuple[list[dict], list[Path]]:
    records: list[dict] = []
    paths: list[Path] = []
    for index in range(N_REPLICATES):
        path = ts.require_file("docs", "sbc", f"replicate_{index:03d}.json")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for key in ("replicate", "posterior_draws", "generating_parameters"):
            if key not in payload:
                raise KeyError(f"{path.name} is missing required field {key!r}")
        if "tau" not in payload["posterior_draws"]:
            raise KeyError(f"{path.name} has no posterior_draws['tau']")
        if "tau" not in payload["generating_parameters"]:
            raise KeyError(f"{path.name} has no generating_parameters['tau']")
        draws = np.asarray(payload["posterior_draws"]["tau"], dtype=float)
        records.append(
            {
                "index": int(payload["replicate"]),
                "seed": payload.get("seed"),
                "draws": draws,
                "true": float(payload["generating_parameters"]["tau"]),
                "entrainment_failed": payload.get("entrainment_failed"),
                "true_transition_count": payload.get("true_transition_count"),
            }
        )
        paths.append(path)
    return records, paths


def load_analysis() -> tuple[dict, Path]:
    path = ts.require_file("docs", "sbc8_analysis.json")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    key = "per_replicate" if "per_replicate" in payload else "replicates"
    if key not in payload:
        raise KeyError(
            "docs/sbc8_analysis.json has neither 'per_replicate' nor 'replicates'"
        )
    payload["_per_replicate_key"] = key
    return payload, path


def run_checks(records: list[dict], analysis: dict) -> dict:
    log = ts.CheckLog("FIGURE 4")
    print("mandatory parse checks:")

    log.check(
        "eight replicate files, indexed 0-7 in campaign order",
        [r["index"] for r in records] == list(range(N_REPLICATES)),
        f"read {[r['index'] for r in records]}",
    )
    sizes = {r["index"]: r["draws"].size for r in records}
    finite = {r["index"]: int(np.isfinite(r["draws"]).sum()) for r in records}
    log.check(
        "every replicate stores exactly 100 finite posterior draws of tau, so "
        "the 5th-95th percentile interval is available identically for all eight",
        set(sizes.values()) == {N_DRAWS} and set(finite.values()) == {N_DRAWS},
        f"sizes {sorted(set(sizes.values()))}, finite {sorted(set(finite.values()))}",
    )

    per_key = analysis["_per_replicate_key"]
    by_index = {int(e["replicate"]): e for e in analysis[per_key]}
    log.check(
        "docs/sbc8_analysis.json carries all eight replicates",
        sorted(by_index) == list(range(N_REPLICATES)),
        f"read {sorted(by_index)}",
    )

    mean_ok, sd_ok, seed_ok, truth_ok = True, True, True, True
    for record in records:
        entry = by_index[record["index"]]
        mean = float(record["draws"].mean())
        sd = float(record["draws"].std(ddof=1))
        mean_ok &= abs(mean - float(entry["tau_posterior_mean"])) < 1e-9
        sd_ok &= abs(sd - float(entry["tau_posterior_sd"])) < 1e-9
        truth_ok &= abs(record["true"] - float(entry["tau_true"])) < 1e-9
        seed_ok &= int(record["seed"]) == int(entry["seed"])
    log.check(
        "posterior mean of the stored draws reproduces tau_posterior_mean in "
        "docs/sbc8_analysis.json for all eight",
        mean_ok,
    )
    log.check(
        "sd (ddof=1) of the stored draws reproduces tau_posterior_sd for all eight",
        sd_ok,
    )
    log.check(
        "generating_parameters['tau'] reproduces tau_true for all eight",
        truth_ok,
    )
    log.check("replicate seeds agree between the two sources", seed_ok)

    headline = records[HEADLINE_INDEX]
    log.check(
        "replicate 7: tau_true = 23.17 h",
        abs(headline["true"] - HEADLINE_TAU_TRUE) < 0.005,
        f"read {headline['true']:.4f}",
    )
    log.check(
        "replicate 7: posterior mean = 31.99 h",
        abs(float(headline["draws"].mean()) - HEADLINE_TAU_MEAN) < 0.005,
        f"read {float(headline['draws'].mean()):.4f}",
    )
    log.check(
        "replicate 7: posterior sd = 0.24 h",
        abs(float(headline["draws"].std(ddof=1)) - HEADLINE_TAU_SD) < 0.005,
        f"read {float(headline['draws'].std(ddof=1)):.4f}",
    )

    summary = analysis.get("summary", {})
    ess_range = summary.get("ess_bulk_min_range", [None, None])
    log.check(
        "campaign min bulk ESS range 1.58 to 16.34, mean 6.61",
        abs(float(ess_range[0]) - 1.58) < 0.005
        and abs(float(ess_range[1]) - 16.34) < 0.005
        and abs(float(summary["ess_bulk_min_mean"]) - 6.61) < 0.005,
        f"read {float(ess_range[0]):.4f} to {float(ess_range[1]):.4f}, "
        f"mean {float(summary['ess_bulk_min_mean']):.4f}",
    )
    log.check(
        "campaign mean tree-depth saturation 0.58 and nine divergences in total",
        abs(float(summary["mean_depth_saturation"]) - 0.575) < 5e-4
        and int(summary["total_divergences"]) == 9,
        f"read {summary['mean_depth_saturation']}, "
        f"{summary['total_divergences']} divergences",
    )

    log.close()
    return {"by_index": by_index}


def build_figure(records: list[dict]) -> tuple[plt.Figure, dict]:
    fig = ts.new_figure(FIGURE_KEY)
    ax = fig.add_axes((0.093, 0.225, 0.900, 0.735))

    lows, highs, means, truths = [], [], [], []
    for record in records:
        draws = record["draws"]
        lows.append(float(np.percentile(draws, LOWER_PERCENTILE)))
        highs.append(float(np.percentile(draws, UPPER_PERCENTILE)))
        means.append(float(draws.mean()))
        truths.append(record["true"])

    positions = np.arange(N_REPLICATES, dtype=float)

    ax.axhspan(
        HEADLINE_INDEX - 0.45,
        HEADLINE_INDEX + 0.45,
        color=ts.SHADE,
        alpha=0.65,
        lw=0,
        zorder=0,
    )

    for y, low, high in zip(positions, lows, highs):
        ax.plot([low, high], [y, y], color=ts.INK, lw=1.0, zorder=2,
                solid_capstyle="butt")
        for edge in (low, high):
            ax.plot([edge, edge], [y - 0.16, y + 0.16], color=ts.INK, lw=0.7,
                    zorder=2)

    ax.plot(means, positions, "o", ls="none", mfc=ts.INK, mec=ts.INK,
            ms=3.4, mew=0.7, zorder=4)
    ax.plot(truths, positions, "D", ls="none", mfc="white", mec=ts.ACCENT_2,
            ms=4.2, mew=1.0, zorder=5)

    ax.set_yticks(positions)
    ax.set_yticklabels([str(r["index"]) for r in records])
    ax.set_ylim(N_REPLICATES - 0.55, -0.55)
    ax.set_ylabel("replicate", labelpad=2.0)
    ax.set_xlabel(r"$\tau$ (h)", labelpad=1.5)
    ax.set_xlim(14.0, 53.5)
    ax.set_xticks([15, 20, 25, 30, 35, 40, 45, 50])
    ax.tick_params(axis="y", length=0)
    ts.tidy(ax)
    ax.spines["left"].set_visible(False)

    return fig, {
        "low": lows,
        "high": highs,
        "mean": means,
        "true": truths,
    }


def main() -> None:
    ts.apply_style()
    records, replicate_paths = load_replicates()
    analysis, analysis_path = load_analysis()
    ts.report_inputs(replicate_paths + [analysis_path])
    print()
    run_checks(records, analysis)
    print()

    fig, plotted = build_figure(records)
    print(
        "interval definition used for all eight replicates: "
        f"{LOWER_PERCENTILE:.0f}th-{UPPER_PERCENTILE:.0f}th percentile of the "
        f"{N_DRAWS} stored posterior draws of tau"
    )
    print("plotted values (h):")
    print("  rep   p5      p95     mean    generating")
    for index in range(N_REPLICATES):
        print(
            f"  {index}    {plotted['low'][index]:6.3f}  "
            f"{plotted['high'][index]:6.3f}  {plotted['mean'][index]:6.3f}  "
            f"{plotted['true'][index]:6.3f}"
        )
    print(
        "figure contains no rank axis, no ECDF, no uniform reference, no "
        "calibration label and no use of the string 'SBC'"
    )
    print()

    saved = ts.save_figure(fig, OUTPUT_DIR, STEM)
    plt.close(fig)
    ts.report_outputs(
        saved,
        "figures/fig4_repeated_recovery/figure4_repeated_recovery.pdf",
    )


if __name__ == "__main__":
    main()
