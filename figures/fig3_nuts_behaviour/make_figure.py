from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import thesis_style as ts


FIGURE_KEY = "fig3"
STEM = "figure3_nuts_behaviour"
OUTPUT_DIR = Path(__file__).resolve().parent

DIAGNOSTIC_VARIABLES = (
    "phase_z1",
    "phase_z2",
    "excursion_fraction",
    "amplitude_fraction",
    "chi_sleep",
    "chi_wake",
    "tau",
)

FROZEN_PER_CHAIN_MIN_ESS = (2.11, 34.62, 18.48, 3.01)
FROZEN_RUN012_MIN_ESS = 1.40
FROZEN_RUN013_MIN_ESS = 1.48
FROZEN_RHAT_MAX = 1.6202307356626455
FROZEN_SD_TAU_234 = 2.64
FROZEN_CHAIN_MEAN_RANGE = 0.512
FROZEN_CHAIN_MEAN_SD = 0.256
FROZEN_WITHIN_CHAIN_SD = 2.63
FROZEN_RUN012_RECOVERY = 7.3e-3
FROZEN_RUN014_RECOVERY = (2.5e-2, 4.2e-2)

OFF_BRANCH_CHAIN = 1


def _bulk_ess(draws: np.ndarray) -> float:
    return float(np.asarray(az.ess(np.atleast_2d(np.asarray(draws)), method="bulk")))


def _rhat(draws: np.ndarray) -> float:
    return float(np.asarray(az.rhat(np.atleast_2d(np.asarray(draws)))))


def read_iterations(run_id: str) -> tuple[list[dict], Path]:
    path = ts.require_file("runs", run_id, "iterations.csv")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"chain", "phase", "iteration", "step_size"}
    if not rows:
        raise ValueError(f"{path} is empty")
    missing = required - set(rows[0])
    if missing:
        raise KeyError(f"{path} is missing columns {sorted(missing)}")
    return rows, path


def read_manifest(run_id: str) -> tuple[dict, Path]:
    path = ts.require_file("runs", run_id, "manifest.json")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle), path


def read_posterior(run_id: str) -> tuple[dict[str, np.ndarray], Path]:
    path = ts.require_file("runs", run_id, "posterior.nc")
    posterior = az.from_netcdf(path).posterior
    values: dict[str, np.ndarray] = {}
    for name in DIAGNOSTIC_VARIABLES:
        if name not in posterior:
            raise KeyError(f"{path} has no posterior variable {name!r}")
        values[name] = np.asarray(posterior[name].values, dtype=float)
    return values, path


def warmup_traces(rows: list[dict]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    traces: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for chain in sorted({int(r["chain"]) for r in rows}):
        selected = [
            (int(r["iteration"]), float(r["step_size"]))
            for r in rows
            if r["phase"] == "warmup" and int(r["chain"]) == chain
        ]
        selected.sort()
        traces[chain] = (
            np.array([i for i, _ in selected], dtype=float),
            np.array([s for _, s in selected], dtype=float),
        )
    return traces


def sampling_step_sizes(rows: list[dict]) -> dict[int, list[float]]:
    result: dict[int, list[float]] = {}
    for chain in sorted({int(r["chain"]) for r in rows}):
        result[chain] = sorted(
            {
                float(r["step_size"])
                for r in rows
                if r["phase"] == "sampling" and int(r["chain"]) == chain
            }
        )
    return result


def run_checks(data: dict) -> dict:
    log = ts.CheckLog("FIGURE 3")
    print("mandatory parse checks:")

    rows12, rows13, rows14 = data["rows12"], data["rows13"], data["rows14"]
    log.check(
        "row counts: RUN-012 300 (1 chain), RUN-013 100, RUN-014 1200 (4 chains)",
        len(rows12) == 300 and len(rows13) == 100 and len(rows14) == 1200,
        f"read {len(rows12)}, {len(rows13)}, {len(rows14)}",
    )
    t12, t14 = data["traces12"], data["traces14"]
    log.check(
        "warmup filter gives 1 x 200 iterations for RUN-012 and 4 x 200 for RUN-014",
        list(t12) == [1]
        and t12[1][0].size == 200
        and list(t14) == [1, 2, 3, 4]
        and all(t14[c][0].size == 200 for c in t14),
        f"RUN-012 {[t12[c][0].size for c in t12]}, "
        f"RUN-014 {[t14[c][0].size for c in t14]}",
    )

    adapted = [float(v) for v in data["manifest14"]["diagnostics"][
        "adapted_step_size_by_chain"]]
    trace_ends = [float(t14[c][1][-1]) for c in sorted(t14)]
    log.check(
        "RUN-014 final warmup step size per chain == "
        "diagnostics.adapted_step_size_by_chain (exact)",
        trace_ends == adapted,
        f"traces {['%.9f' % v for v in trace_ends]}",
    )
    sampled14 = data["sampling14"]
    log.check(
        "RUN-014 sampling-phase step size is one constant per chain, equal to "
        "the adapted value",
        all(len(sampled14[c]) == 1 for c in sampled14)
        and [sampled14[c][0] for c in sorted(sampled14)] == adapted,
        f"unique counts {[len(sampled14[c]) for c in sorted(sampled14)]}",
    )
    log.check(
        "RUN-014 adapted step sizes lie between 2.5e-2 and 4.2e-2",
        all(FROZEN_RUN014_RECOVERY[0] <= v <= FROZEN_RUN014_RECOVERY[1]
            for v in adapted),
        f"min {min(adapted):.6f}, max {max(adapted):.6f}",
    )

    it12, ss12 = t12[1]
    window = (it12 >= 51) & (it12 <= 100)
    log.check(
        "RUN-012 falls to about 1.0e-5 during warmup iterations 51-100",
        abs(float(ss12[window].min()) - 1.0e-5) < 2.0e-6,
        f"window minimum {float(ss12[window].min()):.4e} at iteration "
        f"{int(it12[window][ss12[window].argmin()])}",
    )
    log.check(
        "RUN-012 recovers only to 7.3e-3",
        abs(float(ss12[-1]) - FROZEN_RUN012_RECOVERY) < 5e-5,
        f"final warmup step size {float(ss12[-1]):.6e}",
    )
    log.check(
        "the RUN-014 recovery exceeds RUN-012's by a factor of three to six",
        3.0 <= min(adapted) / float(ss12[-1])
        and max(adapted) / float(ss12[-1]) <= 6.0,
        f"factors {min(adapted) / float(ss12[-1]):.2f} to "
        f"{max(adapted) / float(ss12[-1]):.2f}",
    )

    ss13 = {float(r["step_size"]) for r in rows13}
    log.check(
        "RUN-013 did not adapt: one constant step size equal to RUN-012's "
        "adapted value, so it is excluded from the adaptation traces",
        len(ss13) == 1 and abs(ss13.pop() - float(ss12[-1])) < 1e-15,
        "frozen-step run",
    )

    per_chain = data["per_chain_min_ess"]
    log.check(
        "RUN-014 per-chain minimum bulk ESS reproduces 2.11 / 34.62 / 18.48 / 3.01",
        all(abs(a - b) < 0.005 for a, b in zip(per_chain, FROZEN_PER_CHAIN_MIN_ESS)),
        f"read {[round(v, 4) for v in per_chain]}",
    )
    log.check(
        "RUN-012 minimum bulk ESS 1.40 and RUN-013 (frozen step) 1.48",
        abs(data["min_ess12"] - FROZEN_RUN012_MIN_ESS) < 0.005
        and abs(data["min_ess13"] - FROZEN_RUN013_MIN_ESS) < 0.005,
        f"read {data['min_ess12']:.4f} and {data['min_ess13']:.4f}",
    )
    comparable = [data["min_ess12"], *per_chain]
    log.check(
        "minimum bulk ESS across comparable warmups spans 1.40 to 34.62, a "
        "factor of about 25",
        abs(min(comparable) - 1.40) < 0.005
        and abs(max(comparable) - 34.62) < 0.005
        and abs(max(comparable) / min(comparable) - 25.0) < 1.0,
        f"{min(comparable):.4f} to {max(comparable):.4f}, "
        f"factor {max(comparable) / min(comparable):.2f}",
    )

    log.check(
        "RUN-014 four-chain max R-hat is 1.620, reproduced from posterior.nc and "
        "matching the manifest",
        abs(data["rhat_max"] - FROZEN_RHAT_MAX) < 1e-9
        and abs(float(data["manifest14"]["diagnostics"]["rhat_max"])
                - FROZEN_RHAT_MAX) < 1e-9,
        f"posterior.nc {data['rhat_max']:.6f}, manifest "
        f"{data['manifest14']['diagnostics']['rhat_max']:.6f}",
    )

    tau = data["tau"]
    pooled = float(tau[1:4].ravel().std(ddof=1))
    chain_means = tau[1:4].mean(axis=1)
    within = [float(tau[c].std(ddof=1)) for c in (1, 2, 3)]
    log.check(
        "chains 2-4 sd(tau) reproduces 2.64 h",
        abs(pooled - FROZEN_SD_TAU_234) < 0.005,
        f"read {pooled:.4f} h",
    )
    log.check(
        "the range of the three chain means reproduces 0.512 h",
        abs(float(chain_means.max() - chain_means.min())
            - FROZEN_CHAIN_MEAN_RANGE) < 0.0005,
        f"read {float(chain_means.max() - chain_means.min()):.4f} h",
    )
    log.check(
        "the sd of the three chain means reproduces 0.256 h (a different "
        "quantity from the range, and never labelled as a span)",
        abs(float(chain_means.std(ddof=1)) - FROZEN_CHAIN_MEAN_SD) < 0.0005,
        f"read {float(chain_means.std(ddof=1)):.4f} h",
    )
    log.check(
        "the mean within-chain sd of tau over chains 2-4 reproduces 2.63 h",
        abs(float(np.mean(within)) - FROZEN_WITHIN_CHAIN_SD) < 0.005,
        f"read {float(np.mean(within)):.4f} h "
        f"(per chain {[round(v, 3) for v in within]})",
    )

    log.close()
    return {
        "adapted": adapted,
        "pooled_sd": pooled,
        "chain_means": chain_means,
        "within": within,
        "per_chain_min_ess": per_chain,
    }


def build_figure(data: dict, summary: dict) -> plt.Figure:
    fig = ts.new_figure(FIGURE_KEY)
    ax_trace = fig.add_axes((0.066, 0.185, 0.393, 0.660))
    ax_ess = fig.add_axes((0.529, 0.185, 0.038, 0.660))
    ax_tau = fig.add_axes((0.643, 0.185, 0.347, 0.660))

    for chain in sorted(data["traces14"]):
        iterations, step = data["traces14"][chain]
        ax_trace.plot(iterations, step, color=ts.MID, lw=0.65, zorder=2)
    it12, ss12 = data["traces12"][1]
    ax_trace.plot(
        it12,
        ss12,
        color=ts.ACCENT_2,
        lw=1.0,
        ls=(0, (3.0, 1.6)),
        zorder=3,
    )
    ax_trace.set_yscale("log")
    ax_trace.set_ylim(4e-6, 6.0)
    ax_trace.set_xlim(0, 205)
    ax_trace.set_xticks([0, 50, 100, 150, 200])
    ax_trace.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    ax_trace.set_xlabel("warmup iteration", labelpad=1.5)
    ax_trace.set_ylabel("step size", labelpad=1.5)
    ts.tidy(ax_trace)
    ax_trace.annotate(
        "RUN-014, 4 chains",
        xy=(152.0, 1.0),
        ha="center",
        va="bottom",
        fontsize=ts.FONT_SMALL,
        color=ts.MID,
    )
    ax_trace.annotate(
        "RUN-012",
        xy=(124.0, 1.7e-5),
        ha="center",
        va="bottom",
        fontsize=ts.FONT_SMALL,
        color=ts.ACCENT_2,
    )
    ts.panel_label(ax_trace, "(a)", x=-0.015, y=1.02)

    ess14 = summary["per_chain_min_ess"]
    ax_ess.plot(
        np.zeros(len(ess14)),
        ess14,
        "o",
        ls="none",
        mfc=ts.MID,
        mec=ts.MID,
        ms=3.2,
        mew=0.7,
        zorder=3,
    )
    ax_ess.plot(
        [0.0],
        [data["min_ess12"]],
        "o",
        ls="none",
        mfc=ts.ACCENT_2,
        mec=ts.ACCENT_2,
        ms=3.2,
        mew=0.7,
        zorder=4,
    )
    span_low = min(min(ess14), data["min_ess12"])
    ax_ess.annotate(
        "",
        xy=(0.45, max(ess14)),
        xytext=(0.45, span_low),
        arrowprops={
            "arrowstyle": "<->",
            "linewidth": 0.55,
            "color": ts.INK,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
            "mutation_scale": 4.5,
        },
    )
    ax_ess.annotate(
        r"$\times 25$",
        xy=(0.68, 7.0),
        ha="left",
        va="center",
        fontsize=ts.FONT_SMALL,
        color=ts.INK,
        annotation_clip=False,
    )
    ax_ess.set_yscale("log")
    ax_ess.set_ylim(1.0, 60.0)
    ax_ess.set_xlim(-0.8, 1.4)
    ax_ess.set_xticks([])
    ax_ess.set_yticks([1.0, 10.0])
    ax_ess.set_yticklabels(["1", "10"])
    ax_ess.set_ylabel("min bulk ESS", labelpad=1.5, fontsize=ts.FONT_SMALL)
    ts.tidy(ax_ess)
    ax_ess.spines["bottom"].set_visible(False)
    ax_ess.tick_params(axis="x", length=0)

    tau = data["tau"]
    chain_means = summary["chain_means"]
    for chain, y in ((2, 0.0), (3, 1.0), (4, 2.0)):
        values = tau[chain - 1]
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        ax_tau.add_patch(
            plt.Rectangle(
                (mean - sd, y - 0.30),
                2.0 * sd,
                0.60,
                facecolor=ts.SHADE,
                edgecolor=ts.LIGHT,
                lw=0.5,
                zorder=1,
            )
        )
        ax_tau.plot([mean], [y], "o", mfc=ts.INK, mec=ts.INK, ms=3.2, mew=0.7,
                    zorder=4)

    low, high = float(chain_means.min()), float(chain_means.max())
    for edge in (low, high):
        ax_tau.plot([edge, edge], [-0.42, 2.42], color=ts.ACCENT, lw=0.6,
                    ls=(0, (1.5, 1.3)), zorder=3)
    ax_tau.annotate(
        "between-chain range 0.51 h",
        xy=(0.5 * (low + high), 2.52),
        ha="center",
        va="top",
        fontsize=ts.FONT_SMALL,
        color=ts.ACCENT,
    )
    ax_tau.annotate(
        "within-chain sd 2.63 h",
        xy=(13.7, -0.95),
        ha="left",
        va="center",
        fontsize=ts.FONT_SMALL,
        color=ts.INK,
    )
    ax_tau.annotate(
        r"$\hat{R}_{\max}=1.620$",
        xy=(36.3, -0.95),
        ha="right",
        va="center",
        fontsize=ts.FONT_SMALL,
        color=ts.INK,
    )

    off = tau[OFF_BRANCH_CHAIN - 1]
    off_mean, off_sd = float(off.mean()), float(off.std(ddof=1))
    y_off = 3.55
    ax_tau.axhline(3.05, color=ts.LIGHT, lw=0.5, zorder=0)
    ax_tau.add_patch(
        plt.Rectangle(
            (off_mean - off_sd, y_off - 0.30),
            2.0 * off_sd,
            0.60,
            facecolor="none",
            edgecolor=ts.ACCENT_2,
            lw=0.6,
            ls=(0, (2.2, 1.5)),
            zorder=1,
        )
    )
    ax_tau.plot([off_mean], [y_off], "o", mfc="white", mec=ts.ACCENT_2, ms=3.4,
                mew=0.9, zorder=4)
    ax_tau.annotate(
        "off the 1:1 branch",
        xy=(off_mean - off_sd - 0.7, y_off),
        ha="right",
        va="center",
        fontsize=ts.FONT_SMALL,
        color=ts.ACCENT_2,
    )

    ax_tau.set_yticks([0.0, 1.0, 2.0, y_off])
    ax_tau.set_yticklabels(["2", "3", "4", "1"])
    ax_tau.set_ylim(4.15, -1.35)
    ax_tau.set_xlim(13.5, 36.5)
    ax_tau.set_xticks([15, 20, 25, 30, 35])
    ax_tau.set_xlabel(r"$\tau$ (h)", labelpad=1.5)
    ax_tau.set_ylabel("chain", labelpad=2.0)
    ax_tau.tick_params(axis="y", length=0)
    ts.tidy(ax_tau)
    ax_tau.spines["left"].set_visible(False)
    ts.panel_label(ax_tau, "(b)", x=-0.02, y=1.02)

    return fig


def main() -> None:
    ts.apply_style()

    rows12, path_it12 = read_iterations("RUN-012")
    rows13, path_it13 = read_iterations("RUN-013")
    rows14, path_it14 = read_iterations("RUN-014")
    manifest12, path_m12 = read_manifest("RUN-012")
    manifest13, path_m13 = read_manifest("RUN-013")
    manifest14, path_m14 = read_manifest("RUN-014")
    post12, path_p12 = read_posterior("RUN-012")
    post13, path_p13 = read_posterior("RUN-013")
    post14, path_p14 = read_posterior("RUN-014")

    ts.report_inputs(
        [
            path_it12, path_m12, path_p12,
            path_it13, path_m13, path_p13,
            path_it14, path_m14, path_p14,
        ]
    )
    print("runs deliberately NOT read: RUN-004 and RUN-009 (not comparable with "
          "the final configuration)")
    print()

    tau = post14["tau"]
    if tau.shape != (4, 100):
        raise ValueError(f"RUN-014 tau has shape {tau.shape}, expected (4, 100)")

    data = {
        "rows12": rows12,
        "rows13": rows13,
        "rows14": rows14,
        "traces12": warmup_traces(rows12),
        "traces14": warmup_traces(rows14),
        "sampling14": sampling_step_sizes(rows14),
        "manifest12": manifest12,
        "manifest13": manifest13,
        "manifest14": manifest14,
        "tau": tau,
        "per_chain_min_ess": [
            min(_bulk_ess(post14[name][chain]) for name in DIAGNOSTIC_VARIABLES)
            for chain in range(4)
        ],
        "min_ess12": min(_bulk_ess(post12[name]) for name in DIAGNOSTIC_VARIABLES),
        "min_ess13": min(_bulk_ess(post13[name]) for name in DIAGNOSTIC_VARIABLES),
        "rhat_max": max(_rhat(post14[name]) for name in DIAGNOSTIC_VARIABLES),
    }

    summary = run_checks(data)
    print()
    print("warmup step-size shape, measured from the traces (not assumed):")
    for label, (iterations, step) in (
        [("RUN-012 chain 1", data["traces12"][1])]
        + [(f"RUN-014 chain {c}", data["traces14"][c]) for c in sorted(data["traces14"])]
    ):
        medians = [
            float(np.median(step[(iterations >= lo) & (iterations <= hi)]))
            for lo, hi in ((1, 50), (51, 100), (101, 150), (151, 200))
        ]
        print(
            f"  {label}: median step size by window "
            f"1-50 {medians[0]:.3e}, 51-100 {medians[1]:.3e}, "
            f"101-150 {medians[2]:.3e}, 151-200 {medians[3]:.3e}; "
            f"final {float(step[-1]):.6e}"
        )
    print()

    fig = build_figure(data, summary)
    saved = ts.save_figure(fig, OUTPUT_DIR, STEM)
    plt.close(fig)
    print(
        "panel (b) draws no profile-supported or fibre tau reference band, in "
        "line with the retired-claim replacement"
    )
    print()
    ts.report_outputs(
        saved,
        "figures/fig3_nuts_behaviour/figure3_nuts_behaviour.pdf",
    )


if __name__ == "__main__":
    main()
