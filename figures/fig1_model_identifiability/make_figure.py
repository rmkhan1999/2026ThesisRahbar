from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import thesis_style as ts

sys.path.insert(0, str(ts.REPO_ROOT / "src"))

from twopm.config import load_config
from twopm.hard_switch import simulate_hard_switch
from twopm.posterior_summaries import solve_entrained_transition


FIGURE_KEY = "fig1"
STEM = "figure1_model_identifiability"
OUTPUT_DIR = Path(__file__).resolve().parent

CANONICAL_PHASE = 8.0
CANONICAL_CHI_SLEEP = 4.2
CANONICAL_CHI_WAKE = 18.2
CANONICAL_AMPLITUDE = 0.12
CANONICAL_THRESHOLD_GAP = 0.50

FROZEN_T_ON = 15.4581
FROZEN_T_OFF = 23.3978

RECORDING_BURN_IN = 48.0
RECORDING_DT = 0.001
RETAINED_HOURS = 24.0
EPOCH_HOURS = 0.5

RENDER_BURN_IN = 96.0
RENDER_DT_COARSE = 1.0e-4
RENDER_DT = 2.0e-5

ALTERNATIVES = (
    ("generating truth", 22.58, 15.45808, 23.39776, "o"),
    ("strong-parameter restart", 28.82, 15.43304, 23.37625, "s"),
    ("joint restart", 32.29, 15.43596, 23.38135, "^"),
)


def load_recording() -> tuple[dict, list[Path]]:
    npz_path = ts.require_file("docs", "hard_generated_recording.npz")
    json_path = ts.require_file("docs", "hard_generated_recording.json")
    with np.load(npz_path, mmap_mode=None) as archive:
        arrays = {name: np.array(archive[name]) for name in archive.files}
    with json_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    for name, shape in (
        ("time", (49,)),
        ("gate", (49,)),
        ("observations", (49,)),
        ("true_switch_times", (2,)),
        ("true_switch_states", (2,)),
    ):
        if name not in arrays:
            raise KeyError(f"{npz_path.name} has no array {name!r}")
        if arrays[name].shape != shape:
            raise ValueError(
                f"{npz_path.name}[{name}] has shape {arrays[name].shape}, "
                f"expected {shape}"
            )
    return {"arrays": arrays, "meta": meta}, [npz_path, json_path]


def load_metrics() -> tuple[dict, Path]:
    path = ts.require_file("docs", "identifiability_metrics.json")
    with path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    for key in ("canonical_transition_times", "physical_excursion_split"):
        if key not in metrics:
            raise KeyError(f"{path.name} has no {key!r}")
    block = metrics["physical_excursion_split"]
    for key in ("rank", "kernel_dimension", "coordinates"):
        if key not in block:
            raise KeyError(f"{path.name}['physical_excursion_split'] has no {key!r}")
    return metrics, path


def render(burn_in: float, dt: float) -> object:
    config = load_config(ts.repo_path("config", "model.yaml"))
    model = config.section("model")
    initial = config.section("initial_state")
    if float(model["chi_sleep"]) != CANONICAL_CHI_SLEEP:
        raise ValueError("config/model.yaml chi_sleep is not the canonical 4.2")
    if float(model["chi_wake"]) != CANONICAL_CHI_WAKE:
        raise ValueError("config/model.yaml chi_wake is not the canonical 18.2")
    if float(model["circadian_amplitude"]) != CANONICAL_AMPLITUDE:
        raise ValueError("config/model.yaml circadian_amplitude is not 0.12")
    gap = float(model["upper_base"]) - float(model["lower_base"])
    if abs(gap - CANONICAL_THRESHOLD_GAP) > 1e-12:
        raise ValueError(
            f"config/model.yaml threshold gap is {gap}, not the canonical 0.50"
        )
    return simulate_hard_switch(
        duration=burn_in + RETAINED_HOURS,
        dt=dt,
        chi_sleep=float(model["chi_sleep"]),
        chi_wake=float(model["chi_wake"]),
        mu=float(model["mu"]),
        upper=float(model["upper_base"]),
        lower=float(model["lower_base"]),
        initial_pressure=float(initial["pressure"]),
        initially_asleep=bool(initial["hard_asleep"]),
        amplitude=float(model["circadian_amplitude"]),
        phase=CANONICAL_PHASE,
        period=float(model["circadian_period"]),
        start_time=0.0,
    )


def retained_switches(result: object, burn_in: float) -> np.ndarray:
    times = np.asarray(result.switch_times, dtype=float)
    return times[times >= burn_in] - burn_in


def epoch_labels(result: object, burn_in: float) -> np.ndarray:
    time = np.asarray(result.time, dtype=float) - burn_in
    keep = time >= -1e-9
    time, asleep = time[keep], np.asarray(result.asleep)[keep]
    epochs = EPOCH_HOURS * np.arange(int(round(RETAINED_HOURS / EPOCH_HOURS)) + 1)
    index = np.clip(np.searchsorted(time, epochs, side="left"), 0, time.size - 1)
    previous = np.clip(index - 1, 0, time.size - 1)
    use_previous = np.abs(time[previous] - epochs) < np.abs(time[index] - epochs)
    index = np.where(use_previous, previous, index)
    return asleep[index].astype(float)


def run_checks(recording: dict, metrics: dict, renders: dict) -> dict:
    log = ts.CheckLog("FIGURE 1")
    print("mandatory parse checks:")

    arrays, meta = recording["arrays"], recording["meta"]

    canonical = metrics["canonical_transition_times"]
    log.check(
        "docs/identifiability_metrics.json canonical_transition_times round to "
        "the frozen 15.4581 / 23.3978 h",
        abs(round(float(canonical[0]), 4) - FROZEN_T_ON) < 5e-9
        and abs(round(float(canonical[1]), 4) - FROZEN_T_OFF) < 5e-9,
        f"read {float(canonical[0]):.6f}, {float(canonical[1]):.6f}",
    )

    block = metrics["physical_excursion_split"]
    log.check(
        "rank and kernel dimension come from physical_excursion_split, the "
        "TRAJECTORY chart: rank 2, kernel dimension 3, 5 coordinates",
        int(block["rank"]) == 2
        and int(block["kernel_dimension"]) == 3
        and len(block["coordinates"]) == 5,
        f"rank {block['rank']}, kernel {block['kernel_dimension']}, "
        f"coordinates {block['coordinates']}",
    )
    sampled = metrics["sampled_unconstrained"]
    log.check(
        "the sampled chart is a different space and is NOT used: its kernel "
        "dimension is 5",
        int(sampled["kernel_dimension"]) == 5,
        f"sampled_unconstrained kernel_dimension {sampled['kernel_dimension']}",
    )

    log.check(
        "recording metadata is the canonical vector at lambda = 0.01",
        float(meta["phase"]) == CANONICAL_PHASE
        and float(meta["chi_sleep"]) == CANONICAL_CHI_SLEEP
        and float(meta["chi_wake"]) == CANONICAL_CHI_WAKE
        and float(meta["amplitude"]) == CANONICAL_AMPLITUDE
        and abs(float(meta["upper"]) - float(meta["lower"])
                - CANONICAL_THRESHOLD_GAP) < 1e-12
        and float(meta["misclassification"]) == 0.01,
        f"phase {meta['phase']}, chi_sleep {meta['chi_sleep']}, "
        f"chi_wake {meta['chi_wake']}, amplitude {meta['amplitude']}, "
        f"gap {float(meta['upper']) - float(meta['lower']):.2f}",
    )
    log.check(
        "recording holds 49 epochs on a 0.5 h grid over 24 h and two switches",
        int(meta["n_epochs"]) == 49
        and float(meta["epoch_hours"]) == EPOCH_HOURS
        and float(meta["retained_hours"]) == RETAINED_HOURS
        and arrays["true_switch_times"].size == 2
        and bool(arrays["true_switch_states"][0]) is True
        and bool(arrays["true_switch_states"][1]) is False,
        f"epochs {meta['n_epochs']}, switch states "
        f"{arrays['true_switch_states'].tolist()}",
    )

    log.check(
        "the simulator at the recording's own configuration (burn-in 48 h, "
        "dt 1e-3) reproduces the stored true_switch_times exactly",
        np.array_equal(renders["recording_switches"], arrays["true_switch_times"]),
        f"rendered {np.round(renders['recording_switches'], 6).tolist()}, "
        f"stored {np.round(arrays['true_switch_times'], 6).tolist()}",
    )

    analytic = renders["analytic"]
    log.check(
        "the validated entrained root solver in "
        "src/twopm/posterior_summaries.py reproduces the stored canonical "
        "transition times to better than 1e-9 h",
        abs(analytic[0] - float(canonical[0])) < 1e-9
        and abs(analytic[1] - float(canonical[1])) < 1e-9,
        f"solver {analytic[0]:.9f}, {analytic[1]:.9f}",
    )

    plotted = renders["plot_switches"]
    coarse = renders["coarse_switches"]
    residual_fine = np.abs(plotted[:2] - np.array([canonical[0], canonical[1]]))
    residual_coarse = np.abs(coarse[:2] - np.array([canonical[0], canonical[1]]))
    log.check(
        "the plotted rendering's crossings round to the frozen 15.4581 / "
        "23.3978 h",
        abs(round(float(plotted[0]), 4) - FROZEN_T_ON) < 5e-9
        and abs(round(float(plotted[1]), 4) - FROZEN_T_OFF) < 5e-9,
        f"rendered {float(plotted[0]):.6f}, {float(plotted[1]):.6f}",
    )
    log.check(
        "the residual against the canonical roots is the simulator's O(dt) "
        "first-grid-point detection bias: it shrinks as dt is refined",
        bool(np.all(residual_fine < residual_coarse))
        and bool(np.all(residual_fine * 3600.0 < 1.0)),
        f"dt {RENDER_DT_COARSE:g} residual "
        f"{np.round(residual_coarse * 3600.0, 3).tolist()} s -> "
        f"dt {RENDER_DT:g} residual "
        f"{np.round(residual_fine * 3600.0, 3).tolist()} s",
    )
    log.check(
        "the plotted rendering reproduces the recording's 49 stored epoch "
        "labels exactly, so the continuous and observation layers agree",
        np.array_equal(renders["plot_epoch_labels"], arrays["gate"]),
        f"{int((renders['plot_epoch_labels'] == arrays['gate']).sum())}/49 match",
    )

    spread_on = max(a[2] for a in ALTERNATIVES) - min(a[2] for a in ALTERNATIVES)
    spread_off = max(a[3] for a in ALTERNATIVES) - min(a[3] for a in ALTERNATIVES)
    tau_spread = max(a[1] for a in ALTERNATIVES) - min(a[1] for a in ALTERNATIVES)
    log.check(
        "nearly ten hours of free-running period map to transition times about "
        "a minute apart",
        9.0 < tau_spread < 10.0
        and 0.5 < spread_on * 60.0 < 2.0
        and 0.5 < spread_off * 60.0 < 2.0,
        f"tau spread {tau_spread:.2f} h, t_on spread {spread_on * 60.0:.2f} min, "
        f"t_off spread {spread_off * 60.0:.2f} min",
    )

    log.close()
    return {
        "rank": int(block["rank"]),
        "kernel_dimension": int(block["kernel_dimension"]),
        "canonical": (float(canonical[0]), float(canonical[1])),
    }


def build_figure(recording: dict, renders: dict, parsed: dict) -> plt.Figure:
    fig = ts.new_figure(FIGURE_KEY)

    ax_main = fig.add_axes((0.062, 0.375, 0.358, 0.470))
    ax_strip = fig.add_axes((0.062, 0.238, 0.358, 0.098))
    ax_tau = fig.add_axes((0.497, 0.238, 0.012, 0.607))
    ax_map = fig.add_axes((0.782, 0.238, 0.205, 0.607))

    time = renders["plot_time"]
    ax_main.axvspan(FROZEN_T_ON, FROZEN_T_OFF, color=ts.SHADE, alpha=0.75, lw=0,
                    zorder=0)
    ax_main.plot(time, renders["plot_upper"], color=ts.MID, lw=0.7,
                 ls=(0, (3.2, 1.7)), zorder=2)
    ax_main.plot(time, renders["plot_lower"], color=ts.MID, lw=0.7,
                 ls=(0, (1.4, 1.4)), zorder=2)
    ax_main.plot(time, renders["plot_pressure"], color=ts.INK, lw=1.0, zorder=3)

    for crossing, threshold in (
        (FROZEN_T_ON, renders["upper_at_on"]),
        (FROZEN_T_OFF, renders["lower_at_off"]),
    ):
        ax_main.plot([crossing, crossing], [0.0, 1.0], color=ts.ACCENT, lw=0.55,
                     ls=(0, (1.6, 1.4)), zorder=1)
        ax_main.plot([crossing], [threshold], "o", mfc=ts.ACCENT, mec="white",
                     ms=3.2, mew=0.6, zorder=5)

    ax_main.annotate(r"$U(t)$", xy=(8.0, 0.805), ha="center", va="bottom",
                     fontsize=ts.FONT_SMALL, color=ts.MID)
    ax_main.annotate(r"$L(t)$", xy=(10.6, 0.275), ha="center", va="bottom",
                     fontsize=ts.FONT_SMALL, color=ts.MID)
    ax_main.annotate(r"$t_{\mathrm{on}}$", xy=(FROZEN_T_ON, 0.855), ha="center",
                     va="bottom", fontsize=ts.FONT_SMALL, color=ts.ACCENT)
    ax_main.annotate(r"$t_{\mathrm{off}}$", xy=(23.15, 0.855), ha="right",
                     va="bottom", fontsize=ts.FONT_SMALL, color=ts.ACCENT)

    ax_main.set_xlim(0.0, 24.0)
    ax_main.set_ylim(0.0, 1.0)
    ax_main.set_xticks([0, 6, 12, 18, 24])
    ax_main.set_xticklabels([])
    ax_main.set_yticks([0.0, 0.5, 1.0])
    ax_main.set_yticklabels(["0", "0.5", "1"])
    ax_main.set_ylabel(r"$H$", labelpad=1.5)
    ts.tidy(ax_main)

    epochs = recording["arrays"]["time"]
    asleep = recording["arrays"]["observations"].astype(bool)
    ax_strip.vlines(epochs[~asleep], 0.18, 0.50, color=ts.LIGHT, lw=1.5)
    ax_strip.vlines(epochs[asleep], 0.18, 0.96, color=ts.INK, lw=1.5)
    ax_strip.set_xlim(0.0, 24.0)
    ax_strip.set_ylim(0.0, 1.0)
    ax_strip.set_xticks([0, 6, 12, 18, 24])
    ax_strip.set_yticks([])
    ax_strip.set_xlabel("clock time (h)", labelpad=1.5)
    ax_strip.set_ylabel(
        "epochs",
        rotation=0,
        ha="right",
        va="center",
        labelpad=3.0,
        fontsize=ts.FONT_SMALL,
    )
    ts.tidy(ax_strip)
    ax_strip.spines["left"].set_visible(False)

    for _, tau, _, _, marker in ALTERNATIVES:
        ax_tau.plot([0.0], [tau], marker, mfc="white", mec=ts.INK, ms=3.6,
                    mew=0.8, zorder=3, clip_on=False)
    ax_tau.set_xlim(-1.0, 1.0)
    ax_tau.set_ylim(20.8, 34.1)
    ax_tau.set_xticks([])
    ax_tau.set_yticks([a[1] for a in ALTERNATIVES])
    ax_tau.set_yticklabels([f"{a[1]:.2f}" for a in ALTERNATIVES])
    ax_tau.set_ylabel(r"$\tau$ (h)", labelpad=1.5)
    ts.tidy(ax_tau)
    ax_tau.spines["bottom"].set_visible(False)
    ax_tau.tick_params(axis="x", length=0)

    for _, _, t_on, t_off, marker in ALTERNATIVES:
        ax_map.plot([t_on], [t_off], marker, mfc="white", mec=ts.INK, ms=3.6,
                    mew=0.8, zorder=3)
    ax_map.set_xlim(15.20, 15.70)
    ax_map.set_ylim(23.15, 23.65)
    ax_map.set_xticks([15.3, 15.6])
    ax_map.set_xticklabels(["15.3", "15.6"])
    ax_map.set_yticks([23.3, 23.5])
    ax_map.set_yticklabels(["23.3", "23.5"])
    ax_map.set_xlabel(r"$t_{\mathrm{on}}$ (h)", labelpad=1.5)
    ax_map.set_ylabel(r"$t_{\mathrm{off}}$ (h)", labelpad=1.5)
    ts.tidy(ax_map)

    arrow = FancyArrowPatch(
        (0.545, 0.700),
        (0.700, 0.700),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=6.0,
        lw=0.7,
        color=ts.INK,
        shrinkA=0.0,
        shrinkB=0.0,
    )
    fig.add_artist(arrow)
    fig.text(0.622, 0.725, r"$G$", ha="center", va="bottom",
             fontsize=ts.FONT_ANNOT, color=ts.INK)
    fig.text(
        0.622,
        0.645,
        f"rank $DG={parsed['rank']}$\n"
        rf"$\dim\ker DG={parsed['kernel_dimension']}$" "\n"
        "(trajectory chart)",
        ha="center",
        va="top",
        fontsize=ts.FONT_SMALL,
        color=ts.INK,
        linespacing=1.35,
    )

    fig.text(0.008, 0.875, "(a)", ha="left", va="bottom",
             fontsize=ts.FONT_PANEL_LABEL, color=ts.INK)
    fig.text(0.448, 0.875, "(b)", ha="left", va="bottom",
             fontsize=ts.FONT_PANEL_LABEL, color=ts.INK)
    return fig


def main() -> None:
    ts.apply_style()

    recording, recording_paths = load_recording()
    metrics, metrics_path = load_metrics()
    config_path = ts.require_file("config", "model.yaml")

    ts.report_inputs(
        recording_paths + [metrics_path, config_path]
    )
    print(
        "simulator re-rendered for visualisation only: "
        "src/twopm/hard_switch.py::simulate_hard_switch "
        "(via simulate_hard_switch at config/model.yaml values); "
        "root cross-check via "
        "src/twopm/posterior_summaries.py::solve_entrained_transition"
    )
    print(
        "recording config_sha256 (from docs/hard_generated_recording.json): "
        f"{recording['meta']['config_sha256']}"
    )
    print()

    recording_render = render(RECORDING_BURN_IN, RECORDING_DT)
    coarse_render = render(RENDER_BURN_IN, RENDER_DT_COARSE)
    plot_render = render(RENDER_BURN_IN, RENDER_DT)

    omega_phase = 2.0 * np.pi * CANONICAL_PHASE / 24.0
    analytic = solve_entrained_transition(
        c1=CANONICAL_AMPLITUDE * np.cos(omega_phase),
        c2=CANONICAL_AMPLITUDE * np.sin(omega_phase),
        chi_sleep=CANONICAL_CHI_SLEEP,
        chi_wake=CANONICAL_CHI_WAKE,
        threshold_gap=CANONICAL_THRESHOLD_GAP,
    )
    if not analytic.converged:
        raise RuntimeError(
            f"entrained root solver failed at the canonical vector: "
            f"{analytic.reason}"
        )

    time = np.asarray(plot_render.time, dtype=float) - RENDER_BURN_IN
    keep = (time >= -1e-9) & (time <= RETAINED_HOURS + 1e-9)
    stride = max(1, int(keep.sum() // 4000))
    renders = {
        "recording_switches": retained_switches(recording_render, RECORDING_BURN_IN),
        "coarse_switches": retained_switches(coarse_render, RENDER_BURN_IN),
        "plot_switches": retained_switches(plot_render, RENDER_BURN_IN),
        "plot_epoch_labels": epoch_labels(plot_render, RENDER_BURN_IN),
        "analytic": (analytic.onset, analytic.offset),
        "plot_time": time[keep][::stride],
        "plot_pressure": np.asarray(plot_render.pressure)[keep][::stride],
        "plot_upper": np.asarray(plot_render.upper_threshold)[keep][::stride],
        "plot_lower": np.asarray(plot_render.lower_threshold)[keep][::stride],
    }
    renders["upper_at_on"] = float(
        np.interp(FROZEN_T_ON, renders["plot_time"], renders["plot_upper"])
    )
    renders["lower_at_off"] = float(
        np.interp(FROZEN_T_OFF, renders["plot_time"], renders["plot_lower"])
    )

    parsed = run_checks(recording, metrics, renders)
    print()
    print("re-rendering summary (visualisation only, no number quoted):")
    print(
        f"  recording configuration  burn-in {RECORDING_BURN_IN:g} h, "
        f"dt {RECORDING_DT:g}: crossings "
        f"{np.round(renders['recording_switches'], 6).tolist()} h"
    )
    print(
        f"  plotted rendering        burn-in {RENDER_BURN_IN:g} h, "
        f"dt {RENDER_DT:g}: crossings "
        f"{np.round(renders['plot_switches'], 6).tolist()} h"
    )
    print(
        f"  stored canonical roots   {parsed['canonical'][0]:.6f}, "
        f"{parsed['canonical'][1]:.6f} h "
        f"(frozen {FROZEN_T_ON}, {FROZEN_T_OFF})"
    )
    print(
        "  the recording's own 48 h burn-in leaves a small transient, so its "
        "stored switch times sit "
        f"{(renders['recording_switches'][0] - parsed['canonical'][0]) * 3600:.1f} s "
        "and "
        f"{(renders['recording_switches'][1] - parsed['canonical'][1]) * 3600:.1f} s "
        "later than the limit-cycle roots; both give identical 0.5 h epoch labels"
    )
    print(f"  curve points plotted: {renders['plot_time'].size} per curve")
    print()

    fig = build_figure(recording, renders, parsed)
    saved = ts.save_figure(fig, OUTPUT_DIR, STEM)
    plt.close(fig)
    ts.report_outputs(
        saved,
        "figures/fig1_model_identifiability/figure1_model_identifiability.pdf",
    )


if __name__ == "__main__":
    main()
