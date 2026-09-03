from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "sbc8_analysis.json"


def _corr(xs, ys):
    pr = pearsonr(xs, ys)
    sr = spearmanr(xs, ys)
    return {
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_rho": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }


def _residuals(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def main() -> None:
    rows = []
    for i in range(8):
        path = REPO / "docs" / "sbc" / f"replicate_{i:03d}.json"
        r = json.loads(path.read_text())
        d = r["diagnostics"]
        g = r["generating_parameters"]
        tau = np.asarray(r["posterior_draws"]["tau"], float)
        sites = [
            "tau",
            "chi_wake",
            "chi_sleep",
            "phase",
            "phase_z1",
            "excursion_fraction",
        ]
        n_extreme = int(
            sum(min(r["ranks"][s], 1.0 - r["ranks"][s]) <= 0.05 for s in sites)
        )
        rows.append(
            {
                "replicate": i,
                "seed": r["seed"],
                "ess_bulk_min": d["ess_bulk_min"],
                "ess_bulk": d["ess_bulk"],
                "divergences": d["divergences"],
                "tree_depth_saturation_fraction": d[
                    "tree_depth_saturation_fraction"
                ],
                "wall_hours": d["wall_seconds"] / 3600.0,
                "tau_true": g["tau"],
                "amplitude_true": g["amplitude"],
                "tau_posterior_mean": float(np.mean(tau)),
                "tau_posterior_sd": float(np.std(tau, ddof=1)),
                "contraction": r["contraction"],
                "ranks": {
                    k: r["ranks"][k]
                    for k in ("tau", "chi_wake", "chi_sleep", "phase")
                    if k in r["ranks"]
                },
                "n_extreme_ranks_leq_0p05": n_extreme,
            }
        )

    ess_min = np.asarray([row["ess_bulk_min"] for row in rows], float)
    ess_tau = np.asarray([row["ess_bulk"]["tau"] for row in rows], float)
    tau_true = np.asarray([row["tau_true"] for row in rows], float)
    amp = np.asarray([row["amplitude_true"] for row in rows], float)
    tau_sd = np.asarray([row["tau_posterior_sd"] for row in rows], float)
    rep7 = next(row for row in rows if row["replicate"] == 7)

    payload = {
        "date": "2026-08-11",
        "campaign": "SBC-8",
        "n_replicates": 8,
        "headline_replicate_7": {
            "ess_bulk_min": rep7["ess_bulk_min"],
            "tree_depth_saturation_fraction": rep7[
                "tree_depth_saturation_fraction"
            ],
            "tau_posterior_sd": rep7["tau_posterior_sd"],
            "contraction_tau": rep7["contraction"]["tau"],
            "rank_tau": rep7["ranks"]["tau"],
        },
        "summary": {
            "ess_bulk_min_mean": float(np.mean(ess_min)),
            "ess_bulk_min_range": [float(ess_min.min()), float(ess_min.max())],
            "total_divergences": int(sum(row["divergences"] for row in rows)),
            "mean_wall_hours": float(np.mean([row["wall_hours"] for row in rows])),
        },
        "correlations": {
            "ess_min_vs_generating_tau": _corr(ess_min, tau_true),
            "ess_tau_vs_generating_tau": _corr(ess_tau, tau_true),
            "ess_min_vs_generating_tau_partial_amplitude": _corr(
                _residuals(ess_min, amp), _residuals(tau_true, amp)
            ),
            "ess_tau_vs_generating_tau_partial_amplitude": _corr(
                _residuals(ess_tau, amp), _residuals(tau_true, amp)
            ),
            "ess_tau_vs_posterior_sd_tau_WRONG_TARGET": _corr(ess_tau, tau_sd),
        },
        "replicates": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
