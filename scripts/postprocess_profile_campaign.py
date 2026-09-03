from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUMMARY = REPO / "docs" / "profile_campaign.json"
POINTS_DIR = REPO / "docs" / "profile_campaign"
REL_TOL = 0.05


def main() -> None:
    if SUMMARY.exists():
        payload = json.loads(SUMMARY.read_text())
        profiles = payload.get("profiles", {})
        mle_nll = float(payload["mle_negative_log_likelihood"])
    else:
        profiles = {}
        for path in sorted(POINTS_DIR.glob("*.json")):
            point = json.loads(path.read_text())
            name = point["fixed_parameter"]
            profiles.setdefault(name, {"points": []})["points"].append(point)
        mle_nll = 0.097652
        payload = {
            "mle_negative_log_likelihood": mle_nll,
            "profiles": profiles,
            "note": "assembled from per-point JSON before summary existed",
        }

    nlls = [
        float(point["negative_log_likelihood"])
        for profile in profiles.values()
        for point in profile["points"]
    ]
    if not nlls:
        raise SystemExit("no profile points found")
    best = min(nlls)
    excess = (mle_nll - best) / max(abs(best), 1e-12)
    under = excess > REL_TOL

    per_grid = {}
    for name, profile in profiles.items():
        grid_nlls = [float(p["negative_log_likelihood"]) for p in profile["points"]]
        grid_best = min(grid_nlls)
        below_mle = sum(1 for v in grid_nlls if v < mle_nll - 1e-12)
        per_grid[name] = {
            "n_points": len(grid_nlls),
            "grid_best_nll": grid_best,
            "n_points_below_unrestricted_mle": below_mle,
            "fraction_below_unrestricted_mle": below_mle / len(grid_nlls),
        }
        for point in profile["points"]:
            point["delta_nll_from_campaign_best"] = (
                float(point["negative_log_likelihood"]) - best
            )

    payload["campaign_best_negative_log_likelihood"] = best
    payload["mle_relative_excess_over_campaign_best"] = excess
    payload["unrestricted_mle_underconverged"] = under
    payload["raue_baseline"] = "campaign_best_across_all_profiled_grids"
    payload["mle_relative_tol"] = REL_TOL
    payload["per_grid_vs_unrestricted_mle"] = per_grid
    payload["profiles"] = profiles
    SUMMARY.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {SUMMARY}")
    print(f"campaign_best={best:.6f} mle={mle_nll:.6f} excess={excess:.4f} underconverged={under}")
    for name, stats in per_grid.items():
        print(
            f"  {name}: grid_best={stats['grid_best_nll']:.6f} "
            f"below_mle={stats['n_points_below_unrestricted_mle']}/"
            f"{stats['n_points']}"
        )


if __name__ == "__main__":
    main()
