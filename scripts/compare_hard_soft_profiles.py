from __future__ import annotations

import json
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
HARD = REPOSITORY / "docs" / "profile_hard_chi_wake.json"
SOFT = REPOSITORY / "docs" / "profile_chi_wake_partial.json"
OUT = REPOSITORY / "docs" / "hard_vs_soft_profile.json"
RAUE = 2.0
PRE_CHI_LO = 15.0
PRE_CHI_HI = 26.0
PRE_TON_MAX_MIN = 15.0


def _soft_points() -> list[dict]:
    raw = json.loads(SOFT.read_text())
    return [
        {
            "chi_wake": p["chi_wake"],
            "nll": p["nll"],
            "t_on": p.get("t_on"),
            "tau": p.get("tau"),
        }
        for p in raw["points"]
    ]


def _hard_points() -> list[dict]:
    raw = json.loads(HARD.read_text())
    return [
        {
            "chi_wake": p["fixed_value"],
            "nll": p["negative_log_likelihood"],
            "t_on": p.get("t_on"),
            "tau": p.get("tau"),
        }
        for p in raw["points"]
    ]


def _band(
    points: list[dict],
    *,
    chi_lo: float | None = None,
    chi_hi: float | None = None,
    require_t_on: bool = False,
) -> dict:
    usable = [
        p
        for p in points
        if np.isfinite(p["nll"]) and (not require_t_on or p.get("t_on") is not None)
    ]
    if not usable:
        return {"error": "no usable points"}
    mle = float(min(p["nll"] for p in usable))
    mle_global = float(min(p["nll"] for p in points if np.isfinite(p["nll"])))
    in_band = []
    for p in points:
        if not np.isfinite(p["nll"]):
            continue
        if p["nll"] - mle_global >= RAUE:
            continue
        c = float(p["chi_wake"])
        if chi_lo is not None and c < chi_lo - 1e-12:
            continue
        if chi_hi is not None and c > chi_hi + 1e-12:
            continue
        if require_t_on and p.get("t_on") is None:
            continue
        in_band.append(p)
    if not in_band:
        return {
            "mle_nll": mle_global,
            "raue_threshold": RAUE,
            "n_points_in_band": 0,
            "chi_wake_band_hours": None,
            "chi_wake_band_width_hours": None,
            "t_on_range_minutes": None,
            "tau_range_hours": None,
            "max_delta_nll": None,
        }
    chi = np.asarray([p["chi_wake"] for p in in_band], dtype=float)
    tons = np.asarray(
        [p["t_on"] for p in in_band if p.get("t_on") is not None], dtype=float
    )
    taus = np.asarray(
        [p["tau"] for p in in_band if p.get("tau") is not None], dtype=float
    )
    nlls = np.asarray([p["nll"] for p in in_band], dtype=float)
    return {
        "mle_nll": mle_global,
        "raue_threshold": RAUE,
        "n_points_in_band": int(len(in_band)),
        "chi_wake_band_hours": [float(chi.min()), float(chi.max())],
        "chi_wake_band_width_hours": float(chi.max() - chi.min()),
        "t_on_range_minutes": (
            float((tons.max() - tons.min()) * 60.0) if tons.size else None
        ),
        "tau_range_hours": (
            float(taus.max() - taus.min()) if taus.size else None
        ),
        "max_delta_nll": float(nlls.max() - mle_global),
    }


def _summarise(points: list[dict], label: str) -> dict:
    full = _band(points)
    pre = _band(points, chi_lo=PRE_CHI_LO, chi_hi=PRE_CHI_HI, require_t_on=True)
    covers = (
        full.get("chi_wake_band_hours") is not None
        and full["chi_wake_band_hours"][0] <= PRE_CHI_LO + 1e-9
        and full["chi_wake_band_hours"][1] >= PRE_CHI_HI - 1e-9
    )
    ton_ok = (
        pre.get("t_on_range_minutes") is not None
        and pre["t_on_range_minutes"] < PRE_TON_MAX_MIN
    )
    dnll_ok = (
        pre.get("max_delta_nll") is not None and pre["max_delta_nll"] < RAUE
    )
    return {
        "label": label,
        "full_raue_band": full,
        "preregistered_15_to_26": pre,
        "covers_15_to_26_under_raue": covers,
        "preregistered_flat": bool(covers and ton_ok and dnll_ok),
        "chi_wake_band_hours": full.get("chi_wake_band_hours"),
        "chi_wake_band_width_hours": full.get("chi_wake_band_width_hours"),
        "t_on_range_minutes_full_band": full.get("t_on_range_minutes"),
        "t_on_range_minutes_15_to_26": pre.get("t_on_range_minutes"),
        "tau_range_hours_full_band": full.get("tau_range_hours"),
        "tau_range_hours_15_to_26": pre.get("tau_range_hours"),
        "max_delta_nll_15_to_26": pre.get("max_delta_nll"),
    }


def main() -> None:
    if not HARD.exists():
        raise FileNotFoundError(HARD)
    if not SOFT.exists():
        raise FileNotFoundError(SOFT)
    hard = _summarise(_hard_points(), "hard_generated")
    soft = _summarise(_soft_points(), "soft_generated")

    hard_flat = hard["preregistered_flat"]
    soft_flat = soft["preregistered_flat"] or (
        soft.get("chi_wake_band_width_hours") is not None
        and soft["chi_wake_band_width_hours"] >= 10
        and soft.get("max_delta_nll_15_to_26") is not None
        and soft["max_delta_nll_15_to_26"] < RAUE
        and soft.get("t_on_range_minutes_15_to_26") is not None
        and soft["t_on_range_minutes_15_to_26"] < PRE_TON_MAX_MIN
        and soft.get("covers_15_to_26_under_raue")
    )

    hw = hard.get("chi_wake_band_width_hours")
    sw = soft.get("chi_wake_band_width_hours")
    if hard_flat and soft_flat:
        if hw is not None and sw is not None and abs(hw - sw) < 5:
            verdict = "both_flat_similar"
        else:
            verdict = "hard_flat_narrower_or_wider"
    elif hard_flat and not soft_flat:
        verdict = "hard_flat_soft_not"
    elif not hard_flat and soft_flat:
        verdict = "hard_curved"
    else:
        verdict = "neither_clearly_flat"

    payload = {
        "date": "2026-08-10",
        "soft_source": str(SOFT.relative_to(REPOSITORY)),
        "hard_source": str(HARD.relative_to(REPOSITORY)),
        "soft": soft,
        "hard": hard,
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
