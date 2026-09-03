import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class ProjectConfig:

    data: Mapping[str, Any]
    source: Path

    def section(self, name: str) -> Mapping[str, Any]:
        section = self.data.get(name)
        if not isinstance(section, Mapping):
            raise KeyError(f"missing configuration section: {name}")
        return section


def load_config(path: str | Path) -> ProjectConfig:
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    if not isinstance(data, Mapping):
        raise ValueError("configuration root must be a mapping")

    config = ProjectConfig(data=data, source=source)
    model = config.section("model")
    fixed = config.section("fixed")
    config.section("initial_state")
    config.section("hard_switch")
    soft_gate = config.section("soft_gate")
    config.section("observation")
    priors = config.section("priors")
    inference = config.section("inference")
    designs = config.section("designs")

    lower = float(model["lower_base"])
    upper = float(model["upper_base"])
    mu = float(model["mu"])
    if not 0 < lower < upper < mu:
        raise ValueError(
            "model thresholds must satisfy 0 < lower_base < upper_base < mu"
        )
    if float(model["chi_sleep"]) <= 0 or float(model["chi_wake"]) <= 0:
        raise ValueError("model time constants must be positive")
    threshold_mean = float(fixed["threshold_mean"])
    if not math.isclose(threshold_mean, (lower + upper) / 2):
        raise ValueError(
            "fixed threshold_mean must equal the midpoint of model thresholds"
        )
    if 2 * threshold_mean >= mu:
        raise ValueError(
            "constrained excursion requires 2 * threshold_mean < mu"
        )
    if not 0 < float(soft_gate["p0"]) < 1:
        raise ValueError("soft-gate p0 must lie strictly between zero and one")
    if float(soft_gate["k"]) <= 0 or float(soft_gate["tau_gate"]) <= 0:
        raise ValueError("soft-gate k and tau_gate must be positive")
    if soft_gate["integration_mode"] not in {"adaptive", "fixed"}:
        raise ValueError("soft-gate integration_mode must be adaptive or fixed")
    if float(soft_gate["fixed_step_size"]) <= 0:
        raise ValueError("soft-gate fixed_step_size must be positive")
    if soft_gate["adjoint_mode"] not in {"recursive_checkpoint", "direct"}:
        raise ValueError(
            "soft-gate adjoint_mode must be recursive_checkpoint or direct"
        )
    inferred = tuple(inference["parameters"])
    if set(priors) != set(inferred):
        raise ValueError(
            "configured priors must exactly match inferred parameters"
        )
    default_design = str(inference["default_design"])
    if default_design not in designs:
        raise ValueError("default inference design is not configured")
    for name, design in designs.items():
        if not isinstance(design, Mapping):
            raise ValueError(f"design {name} must be a mapping")
        if float(design["duration"]) <= 0 or float(design["burn_in_hours"]) < 0:
            raise ValueError(f"design {name} has invalid time settings")
        if float(design["reference_amplitude"]) < 0:
            raise ValueError(
                f"design {name} reference amplitude must be nonnegative"
            )
        constrained = design["constrained_prior"]
        concentrations = (
            "excursion_concentration1",
            "excursion_concentration0",
            "amplitude_concentration1",
            "amplitude_concentration0",
        )
        if any(float(constrained[key]) <= 0 for key in concentrations):
            raise ValueError(
                f"design {name} constrained-prior concentrations must be positive"
            )

    return config
