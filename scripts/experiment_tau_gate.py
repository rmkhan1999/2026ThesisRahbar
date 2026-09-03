import numpy as np

from twopm.config import load_config
from twopm.hard_switch import simulate_hard_switch_from_config
from twopm.soft_gate import (
    match_transition_times,
    simulate_soft_gate_from_config,
    soft_transition_times,
)


def main() -> None:
    config = load_config("config/model.yaml")
    validation = config.section("validation")
    level = float(validation["transition_gate_level"])
    hard = simulate_hard_switch_from_config(config)
    print("tau_gate, transition MAE (h), soft sleep fraction")

    for tau_gate in validation["tau_gate_values"]:
        soft = simulate_soft_gate_from_config(
            config,
            tau_gate=float(tau_gate),
        )
        transitions = soft_transition_times(soft.time, soft.gate, level)
        match = match_transition_times(
            hard.switch_times,
            transitions,
        )
        error = np.mean(match.absolute_error)
        print(
            f"{float(tau_gate):.5f}, {error:.6f}, "
            f"{float(np.mean(soft.gate)):.6f}"
        )


if __name__ == "__main__":
    main()
