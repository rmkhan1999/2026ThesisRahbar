import numpy as np

from twopm.config import load_config
from twopm.soft_gate import calibrate_smoothing_grid


def main() -> None:
    config = load_config("config/model.yaml")
    result = calibrate_smoothing_grid(config)
    soft = config.section("soft_gate")
    budget = float(
        config.section("validation")["transition_bias_budget_hours"]
    )

    print("p0,k,transition_mae_hours,solver_steps,unmatched_hard,unmatched_soft")
    for p0_index, p0 in enumerate(result.p0_values):
        for k_index, k in enumerate(result.k_values):
            print(
                f"{p0:g},{k:g},"
                f"{result.mean_absolute_error[p0_index, k_index]:.6f},"
                f"{result.solver_steps[p0_index, k_index]},"
                f"{result.unmatched_reference[p0_index, k_index]},"
                f"{result.unmatched_candidate[p0_index, k_index]}"
            )

    feasible = (
        np.isfinite(result.mean_absolute_error)
        & (result.mean_absolute_error < budget)
        & (result.unmatched_reference == 0)
        & (result.unmatched_candidate == 0)
    )
    if not np.any(feasible):
        print(f"No grid point meets the {budget:g} h error budget.")
        return

    feasible_indices = np.argwhere(feasible)
    largest_p0 = np.max(result.p0_values[feasible_indices[:, 0]])
    p0_rows = feasible_indices[
        result.p0_values[feasible_indices[:, 0]] == largest_p0
    ]
    chosen_index = p0_rows[np.argmin(result.k_values[p0_rows[:, 1]])]
    p0_index, k_index = map(int, chosen_index)
    minimum_index = int(
        np.nanargmin(result.mean_absolute_error[p0_index])
    )
    print(
        "Error-budget-only candidate: "
        f"p0={result.p0_values[p0_index]:g}, "
        f"k={result.k_values[k_index]:g}, "
        "transition MAE="
        f"{result.mean_absolute_error[p0_index, k_index]:.6f} h, "
        f"solver steps={result.solver_steps[p0_index, k_index]}"
    )
    print(
        "Oracle-MAE minimum for chosen p0: "
        f"k={result.k_values[minimum_index]:g}, "
        f"transition MAE="
        f"{result.mean_absolute_error[p0_index, minimum_index]:.6f} h"
    )
    print(
        "Final selection must also pass the multi-day flat-threshold "
        "trajectory check."
    )
    configured_p0_index = int(
        np.where(result.p0_values == float(soft["p0"]))[0][0]
    )
    configured_k_index = int(
        np.where(result.k_values == float(soft["k"]))[0][0]
    )
    print(
        "Configured pair: "
        f"p0={float(soft['p0']):g}, k={float(soft['k']):g}, "
        "transition MAE="
        f"{result.mean_absolute_error[configured_p0_index, configured_k_index]:.6f} h, "
        "solver steps="
        f"{result.solver_steps[configured_p0_index, configured_k_index]}"
    )


if __name__ == "__main__":
    main()
