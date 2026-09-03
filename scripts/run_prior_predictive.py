import jax
import numpy as np

from twopm.config import load_config
from twopm.plotting import plot_prior_predictive
from twopm.predictive import run_prior_predictive


def main() -> None:
    config = load_config("config/model.yaml")
    settings = config.section("prior_predictive")
    key = jax.random.PRNGKey(int(settings["seed"]))
    result = run_prior_predictive(key, config)
    output_path = config.section("figures")["prior_predictive"]["path"]
    destination = plot_prior_predictive(result, str(output_path))

    degenerate_fraction = float(np.mean(result.all_sleep | result.all_wake))
    physical_violation_fraction = float(np.mean(~result.physical_domain))
    minimum_margin = float(np.min(result.normalized_threshold_margin))
    outside_sleep_bounds = np.mean(
        (result.sleep_fraction < float(settings["sleep_fraction_min"]))
        | (result.sleep_fraction > float(settings["sleep_fraction_max"]))
    )
    print(f"draws: {result.sleep_fraction.size}")
    print(f"mean sleep fraction: {np.mean(result.sleep_fraction):.3f}")
    print(
        "entrained reference sleep fraction: "
        f"{float(settings['entrained_sleep_fraction_reference']):.3f}"
    )
    print(f"degenerate recordings: {degenerate_fraction:.1%}")
    print(f"sleep fraction outside bounds: {outside_sleep_bounds:.1%}")
    print(
        "physical threshold-domain violations: "
        f"{physical_violation_fraction:.1%}"
    )
    print(f"minimum normalized threshold margin: {minimum_margin:.6f}")
    print(f"Saved {destination}")
    if physical_violation_fraction > float(
        settings["max_physical_domain_violation_fraction"]
    ):
        raise RuntimeError("prior draws left the physical threshold domain")
    if minimum_margin <= float(
        settings["minimum_normalized_threshold_margin"]
    ):
        raise RuntimeError("prior threshold margin did not remain positive")


if __name__ == "__main__":
    main()
