from twopm.config import load_config
from twopm.plotting import plot_smoothing_convergence
from twopm.soft_gate import smoothing_convergence_study


def main() -> None:
    config = load_config("config/model.yaml")
    result = smoothing_convergence_study(config)
    output_path = config.section("figures")["smoothing_convergence"]["path"]
    destination = plot_smoothing_convergence(result, str(output_path))

    for k, error in zip(
        result.k_values,
        result.mean_absolute_error,
    ):
        print(f"k={k:g}: mean transition error={error:.6f} h")
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
