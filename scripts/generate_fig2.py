from twopm.config import load_config
from twopm.hard_switch import simulate_hard_switch_from_config
from twopm.plotting import plot_hard_switch_trajectory


def main() -> None:
    config = load_config("config/model.yaml")
    figure_settings = config.section("figures")["fig2"]
    result = simulate_hard_switch_from_config(
        config,
        duration=float(figure_settings["duration"]),
    )
    destination = plot_hard_switch_trajectory(
        result,
        str(figure_settings["path"]),
    )
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
