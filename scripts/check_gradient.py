from twopm.config import load_config
from twopm.gradients import check_sleep_fraction_gradient


def main() -> None:
    config = load_config("config/model.yaml")
    check = check_sleep_fraction_gradient(config)
    print(f"jax.grad:          {check.automatic:.8f}")
    print(f"finite difference: {check.finite_difference:.8f}")
    print(f"absolute error:    {check.absolute_error:.8e}")


if __name__ == "__main__":
    main()
