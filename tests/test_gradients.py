import pytest

from twopm.config import load_config
from twopm.gradients import (
    check_sleep_fraction_gradient,
    finite_difference_gradient,
)


def test_centered_finite_difference_on_quadratic():
    derivative = finite_difference_gradient(
        lambda value: value**2,
        value=3.0,
        step=0.0001,
    )

    assert derivative == pytest.approx(6.0, abs=1e-8)


def test_jax_gradient_agrees_with_finite_difference():
    config = load_config("config/model.yaml")
    validation = config.section("validation")
    decimal_places = int(validation["gradient_decimal_places"])
    tolerance = 0.5 * 10 ** (-decimal_places)
    check = check_sleep_fraction_gradient(config)

    assert check.automatic == pytest.approx(
        check.finite_difference,
        abs=tolerance,
    )
