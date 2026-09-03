import pytest

from twopm.analytic import (
    sleep_duration,
    thresholds_from_mean_gap,
    wake_duration,
)


def test_standard_durations_match_reference_values():
    sleep = sleep_duration(chi_sleep=4.2, upper=0.67, lower=0.17)
    wake = wake_duration(
        chi_wake=18.2,
        mu=1.0,
        upper=0.67,
        lower=0.17,
    )

    assert sleep == pytest.approx(5.760, abs=0.001)
    assert wake == pytest.approx(16.787, abs=0.001)


def test_thresholds_from_mean_gap_returns_ordered_pair():
    upper, lower = thresholds_from_mean_gap(mean=0.42, gap=0.50)

    assert upper == pytest.approx(0.67)
    assert lower == pytest.approx(0.17)


@pytest.mark.parametrize("gap", [0.0, -0.1])
def test_thresholds_from_mean_gap_rejects_nonpositive_gap(gap):
    with pytest.raises(ValueError, match="gap must be positive"):
        thresholds_from_mean_gap(mean=0.42, gap=gap)
