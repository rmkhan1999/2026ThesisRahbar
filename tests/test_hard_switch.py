import numpy as np
import pytest

from twopm.analytic import sleep_duration, wake_duration
from twopm.hard_switch import bout_durations, simulate_hard_switch


def test_hard_switch_bouts_match_closed_form_within_step_size():
    dt = 0.01
    chi_sleep = 4.2
    chi_wake = 18.2
    mu = 1.0
    upper = 0.67
    lower = 0.17

    result = simulate_hard_switch(
        duration=120.0,
        dt=dt,
        chi_sleep=chi_sleep,
        chi_wake=chi_wake,
        mu=mu,
        upper=upper,
        lower=lower,
        initial_pressure=upper,
        initially_asleep=True,
        amplitude=0.0,
        phase=0.0,
        period=24.0,
        start_time=0.0,
    )
    bouts = bout_durations(result.switch_times, result.switch_states)

    expected_sleep = sleep_duration(chi_sleep, upper, lower)
    expected_wake = wake_duration(chi_wake, mu, upper, lower)

    assert bouts.sleep.size >= 3
    assert bouts.wake.size >= 3
    assert np.mean(bouts.sleep) == pytest.approx(expected_sleep, abs=dt)
    assert np.mean(bouts.wake) == pytest.approx(expected_wake, abs=dt)


def test_switches_alternate_between_wake_and_sleep():
    result = simulate_hard_switch(
        duration=80.0,
        dt=0.01,
        chi_sleep=4.2,
        chi_wake=18.2,
        mu=1.0,
        upper=0.67,
        lower=0.17,
        initial_pressure=0.67,
        initially_asleep=True,
        amplitude=0.0,
        phase=0.0,
        period=24.0,
        start_time=0.0,
    )

    assert result.switch_states.size > 2
    assert np.all(result.switch_states[1:] != result.switch_states[:-1])
    assert np.all(np.diff(result.switch_times) > 0)
