import numpy as np

from twopm.circadian import circadian_thresholds
from twopm.hard_switch import simulate_hard_switch


def _circadian_simulation(phase):
    return simulate_hard_switch(
        duration=720.0,
        dt=0.01,
        chi_sleep=4.2,
        chi_wake=18.2,
        mu=1.0,
        upper=0.67,
        lower=0.17,
        initial_pressure=0.67,
        initially_asleep=True,
        amplitude=0.12,
        phase=phase,
        period=24.0,
        start_time=0.0,
    )


def test_circadian_threshold_gap_stays_constant():
    time = np.linspace(0.0, 24.0, 101)
    upper, lower = circadian_thresholds(
        time,
        upper_base=0.67,
        lower_base=0.17,
        amplitude=0.12,
        phase=3.0,
        period=24.0,
    )

    assert np.allclose(upper - lower, 0.50)


def test_circadian_model_produces_one_sleep_onset_per_day():
    result = _circadian_simulation(phase=0.0)
    late_onsets = result.switch_times[
        result.switch_states & (result.switch_times >= 552.0)
    ]

    assert late_onsets.size == 7
    assert np.allclose(np.diff(late_onsets), 24.0, atol=0.01)


def test_phase_shift_is_identical_after_aligning_sleep_onset():
    lark = _circadian_simulation(phase=0.0)
    owl = _circadian_simulation(phase=6.0)
    lark_onset = lark.switch_times[lark.switch_states][-2]
    owl_onset = owl.switch_times[owl.switch_states][-2]
    lark_start = np.searchsorted(lark.time, lark_onset)
    owl_start = np.searchsorted(owl.time, owl_onset)
    day_points = int(24.0 / 0.01) + 1

    lark_slice = slice(lark_start, lark_start + day_points)
    owl_slice = slice(owl_start, owl_start + day_points)

    assert owl_onset - lark_onset == 6.0
    assert np.array_equal(lark.asleep[lark_slice], owl.asleep[owl_slice])
    assert np.allclose(
        lark.pressure[lark_slice],
        owl.pressure[owl_slice],
        atol=1e-12,
    )
    assert np.allclose(
        lark.upper_threshold[lark_slice],
        owl.upper_threshold[owl_slice],
        atol=1e-12,
    )
