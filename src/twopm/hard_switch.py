from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from twopm.circadian import circadian_thresholds
from twopm.config import ProjectConfig


@dataclass(frozen=True)
class HardSwitchResult:

    time: NDArray[np.float64]
    pressure: NDArray[np.float64]
    asleep: NDArray[np.bool_]
    upper_threshold: NDArray[np.float64]
    lower_threshold: NDArray[np.float64]
    switch_times: NDArray[np.float64]
    switch_states: NDArray[np.bool_]


@dataclass(frozen=True)
class BoutDurations:

    sleep: NDArray[np.float64]
    wake: NDArray[np.float64]


def simulate_hard_switch(
    *,
    duration: float,
    dt: float,
    chi_sleep: float,
    chi_wake: float,
    mu: float,
    upper: float,
    lower: float,
    initial_pressure: float,
    initially_asleep: bool,
    amplitude: float,
    phase: float,
    period: float,
    start_time: float,
) -> HardSwitchResult:
    if duration <= 0 or dt <= 0:
        raise ValueError("duration and dt must be positive")
    if chi_sleep <= 0 or chi_wake <= 0:
        raise ValueError("time constants must be positive")
    if not 0 < lower < upper < mu:
        raise ValueError("thresholds must satisfy 0 < lower < upper < mu")

    step_count = int(np.floor(duration / dt))
    time = start_time + dt * np.arange(step_count + 1, dtype=np.float64)
    upper_threshold, lower_threshold = circadian_thresholds(
        time,
        upper_base=upper,
        lower_base=lower,
        amplitude=amplitude,
        phase=phase,
        period=period,
    )
    pressure = np.empty(step_count + 1, dtype=np.float64)
    asleep = np.empty(step_count + 1, dtype=np.bool_)
    switch_times: list[float] = []
    switch_states: list[bool] = []

    pressure[0] = initial_pressure
    asleep[0] = initially_asleep

    for index in range(step_count):
        current_pressure = pressure[index]
        current_asleep = bool(asleep[index])

        if current_asleep:
            derivative = -current_pressure / chi_sleep
        else:
            derivative = (mu - current_pressure) / chi_wake

        next_pressure = current_pressure + dt * derivative
        next_asleep = current_asleep

        if current_asleep and next_pressure <= lower_threshold[index + 1]:
            next_asleep = False
            switch_times.append(float(time[index + 1]))
            switch_states.append(next_asleep)
        elif (
            not current_asleep
            and next_pressure >= upper_threshold[index + 1]
        ):
            next_asleep = True
            switch_times.append(float(time[index + 1]))
            switch_states.append(next_asleep)

        pressure[index + 1] = next_pressure
        asleep[index + 1] = next_asleep

    return HardSwitchResult(
        time=time,
        pressure=pressure,
        asleep=asleep,
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
        switch_times=np.asarray(switch_times, dtype=np.float64),
        switch_states=np.asarray(switch_states, dtype=np.bool_),
    )


def bout_durations(
    switch_times: NDArray[np.float64],
    switch_states: NDArray[np.bool_],
) -> BoutDurations:
    if switch_times.ndim != 1 or switch_states.ndim != 1:
        raise ValueError("switch arrays must be one-dimensional")
    if switch_times.size != switch_states.size:
        raise ValueError("switch arrays must have equal lengths")
    if np.any(np.diff(switch_times) <= 0):
        raise ValueError("switch times must be strictly increasing")

    durations = np.diff(switch_times)
    interval_states = switch_states[:-1]
    return BoutDurations(
        sleep=durations[interval_states],
        wake=durations[~interval_states],
    )


def simulate_hard_switch_from_config(
    config: ProjectConfig,
    *,
    duration: float | None = None,
    phase: float | None = None,
    amplitude: float | None = None,
) -> HardSwitchResult:
    model = config.section("model")
    settings = config.section("hard_switch")
    initial = config.section("initial_state")
    configured_duration = (
        float(settings["duration"]) if duration is None else duration
    )
    configured_phase = float(model["phase"]) if phase is None else phase
    configured_amplitude = (
        float(model["circadian_amplitude"])
        if amplitude is None
        else amplitude
    )

    return simulate_hard_switch(
        duration=configured_duration,
        dt=float(settings["dt"]),
        chi_sleep=float(model["chi_sleep"]),
        chi_wake=float(model["chi_wake"]),
        mu=float(model["mu"]),
        upper=float(model["upper_base"]),
        lower=float(model["lower_base"]),
        initial_pressure=float(initial["pressure"]),
        initially_asleep=bool(initial["hard_asleep"]),
        amplitude=configured_amplitude,
        phase=configured_phase,
        period=float(model["circadian_period"]),
        start_time=float(settings["start_time"]),
    )
