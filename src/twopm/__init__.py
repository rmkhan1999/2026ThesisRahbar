import jax

jax.config.update("jax_enable_x64", True)

from twopm.analytic import (
    sleep_duration,
    thresholds_from_mean_gap,
    wake_duration,
)
from twopm.circadian import circadian_thresholds
from twopm.config import ProjectConfig, load_config
from twopm.gradients import (
    GradientCheck,
    check_sleep_fraction_gradient,
    finite_difference_gradient,
    sleep_fraction,
    sleep_fraction_for_chi_sleep,
)
from twopm.generative import (
    GeneratedRecording,
    Parameters,
    generate_recording,
    sample_parameters,
    standard_parameters,
)
from twopm.hard_switch import (
    BoutDurations,
    HardSwitchResult,
    bout_durations,
    simulate_hard_switch,
    simulate_hard_switch_from_config,
)
from twopm.inference import (
    constrained_physical_parameters,
    free_running_period,
    numpyro_model,
    parameter_mapping_to_vector,
    prior_distribution,
    sampled_parameter_names,
)
from twopm.likelihood import (
    LIKELIHOOD_PARAMETER_NAMES,
    likelihood_parameter_names,
    log_likelihood,
    standard_likelihood_parameters,
)
from twopm.observation import sample_observations, sleep_probabilities
from twopm.posterior_summaries import (
    EntrainedTransition,
    PosteriorTransitions,
    entrained_transition_equations,
    posterior_transition_times,
    solve_entrained_transition,
    variance_contraction,
)
from twopm.plotting import (
    plot_hard_switch_trajectory,
    plot_prior_predictive,
    plot_smoothing_convergence,
)
from twopm.predictive import (
    PriorPredictiveResult,
    RecordingSummary,
    recording_summary,
    run_prior_predictive,
    threshold_domain_margins,
)
from twopm.soft_gate import (
    ConvergenceResult,
    SmoothingCalibrationResult,
    SoftGateResult,
    TransitionMatch,
    cartesian_circadian_displacement,
    calibrate_smoothing_grid,
    circadian_amplitude_phase,
    circadian_coefficients,
    gate_offset,
    gate_target,
    match_transition_times,
    simulate_soft_gate,
    simulate_soft_gate_from_config,
    smoothing_convergence_study,
    soft_gate_vector_field,
    soft_transition_times,
)

__version__ = "0.1.0"

__all__ = [
    "BoutDurations",
    "ConvergenceResult",
    "GradientCheck",
    "GeneratedRecording",
    "HardSwitchResult",
    "EntrainedTransition",
    "LIKELIHOOD_PARAMETER_NAMES",
    "ProjectConfig",
    "Parameters",
    "PriorPredictiveResult",
    "PosteriorTransitions",
    "RecordingSummary",
    "SmoothingCalibrationResult",
    "SoftGateResult",
    "TransitionMatch",
    "bout_durations",
    "cartesian_circadian_displacement",
    "calibrate_smoothing_grid",
    "check_sleep_fraction_gradient",
    "circadian_thresholds",
    "circadian_amplitude_phase",
    "circadian_coefficients",
    "constrained_physical_parameters",
    "entrained_transition_equations",
    "free_running_period",
    "load_config",
    "likelihood_parameter_names",
    "log_likelihood",
    "match_transition_times",
    "finite_difference_gradient",
    "generate_recording",
    "gate_offset",
    "gate_target",
    "plot_hard_switch_trajectory",
    "plot_prior_predictive",
    "plot_smoothing_convergence",
    "posterior_transition_times",
    "parameter_mapping_to_vector",
    "prior_distribution",
    "numpyro_model",
    "sample_observations",
    "sample_parameters",
    "sampled_parameter_names",
    "recording_summary",
    "run_prior_predictive",
    "simulate_hard_switch",
    "simulate_hard_switch_from_config",
    "simulate_soft_gate",
    "simulate_soft_gate_from_config",
    "solve_entrained_transition",
    "sleep_duration",
    "sleep_fraction",
    "sleep_fraction_for_chi_sleep",
    "sleep_probabilities",
    "smoothing_convergence_study",
    "soft_gate_vector_field",
    "soft_transition_times",
    "standard_parameters",
    "standard_likelihood_parameters",
    "thresholds_from_mean_gap",
    "threshold_domain_margins",
    "variance_contraction",
    "wake_duration",
]
