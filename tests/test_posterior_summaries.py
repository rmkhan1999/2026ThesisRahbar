import numpy as np

from twopm.posterior_summaries import (
    posterior_transition_times,
    solve_entrained_transition,
    variance_contraction,
)


def test_canonical_entrained_root_matches_verified_transition_times():
    phase = 8.0
    angle = 2 * np.pi * phase / 24.0
    result = solve_entrained_transition(
        c1=0.12 * np.cos(angle),
        c2=0.12 * np.sin(angle),
        chi_sleep=4.2,
        chi_wake=18.2,
        threshold_gap=0.5,
    )
    assert result.converged
    np.testing.assert_allclose(result.onset, 15.4580834, atol=1e-6)
    np.testing.assert_allclose(result.offset, 23.3977647, atol=1e-6)


def test_failed_draws_are_nan_and_counted():
    posterior = {
        "c1": np.asarray([[0.12, np.nan]]),
        "c2": np.asarray([[0.0, 0.0]]),
        "chi_sleep": np.asarray([[4.2, 4.2]]),
        "chi_wake": np.asarray([[18.2, 18.2]]),
        "threshold_gap": np.asarray([[0.5, 0.5]]),
    }
    result = posterior_transition_times(posterior)
    assert result.total == 2
    assert result.failed == 1
    assert np.isnan(result.onset[0, 1])
    assert result.failure_reasons == {"non_finite_parameter": 1}


def test_variance_contraction_uses_prior_and_posterior_sample_variances():
    prior = np.asarray((-2.0, 0.0, 2.0))
    posterior = np.asarray((-1.0, 0.0, 1.0))
    assert variance_contraction(prior, posterior) == 0.75
