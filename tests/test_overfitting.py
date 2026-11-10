import math

import numpy as np

from sample.overfitting import cscv_pbo, deflated_sharpe, expected_max_sharpe


def test_cscv_complete_ties_use_average_rank_and_lambda_zero():
    # All N strategies tie in every split: average rank -> omega=1/2,
    # lambda=log(1)=0, and the specified lambda <= 0 rule counts each split.
    result = cscv_pbo(np.zeros((16, 3)), segments=4)
    assert result["combinations"] == 6
    assert result["pbo"] == 1.0
    assert np.array_equal(result["lambdas"], np.zeros(6))


def test_cscv_consistently_dominant_strategy_has_zero_pbo():
    returns = np.zeros((16, 3))
    returns[:, 0] = 0.01 + np.tile([0.001, -0.001], 8)
    result = cscv_pbo(returns, segments=4)
    assert result["pbo"] == 0.0


def test_dsr_normal_case_matches_hand_calculated_standard_normal_cdf():
    # z = (0.10 - 0.05)*sqrt(100)/sqrt(1 + 0.10^2/2) = 0.498756...
    z = 0.5 / math.sqrt(1.005)
    expected = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    assert math.isclose(deflated_sharpe(0.10, 0.05, 101, 0.0, 3.0), expected, abs_tol=1e-14)


def test_expected_max_sharpe_single_trial_is_zero():
    assert expected_max_sharpe(np.array([0.25])) == 0.0
