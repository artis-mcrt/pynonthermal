"""Check the charge-neutral root search for a free electron density near the lower limit of double precision."""

import math

import pytest

import pynonthermal.ionbalance


def test_upper_bracket_is_the_root() -> None:
    # the first lower trial 1e-270 * 1e-60 is zero in double precision, and the function raised an error
    assert pynonthermal.ionbalance.solve_charge_neutral_n_e(0.0, lambda _n_e: 1e-270, 0.0, 1e-270) == 1e-270


@pytest.mark.parametrize("root", [1e-280, 1e-300, 1e-310])
def test_root_inside_a_bracket_with_a_first_trial_that_underflows(root: float) -> None:
    # the root is between the smallest positive double and the upper bracket
    n_e = pynonthermal.ionbalance.solve_charge_neutral_n_e(0.0, lambda _n_e: root, 0.0, 1e-270)
    # math.isclose() has no absolute tolerance by default
    assert math.isclose(n_e, root, rel_tol=1e-9)


def test_no_charge_keeps_its_error() -> None:
    with pytest.raises(ValueError, match="free electron density is zero"):
        pynonthermal.ionbalance.solve_charge_neutral_n_e(0.0, lambda _n_e: 0.0, 0.0, 1e-270)


def test_ordinary_root_does_not_change() -> None:
    # x^2 n / (1 - x) = K for one element with two stages
    n_elem, ratio_coeff = 1e8, 1e6
    n_e = pynonthermal.ionbalance.solve_charge_neutral_n_e_ratios(0.0, [(n_elem, 1, [ratio_coeff])])
    expected = 0.5 * (-ratio_coeff + math.sqrt(ratio_coeff**2 + 4.0 * ratio_coeff * n_elem))
    assert math.isclose(n_e, expected, rel_tol=1e-12)
