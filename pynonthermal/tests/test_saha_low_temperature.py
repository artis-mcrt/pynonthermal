"""Check the Saha coefficient where its Boltzmann factor alone is below the range of double precision."""

import math

import pynonthermal.ionbalance
from pynonthermal.ionbalance import K_B
from pynonthermal.ionbalance import SAHA_CONST

# the NIST ionisation potential of hydrogen [eV]
IONPOT_H_EV = 13.598434599702


def test_saha_factor_at_a_low_temperature() -> None:
    # exp(-chi / (k_B T)) is 1.9e-327 at 210 K, below the smallest double of 4.9e-324. The complete
    # coefficient is in the range, and the product of the factors gave zero.
    assert math.exp(-IONPOT_H_EV / K_B / 210.0) == 0.0
    saha_factor = pynonthermal.ionbalance.get_saha_factor(210.0, IONPOT_H_EV, 2.0, 1.0)
    # math.isclose() has no absolute tolerance by default. pytest.approx() has one of 1e-12, which accepts zero.
    assert math.isclose(saha_factor, 3.2970e-308, rel_tol=1e-4)

    # the charge-neutral ionised fraction x of hydrogen satisfies x^2 n / (1 - x) = K, so x is sqrt(K / n)
    n_elem = 1e10
    fractions = pynonthermal.ionbalance.get_saha_ion_fractions(
        1, [1, 2], 210.0, n_elem=n_elem, partfuncs={1: 2.0, 2: 1.0}
    )
    # the coefficient is near the lower limit of the normal doubles, so the root search loses digits
    assert math.isclose(fractions[2], math.sqrt(saha_factor / n_elem), rel_tol=1e-5)
    assert math.isclose(fractions[2], 1.82e-159, rel_tol=5e-3)
    assert fractions[1] == 1.0


def test_saha_factor_agrees_with_the_product_of_its_factors() -> None:
    for temperature, ionpot_ev, partfunc_lower, partfunc_upper in (
        (6000.0, IONPOT_H_EV, 2.0, 1.0),
        (1000.0, 7.9024681, 25.0, 30.0),
        (1e5, 54.4177655, 2.0, 1.0),
    ):
        product = (
            2.0
            * partfunc_upper
            / partfunc_lower
            * SAHA_CONST
            * math.pow(temperature, 1.5)
            * math.exp(-ionpot_ev / K_B / temperature)
        )
        assert product > 0.0
        assert math.isclose(
            pynonthermal.ionbalance.get_saha_factor(temperature, ionpot_ev, partfunc_lower, partfunc_upper),
            product,
            rel_tol=1e-12,
        )


def test_saha_factor_below_the_range_is_zero() -> None:
    # at 100 K the coefficient of hydrogen is about 1e-682, so zero is the correct double, and the
    # charge-neutral gas is neutral
    assert pynonthermal.ionbalance.get_saha_factor(100.0, IONPOT_H_EV, 2.0, 1.0) == 0.0
    fractions = pynonthermal.ionbalance.get_saha_ion_fractions(
        1, [1, 2], 100.0, n_elem=1e10, partfuncs={1: 2.0, 2: 1.0}
    )
    assert fractions == {1: 1.0, 2: 0.0}
