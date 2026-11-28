"""Check the ionisation balance for a trial step above the density limit and for a large change of a population."""

import math

import numpy as np
import pytest

import pynonthermal


def _get_dense_hydrogen() -> pynonthermal.SpencerFanoSolver:
    sf = pynonthermal.SpencerFanoSolver(emin_ev=0.1, emax_ev=1000, npts=128)
    sf.add_element(Z=1, n_elem=6.721055158836534e17, recomb_ratecoeffs={2: 1e-12})
    return sf


def test_balance_reduces_a_trial_step_above_the_density_limit() -> None:
    # The loss function at 0.1 eV is valid below 6.721e16 cm^-3. The solution is at 6.049e16 cm^-3. The
    # first trial step from the provisional populations gave 7.427e16 cm^-3, and solve() raised an error.
    deposition = 1.4919643291051824e23
    sf = _get_dense_hydrogen()
    sf.solve(deposition_ev_per_s_per_cm3=deposition, balance_tol=1e-8)

    # a solve at a lower deposition rate first gives a start that needs no reduced step
    sf_two_solves = _get_dense_hydrogen()
    sf_two_solves.solve(deposition_ev_per_s_per_cm3=1.1243756814216838e23, balance_tol=1e-8)
    sf_two_solves.solve(deposition_ev_per_s_per_cm3=deposition, balance_tol=1e-8)

    # the tolerance allows a small change of the source of the solver
    assert sf.get_n_e() == pytest.approx(6.049e16, rel=1e-3)
    assert math.isclose(sf.get_n_e(), sf_two_solves.get_n_e(), rel_tol=1e-7)
    assert math.isclose(sf.get_frac_sum(), 1.0, rel_tol=1e-12)
    # the populations of the solution give the ionisation rate that is equal to the recombination rate
    n_e = sf.get_n_e()
    assert math.isclose(
        sf.ionpopdict[1, 1] * sf.get_ionisation_ratecoeff(1, 1), sf.ionpopdict[1, 2] * n_e * 1e-12, rel_tol=1e-7
    )


def test_balance_keeps_the_error_for_a_solution_above_the_density_limit() -> None:
    # at this deposition rate the solution is above the limit, so no number of reduced steps finds it
    sf = _get_dense_hydrogen()
    with pytest.raises(ValueError, match="the free-electron loss function is not valid"):
        sf.solve(deposition_ev_per_s_per_cm3=1e26)


# the top stage O II of this short chain takes a part of the ionisation, which is not the subject here
@pytest.mark.filterwarnings("ignore:.*of the ionisation events of the element leave the chain")
def test_balance_fills_the_matrix_again_after_a_large_population_decrease() -> None:
    # The balance adds the change of each population to the matrix. After the decrease of O II from
    # 5.7e7 to 6.9e-13 cm^-3, the residue of the sum was larger than the terms of O II. The heating
    # fraction was 0.03132 and the energy sum was 0.76973.
    sf = pynonthermal.SpencerFanoSolver(emin_ev=0.1, emax_ev=300, npts=128)
    sf.set_temperature(6000)
    sf.add_element(Z=8, n_elem=1e8, recomb_ratecoeffs={2: 1e-12})
    sf.add_ion_excitation(8, 2, None)
    sf.solve(deposition_ev_per_s_per_cm3=1e6)
    assert sf.ionpopdict[8, 2] > 1e7
    sf.solve(deposition_ev_per_s_per_cm3=1e-35, balance_tol=1e-8)
    assert sf.ionpopdict[8, 2] < 1e-10

    sf_fresh = pynonthermal.SpencerFanoSolver(emin_ev=0.1, emax_ev=300, npts=128)
    sf_fresh.set_temperature(6000)
    sf_fresh.add_element(Z=8, ion_densities={1: sf.ionpopdict[8, 1], 2: sf.ionpopdict[8, 2]})
    sf_fresh.add_ion_excitation(8, 2, None)
    sf_fresh.solve(deposition_ev_per_s_per_cm3=1e-35)

    np.testing.assert_allclose(sf.sfmatrix, sf_fresh.sfmatrix, rtol=1e-9, atol=0.0)
    np.testing.assert_allclose(sf.yvec, sf_fresh.yvec, rtol=1e-9)
    assert math.isclose(sf.get_frac_heating(), sf_fresh.get_frac_heating(), rel_tol=1e-9)
    # the heating fraction was 0.03132 with the residue in the matrix
    assert sf.get_frac_heating() == pytest.approx(0.2615, abs=5e-4)
    assert math.isclose(sf.get_frac_sum(), 1.0, rel_tol=1e-12)


# the top stage O II of this short chain takes a part of the ionisation, which is not the subject here
@pytest.mark.filterwarnings("ignore:.*of the ionisation events of the element leave the chain")
def test_balance_adds_a_small_population_change_without_a_new_fill(monkeypatch: pytest.MonkeyPatch) -> None:
    # a change of the deposition rate by a factor of 10 keeps each population above the ratio of
    # BALANCE_REBUILD_POPULATION_RATIO, so the balance only adds the changes
    sf = pynonthermal.SpencerFanoSolver(emin_ev=0.1, emax_ev=300, npts=128)
    sf.add_element(Z=8, n_elem=1e8, recomb_ratecoeffs={2: 1e-12})
    sf.solve(deposition_ev_per_s_per_cm3=1e6)
    rebuilds = []
    rebuild = sf._rebuild_sfmatrix

    def count_rebuild() -> None:
        rebuilds.append(1)
        rebuild()

    monkeypatch.setattr(sf, "_rebuild_sfmatrix", count_rebuild)
    sf.solve(deposition_ev_per_s_per_cm3=1e5)
    assert not rebuilds
    assert math.isclose(sf.get_frac_sum(), 1.0, rel_tol=1e-12)
