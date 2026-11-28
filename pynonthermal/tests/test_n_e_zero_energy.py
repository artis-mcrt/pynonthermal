"""Check N(E) at zero energy against its limit from positive energies."""

import math

import numpy as np

import pynonthermal


def test_n_e_at_zero_energy_is_the_limit_from_positive_energies() -> None:
    # N(E) approaches 1.5998658 at positive energies near zero, and the method gave exactly zero at E = 0
    sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=300, npts=300)
    sf.add_ionisation(8, 1, 1e6)
    sf.override_n_e(1e5)
    sf.solve(deposition_ev_per_s_per_cm3=1e3)

    n_e_zero = sf.calculate_N_e(0.0)
    # the tolerance allows a small change of the source of the solver
    assert math.isclose(n_e_zero, 1.5999, rel_tol=2e-3)
    # the values at small positive energies approach the value at zero
    assert math.isclose(sf.calculate_N_e(1e-4), n_e_zero, rel_tol=1e-5)
    assert math.isclose(sf.calculate_N_e(1e-9), n_e_zero, rel_tol=1e-7)
    # below about 1e-12 eV the sum 2 E + I lost the digits of E, and the value had an error of 1e-4
    for energy_ev in (1e-300, 1e-15, 1e-12):
        assert sf.calculate_N_e(energy_ev) == n_e_zero


def test_n_e_at_zero_energy_with_an_excitation() -> None:
    # the excitation term at zero energy is the product of y and the cross section at the transition energy
    sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=300, npts=300)
    sf.add_excitation(8, 1, 1e6, np.where(sf.engrid_mid >= 9.5, 1e-16, 0.0), 9.5)
    sf.override_n_e(1e5)
    sf.solve(deposition_ev_per_s_per_cm3=1e3)

    n_e_zero = sf.calculate_N_e(0.0)
    assert n_e_zero > 0.0
    assert math.isclose(sf.calculate_N_e(1e-9), n_e_zero, rel_tol=1e-7)
    assert math.isclose(sf.calculate_N_e(1e-3), n_e_zero, rel_tol=1e-2)
    # The term of an excitation has no singular point, so it uses the energy of the call and not the lowest
    # energy of the ionisation integrals. The transition energy 9.5 eV is the middle of the interval with
    # the index 8, so the interpolation gives the product of that interval.
    assert sf.engrid_mid[8] == 9.5
    assert math.isclose(n_e_zero, 1e6 * sf.yvec[8] * 1e-16, rel_tol=1e-12)


def test_n_e_at_zero_energy_with_a_small_emin_and_a_large_potential() -> None:
    # The potential of O VIII is 871 eV. With 1e-6 emin_ev alone as the lowest energy, the sum 2 E + I was
    # equal to I in double precision, and N(0) had an error of 5e-4.
    sf = pynonthermal.SpencerFanoSolver(emin_ev=1e-7, emax_ev=2000, npts=2000)
    sf.add_ionisation(8, 8, 1e6)
    sf.override_n_e(1e-3)
    sf.solve(deposition_ev_per_s_per_cm3=1e3)

    n_e_zero = sf.calculate_N_e(0.0)
    assert math.isclose(n_e_zero, sf.calculate_N_e(1e-7), rel_tol=1e-7)
    assert math.isclose(n_e_zero, sf.calculate_N_e(1e-5), rel_tol=1e-5)
