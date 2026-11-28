"""Check N(E) at zero energy against its limit from positive energies."""

import math

import numpy as np
import numpy.typing as npt

import pynonthermal


def _xs_step(energies_ev: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.where(energies_ev >= 9.5, 1e-16, 0.0)


def test_n_e_at_zero_energy_is_the_limit_from_positive_energies() -> None:
    # N(E) approaches 1.5998658 at positive energies near zero, and the method gave exactly zero at E = 0
    sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=300, npts=300)
    sf.add_ionisation(8, 1, 1e6)
    sf.override_n_e(1e5)
    sf.solve(deposition_ev_per_s_per_cm3=1e3)

    n_e_zero = sf.calculate_N_e(0.0)
    assert math.isclose(n_e_zero, 1.5998658, rel_tol=1e-7)
    # the values at small positive energies approach the value at zero
    assert math.isclose(sf.calculate_N_e(1e-4), n_e_zero, rel_tol=1e-5)
    assert math.isclose(sf.calculate_N_e(1e-9), n_e_zero, rel_tol=1e-7)
    # below about 1e-12 eV the sum 2 E + I lost the digits of E, and the value had an error of 1e-4
    for energy_ev in (1e-300, 1e-15, 1e-12):
        assert sf.calculate_N_e(energy_ev) == n_e_zero


def test_n_e_at_zero_energy_with_an_excitation() -> None:
    # the excitation term at zero energy is the product of y and the cross section at the transition energy
    sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=300, npts=300)
    sf.add_excitation(8, 1, 1e6, _xs_step, 9.5)
    sf.override_n_e(1e5)
    sf.solve(deposition_ev_per_s_per_cm3=1e3)

    n_e_zero = sf.calculate_N_e(0.0)
    assert n_e_zero > 0.0
    assert math.isclose(sf.calculate_N_e(1e-9), n_e_zero, rel_tol=1e-7)
    assert math.isclose(sf.calculate_N_e(1e-3), n_e_zero, rel_tol=1e-2)
