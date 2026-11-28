"""Check N(E) for a band that has its threshold between two energies of the grid."""

import numpy as np
import pytest

import pynonthermal


def _get_n_e_below_threshold(npts: int, autoionisation: bool) -> float:
    sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=101, npts=npts)
    # a step of the cross section at the threshold of 10 eV
    xs_vec = np.where(sf.engrid_mid >= 10.0, 1e-16, 0.0)
    if autoionisation:
        # hydrogen keeps 13.6 eV, so the 10 eV threshold gives a warning about the energy of the ion
        with pytest.warns(UserWarning, match="is less than the energy that the ion must keep"):
            sf.add_ionisation_channel(1, 1, 1e8, 10.0, xs_vec, autoionisation=True)
    else:
        sf.add_excitation(1, 1, 1e8, xs_vec, 10.0)
    sf.override_n_e(100.0)
    sf.solve(deposition_ev_per_s_per_cm3=1e8)
    return sf.calculate_N_e(0.1)


@pytest.mark.parametrize("autoionisation", [False, True])
def test_n_e_interpolates_the_product_across_a_threshold(autoionisation: bool) -> None:
    # The electron of N(0.1 eV) comes from 10.1 eV, between the middle energies 9.5 eV and 10.5 eV of
    # the 101-point grid. On that grid, y at 9.5 eV is about 300 times larger than y at 10.5 eV. An
    # interpolation of y and of the cross section one by one gave 1.835e7 on the 101-point grid.
    n_e_coarse = _get_n_e_below_threshold(101, autoionisation)
    n_e_fine = _get_n_e_below_threshold(801, autoionisation)
    n_e_finer = _get_n_e_below_threshold(1601, autoionisation)

    # a 9601-point grid gives 3.009e5, and the 801-point grid is 1.1 % below that value
    assert n_e_fine == pytest.approx(3.009e5, rel=0.02)
    assert n_e_finer == pytest.approx(n_e_fine, rel=2e-3)
    # the threshold is inside one interval of the coarse grid, so the grid error there is of first order
    assert 0.4 * n_e_finer < n_e_coarse < 1.2 * n_e_finer
