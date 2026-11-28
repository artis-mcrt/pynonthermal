"""Check that the source, the cross sections, and the integrals of the analysis all end at emax_ev."""

import math

import numpy as np
import numpy.typing as npt
import pytest

import pynonthermal


def _xs_above_100_ev(energies_ev: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.where(energies_ev > 100.0, 1e-16, 0.0)


def test_cross_section_above_emax_gives_no_ionisation() -> None:
    # The middle of the last column is at 100.495 eV, above the emax_ev of 100 eV. With a source in that
    # column, a channel with a cross section only above 100 eV had an ionisation fraction of 0.25022.
    sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=100, npts=101)
    sf.add_ionisation_channel(8, 1, 1e10, 99.9, _xs_above_100_ev, channelkey="above")
    sf.override_n_e(1e5)
    sf.solve(deposition_ev_per_s_per_cm3=1e5)

    channel = sf._ionisation_channels[8, 1][0]
    assert channel.xs_grid[-1] == 1e-16
    assert not channel.xs_grid[:-1].any()
    assert sf.get_ionisation_ratecoeff(8, 1) == 0.0
    assert sf.get_frac_ionisation_ion(8, 1) == 0.0
    assert math.isclose(sf.get_frac_heating(), 1.0, rel_tol=1e-12)
    assert math.isclose(sf.get_frac_sum(), 1.0, rel_tol=1e-12)


@pytest.mark.parametrize("npts", [2, 3, 101, 200])
def test_source_ends_at_emax(npts: int) -> None:
    # The source is in the intervals below emax_ev, so the mean energy of a source electron is below
    # emax_ev, and the solution is zero in the column that starts at emax_ev.
    sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=100, npts=npts)
    sf.add_ionisation(8, 1, 1e6)
    sf.override_n_e(1e5)
    sf.solve(deposition_ev_per_s_per_cm3=1e5)

    n_source = math.ceil(npts / 30.0)
    assert math.isclose(sf.E_init_ev, 100.0 - 0.5 * (n_source - 1) * sf.deltaen, rel_tol=1e-12)
    assert sf.rhsvec[-1] == 0.0
    assert math.isclose(sf.rhsvec[0], 1.0, rel_tol=1e-12)
    assert sf.yvec[-1] == 0.0
    # each interval of the source has electrons
    assert (sf.yvec[-1 - n_source : -1] > 0.0).all()
    assert math.isclose(sf.get_frac_sum(), 1.0, rel_tol=1e-9)


def test_n_e_uses_the_last_interval_below_emax() -> None:
    # The first integral of N(E) at 49.6 eV uses the solution from 63.2 eV to 100 eV, which includes the top
    # half of the last interval below emax_ev. The values of that interval apply up to emax_ev. Thus a grid
    # of 101 points agrees with a grid of 801 points. An interpolation to the zero of the last column gave a
    # value that was too small.
    def get_n_e(npts: int) -> float:
        sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=100, npts=npts)
        sf.add_ionisation(8, 1, 1e6)
        sf.override_n_e(1e5)
        sf.solve(deposition_ev_per_s_per_cm3=1e5)
        return sf.calculate_N_e(49.6)

    assert get_n_e(101) == pytest.approx(get_n_e(801), rel=5e-3)
