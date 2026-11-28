"""Check the storage of the heating-only approximation and the matrix fill of inactive shells."""

import math
import tracemalloc

import numpy as np
import pytest

import pynonthermal
from pynonthermal.base import _get_lossvec
from pynonthermal.spencerfano import LotzApproximationWarning
from pynonthermal.spencerfano import solve_upper_triangular


def test_heating_only_solves_the_diagonal_equation() -> None:
    # the spectrum is that of the triangular solve with a matrix of zeros
    sf = pynonthermal.SpencerFanoSolver(npts=256, emax_ev=1000, heating_only_approximation=True)
    sf.set_temperature(6000)
    sf.add_ionisation(8, 2, 1e8)
    sf.add_ion_excitation(8, 2)
    sf.solve(deposition_ev_per_s_per_cm3=1e5)

    lossvec = _get_lossvec(sf.engrid_mid, sf.get_n_e())
    yvec_triangular = solve_upper_triangular(np.zeros((256, 256)), sf.rhsvec, diag_add=lossvec) * 1e5 / sf.E_init_ev
    np.testing.assert_allclose(sf.yvec, yvec_triangular, rtol=1e-14)
    assert sf.sfmatrix.shape == (256, 256)
    assert not sf.sfmatrix.any()
    assert not sf.sfmatrix.flags.writeable
    # the channel fractions come from the approximate solution, and each one is above zero
    assert math.isclose(sf.get_frac_heating(), 1.0, rel_tol=1e-9)
    assert sf.get_frac_ionisation_tot() > 0.0
    assert sf.get_frac_excitation_tot() > 0.0


def test_heating_only_needs_linear_storage() -> None:
    # a matrix of 4096 x 4096 values takes 128 mebibytes
    npts = 4096
    # the first solver reads the data files, which the later solvers share
    pynonthermal.SpencerFanoSolver(npts=10).add_ionisation(8, 2, 1e8)
    tracemalloc.start()
    try:
        sf = pynonthermal.SpencerFanoSolver(npts=npts, heating_only_approximation=True)
        sf.add_ionisation(8, 2, 1e8)
        sf.solve(deposition_ev_per_s_per_cm3=1e5)
        sf.analyse_ntspectrum()
        peak_bytes = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak_bytes < 0.05 * npts * npts * 8
    assert math.isclose(sf.get_frac_heating(), 1.0, rel_tol=1e-9)


def test_heating_only_balance_adds_nothing_to_the_matrix() -> None:
    sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300, heating_only_approximation=True)
    sf.add_element(Z=8, n_elem=1e8, recomb_ratecoeffs={2: 1e-12, 3: 3e-12, 4: 5e-12, 5: 1e-11})
    sf.solve(deposition_ev_per_s_per_cm3=1e3)
    sf.solve(deposition_ev_per_s_per_cm3=1e-30)
    assert not sf.sfmatrix.any()
    assert not sf._stage_matrices
    assert math.isclose(sum(sf.get_ion_fractions(8).values()), 1.0, rel_tol=1e-9)


def test_inactive_shells_have_no_matrix_fill(monkeypatch: pytest.MonkeyPatch) -> None:
    # 9 of the 27 shells of U II have a potential above the emax_ev of 3000 eV, so their cross section is
    # zero at each energy of the grid
    sf = pynonthermal.SpencerFanoSolver(emin_ev=0.1, emax_ev=3000, npts=1024)
    fills = []
    build_channel_fill = sf._build_channel_fill

    def count_fill(channel: pynonthermal.IonisationChannel) -> object:
        fills.append(channel.key)
        return build_channel_fill(channel)

    monkeypatch.setattr(sf, "_build_channel_fill", count_fill)
    with pytest.warns(LotzApproximationWarning):
        sf.add_ionisation(92, 2, 1e8)
    channels = sf._ionisation_channels[92, 2]
    inactive = [channel for channel in channels if not channel.xs_grid.any()]
    assert len(channels) == 27
    assert len(inactive) == 9
    assert len(fills) == 18
    assert not {channel.key for channel in inactive}.intersection(fills)

    # the fill of an inactive shell has only zeros, so the matrix is the same without it
    for channel in inactive:
        fill = build_channel_fill(channel)
        assert not fill.unit_prefactors.any()
        assert not fill.partial_unit_terms.any()

    sf.solve(deposition_ev_per_s_per_cm3=1e5)
    assert math.isclose(sf.get_frac_sum(), 1.0, rel_tol=1e-9)
    # the records of the inactive shells stay, and the rate of the ion is the sum over the active shells
    ratecoeff_active = sf.deltaen * sum(
        float(np.dot(sf.yvec, channel.xs_grid)) for channel in channels if channel.xs_grid.any()
    )
    assert ratecoeff_active > 0.0
    assert math.isclose(sf.get_ionisation_ratecoeff(92, 2), ratecoeff_active, rel_tol=1e-12)


def test_stage_with_only_inactive_shells_has_no_unit_matrix() -> None:
    # O II has a potential of 35.1 eV, above the emax_ev of 30 eV, so each of its shells is inactive
    sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=30, npts=100)
    sf.add_element(Z=8, n_elem=1e8, recomb_ratecoeffs={2: 1e-12, 3: 3e-12})
    assert sf._ionisation_channels[8, 2]
    assert not any(channel.xs_grid.any() for channel in sf._ionisation_channels[8, 2])
    with pytest.warns(UserWarning, match="the non-thermal ionisation rate of Z=8 ion_stage 2 is zero"):
        sf.solve(deposition_ev_per_s_per_cm3=1e3)
    assert (8, 1) in sf._stage_matrices
    assert (8, 2) not in sf._stage_matrices
    assert sf.get_ionisation_ratecoeff(8, 2) == 0.0
    assert sf.get_ion_fractions(8)[3] == 0.0
    assert math.isclose(sf.get_frac_sum(), 1.0, rel_tol=1e-9)
