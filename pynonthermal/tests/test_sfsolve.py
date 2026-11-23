"""Tests of the Spencer-Fano solution and of its analysis.

The comments cite these publications by their labels:

- A80: Axelrod 1980, PhD thesis, University of California, Santa Cruz, ADS 1980PhDT.........1A
- L67: Lotz 1967, Zeitschrift fuer Physik, 206, 205-211, doi:10.1007/BF01325928
"""

import dataclasses
import math
import typing as t
import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

import pynonthermal

pytestmark = pytest.mark.benchmark
outputfolder = Path(__file__).absolute().parent / "output"


def test_lotz_heavy_element() -> None:
    # elements heavier than Ni (Z>28) use the A80/L67 cross section approximation
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=400, verbose=True) as sf:
        with pytest.warns(pynonthermal.LotzApproximationWarning, match="Z=56 ion_stage 2"):
            sf.add_ionisation(56, 2, n_ion=1.0)
        sf.override_n_e(1.0)
        sf.solve(deposition_ev_per_s_per_cm3=100)

        # per-ion getter triggers the analysis lazily
        assert sf.get_frac_ionisation_ion(56, 2) > 0.0
        assert sf.get_eff_ionpot(56, 2) > 0.0
        assert math.isclose(sf.get_frac_sum(), 1.0, abs_tol=0.01)
        assert sf.get_ionisation_ratecoeff(56, 2) > 0.0


class CountingAnalysisSolver(pynonthermal.SpencerFanoSolver):
    """A solver that counts the calls to analyse_ntspectrum."""

    analyse_count: int = 0

    @t.override
    def analyse_ntspectrum(self) -> None:
        self.analyse_count += 1
        super().analyse_ntspectrum()


def test_api_guards() -> None:
    with CountingAnalysisSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        sf.set_temperature(3000)
        sf.set_atomic_data(use_collstrengths=False)
        sf.add_ionisation(2, 1, n_ion=1.0)

        # getters require solve() to have been called
        with pytest.raises(RuntimeError):
            sf.get_frac_heating()
        with pytest.raises(RuntimeError):
            sf.get_excitation_ratecoeff(2, 1, 0)

        # the same ion can't be added twice: its channel keys are already present
        with pytest.raises(ValueError, match="already added"):
            sf.add_ionisation(2, 1, n_ion=1.0)

        # a bare nucleus (here H II) has no electron to remove, so it is rejected
        with pytest.raises(ValueError, match="bare nucleus"):
            sf.add_ionisation(1, 2, n_ion=1.0)

        # xs_vec must be defined on the full energy grid
        with pytest.raises(ValueError, match="engrid"):
            sf.add_excitation(2, 1, levelnumberdensity=1.0, xs_vec=np.zeros(3), epsilon_trans_ev=10.0)

        # an omitted transitionkey gets an automatic index, and duplicate keys are rejected
        sf.add_excitation(2, 1, levelnumberdensity=1.0, xs_vec=np.zeros(200), epsilon_trans_ev=25.0)
        with pytest.raises(ValueError, match="already added"):
            sf.add_excitation(
                2, 1, levelnumberdensity=1.0, xs_vec=np.zeros(200), epsilon_trans_ev=25.0, transitionkey=0
            )

        # excitation data must exist for the requested ion
        sf.set_atomic_data(adata_polars=pl.DataFrame({"Z": [2], "ion_stage": [1]}))
        with pytest.raises(ValueError, match=r"No excitation data .* has the ions \[\(2, 1\)\]"):
            sf.add_ion_excitation(3, 1, n_ion=1.0)

        # a zero-population ion is registered with no channel, and a negative population is rejected
        sf.add_ionisation(8, 3, n_ion=0.0)
        assert sf.ionpopdict[(8, 3)] == 0.0
        assert (8, 3) not in sf._ionisation_channels
        with pytest.raises(ValueError, match="non-negative"):
            sf.add_ionisation(8, 3, n_ion=-1.0)

        # second ion with no excitation channels (exercises the zero-excitation analysis path)
        sf.add_ionisation(8, 2, n_ion=0.1)

        sf.override_n_e(1e-4)
        sf.solve(deposition_ev_per_s_per_cm3=100)

        # a legitimately-zero excitation fraction must not trigger repeated re-analysis
        assert sf.get_frac_excitation_tot() == 0.0
        count_after_first_call = sf.analyse_count
        assert count_after_first_call >= 1
        assert sf.get_frac_excitation_tot() == 0.0
        assert sf.analyse_count == count_after_first_call

        # each getter triggers the analysis lazily from a freshly-solved (un-analysed) state
        sf.override_n_e(1e-4)
        sf.solve(deposition_ev_per_s_per_cm3=100)
        assert sf.get_frac_ionisation_tot() > 0.0
        sf.override_n_e(1e-4)
        sf.solve(deposition_ev_per_s_per_cm3=100)
        assert sf.get_frac_ionisation_ion(2, 1) > 0.0
        sf.override_n_e(1e-4)
        sf.solve(deposition_ev_per_s_per_cm3=100)
        assert sf.get_eff_ionpot(2, 1) > 0.0
        sf.override_n_e(1e-4)
        sf.solve(deposition_ev_per_s_per_cm3=100)
        assert sf.get_ionisation_ratecoeff(2, 1) > 0.0

        # additions are locked after solving
        with pytest.raises(RuntimeError):
            sf.add_ionisation(2, 2, n_ion=1e-4)


def test_invalid_excitation_fraction_reported() -> None:
    # a stored transition that the matrix does not hold gives frac_excitation > 1. It is reported
    # but kept in the total, as for frac_ionisation_shell, so that the problem stays visible
    # rather than being hidden behind a total that silently dropped the channel. The solver keeps
    # the stored transitions and the matrix consistent, so the test edits the stored transition.
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=100) as sf:
        sf.add_ionisation(2, 1, n_ion=1.0)
        sf.add_excitation(2, 1, levelnumberdensity=1.0, xs_vec=np.full(100, 1e-18), epsilon_trans_ev=25.0)
        sf.override_n_e(1e-4)
        sf.solve(deposition_ev_per_s_per_cm3=100)
        transition = sf.excitationlists[(2, 1)][0]
        sf.excitationlists[(2, 1)][0] = dataclasses.replace(transition, levelnumberdensity=1e30)

        with (
            pytest.warns(UserWarning, match="invalid frac_excitation_ion"),
            pytest.warns(UserWarning, match="energy fractions sum"),
        ):
            frac_excitation_tot = sf.get_frac_excitation_tot()

        assert frac_excitation_tot > 1.0


def test_helium() -> None:
    # KF1992 Figure 3. Pure-Helium Plasma
    x_e = 1e-4
    ions = [
        # Z ion_stage numberdensity
        (2, 1, 1.0 - x_e),
        (2, 2, x_e),
    ]

    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=2000, verbose=True) as sf:
        sf.set_temperature(3000)
        sf.set_atomic_data(use_collstrengths=False)
        for Z, ion_stage, n_ion in ions:
            sf.add_ionisation(Z, ion_stage, n_ion=n_ion)
            sf.add_ion_excitation(Z, ion_stage, n_ion=n_ion)

        # re-adding an ion's excitations with a different population is not allowed
        with pytest.raises(ValueError, match="different populations"):
            sf.add_ion_excitation(2, 1, n_ion=99.9)

        # call solve twice to test that it can be called multiple times without error
        sf.solve(deposition_ev_per_s_per_cm3=1000)
        sf.solve(deposition_ev_per_s_per_cm3=100)

        sf.analyse_ntspectrum()
        frac_excitation_tot = sf.get_frac_excitation_tot()
        frac_ionisation_tot = sf.get_frac_ionisation_tot()
        frac_heating = sf.get_frac_heating()

        frac_sum = frac_excitation_tot + frac_ionisation_tot + frac_heating
        assert math.isclose(frac_sum, 1.0, abs_tol=0.005)
        assert math.isclose(frac_excitation_tot, 0.3315, abs_tol=0.05)
        assert math.isclose(frac_ionisation_tot, 0.4849, abs_tol=0.05)
        assert math.isclose(sf.get_frac_ionisation_ion(Z=ions[0][0], ion_stage=ions[0][1]), 0.4807, abs_tol=0.05)
        assert math.isclose(sf.get_frac_ionisation_ion(Z=ions[1][0], ion_stage=ions[1][1]), 0.0, abs_tol=0.05)

        sf.plot_spec_channels(outputfilename=outputfolder / "spec_channels.pdf", xscalelog=True)
        sf.plot_yspectrum(outputfilename=outputfolder / "yspectrum.pdf")
        sf.plot_channels(outputfilename=outputfolder / "channels.pdf", xscalelog=True)


def test_heating_only_approximation() -> None:
    # with heating_only_approximation=True, the matrix contains no excitation or ionisation loss
    # terms, but the analysis still calculates the channel rates from the heating-only solution
    x_e = 1e-4
    ions = [
        # Z ion_stage numberdensity
        (2, 1, 1.0 - x_e),
        (2, 2, x_e),
    ]

    def build_solver(heating_only_approximation: bool) -> pynonthermal.SpencerFanoSolver:
        # the assertions are qualitative, so a coarse grid is enough (test_helium covers npts=2000)
        sf = pynonthermal.SpencerFanoSolver(
            emin_ev=1, emax_ev=3000, npts=512, heating_only_approximation=heating_only_approximation
        )
        sf.set_temperature(3000)
        sf.set_atomic_data(use_collstrengths=False)
        for Z, ion_stage, n_ion in ions:
            sf.add_ionisation(Z, ion_stage, n_ion=n_ion)
            sf.add_ion_excitation(Z, ion_stage, n_ion=n_ion)
        return sf

    with (
        build_solver(heating_only_approximation=True) as sf_heat,
        build_solver(heating_only_approximation=False) as sf_full,
    ):
        # the excitation and ionisation terms stay out of the matrix
        assert not sf_heat.sfmatrix.any()
        assert sf_full.sfmatrix.any()

        sf_heat.solve(deposition_ev_per_s_per_cm3=100)
        sf_full.solve(deposition_ev_per_s_per_cm3=100)

        # the analysis of the heating-only solution must not show a solver warning
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            sf_heat.analyse_ntspectrum()

        # heating takes almost all of the deposited energy
        assert math.isclose(sf_heat.get_frac_heating(), 1.0, rel_tol=0.05)

        # the channel rates are still available from the stored data
        assert sf_heat.get_frac_ionisation_tot() > 0.0
        assert sf_heat.get_frac_excitation_tot() > 0.0
        assert sf_heat.get_ionisation_ratecoeff(2, 1) > 0.0
        assert sf_heat.get_frac_ionisation_ion(2, 1) > 0.0

        # without the competing loss terms, y(E) is larger, so the rates are larger
        assert sf_heat.get_frac_ionisation_tot() > sf_full.get_frac_ionisation_tot()
        assert sf_heat.get_frac_excitation_tot() > sf_full.get_frac_excitation_tot()


def test_iron() -> None:
    ions = [
        # Z ion_stage numberdensity
        (26, 2, 0.3),
        (26, 3, 0.7),
    ]

    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=16000, npts=1024, verbose=True) as sf:
        sf.set_temperature(3000)
        sf.set_atomic_data(use_collstrengths=False)
        for Z, ion_stage, n_ion in ions:
            sf.add_ionisation(Z, ion_stage, n_ion=n_ion)
            sf.add_ion_excitation(Z, ion_stage, n_ion=n_ion)

        sf.solve(deposition_ev_per_s_per_cm3=100)

        sf.analyse_ntspectrum()
        frac_excitation_tot = sf.get_frac_excitation_tot()
        frac_ionisation_tot = sf.get_frac_ionisation_tot()
        frac_heating = sf.get_frac_heating()

        frac_sum = frac_excitation_tot + frac_ionisation_tot + frac_heating
        assert math.isclose(frac_sum, 1.0, abs_tol=1e-9)
        # relative tolerances, because an absolute tolerance of 0.05 accepted an excitation fraction of zero. A
        # run with maxnlevelslower=2 kept 291 of the 630 transitions and passed. The reference values are those
        # of npts=32768. This grid of 1024 points gives 0.022526 and 0.176099 with the loss function and the
        # cross sections at the middle of each grid interval. With them at the left end of each interval, it
        # gave 0.022263 and 0.175606, which is 1.05% and 0.32% below the reference values.
        assert math.isclose(frac_excitation_tot, 0.022500, rel_tol=0.003)
        assert math.isclose(frac_ionisation_tot, 0.176174, rel_tol=0.001)

        # the values at the left end of each interval were 0.51% and 0.34% below these values of npts=32768
        assert math.isclose(sf.get_ionisation_ratecoeff(26, 2), 4.48128e-01, rel_tol=0.002)
        assert math.isclose(sf.get_ionisation_ratecoeff(26, 3), 3.74206e-01, rel_tol=0.002)

        # rate coefficients are in s^-1 and so scale with the deposition rate density of 100 eV/s/cm3. The
        # reference values are those of npts=16384.
        assert math.isclose(sf.get_excitation_ratecoeff(26, 2, (0, 100)), 4.0293e-05, rel_tol=0.01)
        assert math.isclose(sf.get_excitation_ratecoeff(26, 2, (1, 100)), 6.7147e-06, rel_tol=0.01)

        sf.plot_spec_channels(outputfilename=outputfolder / "spec_channels_iron.pdf", xscalelog=True)


def test_grid_error_of_the_default_grid() -> None:
    # Fe I at an electron fraction of 0.25 on the default grid of 4096 points, against the limit of a fine
    # grid. The limit is the Richardson extrapolation of order 1 from 16384 and 32768 points. The rule with the
    # inputs at the left end of each interval and the rule with them at the middle give the same limit to 1e-4.
    # With the loss function and the cross sections at the left end of each grid interval, the errors of this
    # grid were +1.24% (heating), -0.66% (ionisation), -2.16% (excitation), and -1.02% (rate coefficient).
    # With them at the middle, the errors are +0.01%, -0.19%, +0.32%, and -0.26%. For 41 grids from 3600 to
    # 4600 points, the largest errors are 0.17%, 0.30%, 0.32%, and 0.45%.
    n_ion = 1.0e6
    with pynonthermal.SpencerFanoSolver() as sf:
        assert len(sf.engrid) == 4096
        sf.set_temperature(6000.0)
        sf.add_ionisation(26, 1, n_ion=n_ion)
        sf.add_ion_excitation(26, 1, n_ion=n_ion)
        sf.override_n_e(0.25 * n_ion)
        sf.solve(deposition_ev_per_s_per_cm3=n_ion)

        assert abs(sf.get_frac_sum() - 1.0) < 1e-11
        assert math.isclose(sf.get_frac_heating(), 0.48892, rel_tol=0.003)
        assert math.isclose(sf.get_frac_ionisation_tot(), 0.33153, rel_tol=0.004)
        assert math.isclose(sf.get_frac_excitation_tot(), 0.17956, rel_tol=0.006)
        assert math.isclose(sf.get_ionisation_ratecoeff(26, 1), 0.022199, rel_tol=0.005)


def test_inputs_are_at_the_middle_of_each_grid_interval() -> None:
    # column j of the matrix holds the electrons from engrid[j] to engrid[j] + deltaen, and the solver takes
    # the loss function and each cross section of the column at engrid_mid[j]
    def bethe_like_xs(en_ev: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        u = np.maximum(np.asarray(en_ev, dtype=np.float64) / 30.0, 1.0)
        return 1e-17 * np.log(u) / u

    def excitation_xs(en_ev: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return 1e-16 * 12.0 / np.asarray(en_ev, dtype=np.float64)

    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf:
        assert not sf.engrid_mid.flags.writeable
        assert np.array_equal(sf.engrid_mid, sf.engrid + 0.5 * sf.deltaen)
        sf.add_ionisation_channel(10, 1, 1e8, 30.0, bethe_like_xs, "direct")
        sf.add_excitation(10, 1, 1e8, excitation_xs, 12.0, "line")
        sf.override_n_e(1e7)
        xs_grid = sf._ionisation_channels[(10, 1)][0].xs_grid
        above = sf.engrid > 30.0
        assert np.array_equal(xs_grid[above], bethe_like_xs(sf.engrid_mid[above]))
        assert not xs_grid[~above].any()
        xs_vec = sf.excitationlists[(10, 1)]["line"].xs_vec
        at_or_above = sf.engrid_mid >= 12.0
        assert np.array_equal(xs_vec[at_or_above], excitation_xs(sf.engrid_mid[at_or_above]))
        assert not xs_vec[~at_or_above].any()
        sf.solve(deposition_ev_per_s_per_cm3=1e8)
        # the heating integral over the grid, with the loss function at the middle of each interval
        lossvec = pynonthermal.base._get_lossvec(sf.engrid_mid, sf.get_n_e())
        heating_on_grid = sf.deltaen * float(np.dot(lossvec, sf.yvec)) / sf.deposition_ev_per_s_per_cm3
        assert 0.0 < heating_on_grid < sf.get_frac_heating()
        assert abs(sf.get_frac_sum() - 1.0) < 1e-12


@pytest.mark.parametrize(
    ("ionpot_ev", "first_column"),
    [
        # the grid points are at 1, 2, 3, ... eV, and the middle of the interval of column j is at j + 1.5 eV
        (30.0, 30),  # the potential on the grid point of column 29: the fill starts at column 30
        (30.25, 30),  # the potential between the grid point of column 29 and the middle of its interval
        (30.75, 30),  # the potential above the middle of the interval of column 29
    ],
)
def test_fractions_sum_to_one_at_the_start_of_an_ionisation(ionpot_ev: float, first_column: int) -> None:
    # a direct ionisation starts at the first grid point above the potential, also where the middle of the
    # interval that contains the potential is above it. The matrix and the analysis use the same columns.
    def xs(en_ev: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        u = np.maximum(np.asarray(en_ev, dtype=np.float64) / ionpot_ev, 1.0)
        return 1e-16 * np.log(u) / u

    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3001, npts=3001) as sf:
        assert sf.deltaen == 1.0
        sf.add_ionisation_channel(10, 1, 1e8, ionpot_ev, xs, "direct")
        sf.override_n_e(1e6)
        xs_grid = sf._ionisation_channels[(10, 1)][0].xs_grid
        assert not xs_grid[:first_column].any()
        assert xs_grid[first_column] == xs(sf.engrid_mid[first_column : first_column + 1])[0] > 0.0
        sf.solve(deposition_ev_per_s_per_cm3=1e8)
        assert sf.get_frac_ionisation_tot() > 0.1
        assert abs(sf.get_frac_sum() - 1.0) < 1e-12


@pytest.mark.parametrize(
    ("threshold_ev", "first_column"),
    [
        # the grid points are at 1, 2, 3, ... eV, and the middle of the interval of column j is at j + 1.5 eV
        (30.5, 29),  # the threshold at the middle of the interval of column 29: the column keeps its value
        (30.25, 29),  # the threshold below that middle, and above the grid point of the column
        (30.75, 30),  # the threshold above that middle: the band starts at the next column
        (0.6 + 1.0, 1),  # a threshold of less than two grid intervals
    ],
)
@pytest.mark.parametrize("autoionisation", [False, True])
def test_fractions_sum_to_one_at_the_start_of_an_excitation(
    threshold_ev: float, first_column: int, autoionisation: bool
) -> None:
    # an excitation, and an autoionisation channel with its Auger electron, start at the first column whose
    # middle energy is at or above the threshold. That column can start below the threshold. The band, the
    # source rows of the Auger electron, and the analysis all take the first column from the stored array.
    def xs(en_ev: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return 1e-16 * threshold_ev / np.asarray(en_ev, dtype=np.float64)

    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3001, npts=3001) as sf:
        assert sf.deltaen == 1.0
        if autoionisation:
            if threshold_ev < 25.0:
                pytest.skip("the threshold of an autoionisation is above the ionisation potential")
            sf.add_ionisation_channel(10, 1, 1e8, threshold_ev, xs, "auto", autoionisation=True)
            channel = sf._ionisation_channels[(10, 1)][0]
            assert channel.auger_electron_energy_ev > 8.0
            xs_stored = channel.xs_grid
        else:
            sf.add_excitation(10, 1, 1e8, xs, threshold_ev, "line")
            xs_stored = sf.excitationlists[(10, 1)]["line"].xs_vec
        sf.override_n_e(1e6)
        assert not xs_stored[:first_column].any()
        assert xs_stored[first_column] == xs(sf.engrid_mid[first_column : first_column + 1])[0]
        assert sf.engrid_mid[first_column - 1] < threshold_ev <= sf.engrid_mid[first_column]
        sf.solve(deposition_ev_per_s_per_cm3=1e8)
        fraction = sf.get_frac_ionisation_tot() if autoionisation else sf.get_frac_excitation_tot()
        assert fraction > 1e-3
        assert abs(sf.get_frac_sum() - 1.0) < 1e-12
