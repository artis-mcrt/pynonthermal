"""Regression tests for solver correctness and input validation.

Deliberately not marked as benchmarks: these exercise error paths and cross section edge cases
rather than solver performance, so instrumenting them would only lengthen the CodSpeed run and
add noise to the tracked benchmark set. Performance-relevant cases live in test_sfsolve.py.

The comments cite these publications by their labels:

- KF92: Kozma & Fransson 1992, ApJ, 390, 602-621, doi:10.1086/171311
- LHD12: Li, Hillier & Dessart 2012, MNRAS, 426, 1671-1686, doi:10.1111/j.1365-2966.2012.21198.x
- OPB71: Opal, Peterson & Beaty 1971, J. Chem. Phys., 55, 4100-4106, doi:10.1063/1.1676707
"""

import contextlib
import itertools
import json
import math
import pickle
import typing as t
import warnings
from collections.abc import Generator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

import pynonthermal
from pynonthermal import spencerfano


def test_grid_validation() -> None:
    with pytest.raises(ValueError, match="npts must be at least 2"):
        pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=1)
    with pytest.raises(ValueError, match="emin_ev must be greater than zero"):
        pynonthermal.SpencerFanoSolver(emin_ev=0, emax_ev=3000, npts=100)
    with pytest.raises(ValueError, match="must be greater than emin_ev"):
        pynonthermal.SpencerFanoSolver(emin_ev=3000, emax_ev=1, npts=100)


def test_ionpot_below_emin_rejected() -> None:
    # Fe I has 7.9 and 9.0 eV shells. KF92 require all thresholds above the
    # low-energy cutoff, so a grid starting above them would not conserve energy.
    with (
        pynonthermal.SpencerFanoSolver(emin_ev=12.0, emax_ev=3000, npts=200) as sf,
        pytest.raises(ValueError, match="below emin_ev"),
    ):
        sf.add_ionisation(26, 1, n_ion=1.0)

    # the same ion is accepted once the cutoff is below its lowest shell, and then conserves energy.
    # npts has to resolve the 7.9 and 9.0 eV shells: at npts=1000 the 3 eV grid spacing leaves the
    # fractions 5.6% short of unity, almost all of it in the ionisation term.
    with pynonthermal.SpencerFanoSolver(emin_ev=7.9, emax_ev=3000, npts=4000) as sf:
        sf.add_ionisation(26, 1, n_ion=0.99)
        sf.add_ionisation(26, 2, n_ion=0.01)
        sf.solve(deposition_ev_per_s_per_cm3=1e6)
        assert math.isclose(sf.get_frac_sum(), 1.0, abs_tol=0.05)


def test_zero_free_electron_density() -> None:
    # a plasma of only neutral ions has no thermal electron loss channel
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        sf.add_ionisation(2, 1, n_ion=1.0)
        assert sf.get_n_e() == 0.0
        with pytest.raises(ValueError, match="free electron density is zero"):
            sf.solve(deposition_ev_per_s_per_cm3=100)

        # override_n_e() is the documented way out, but must be a usable density
        for badvalue in (0.0, -1.0):
            with pytest.raises(ValueError, match="override_n_e must be greater than zero"):
                sf.override_n_e(badvalue)

        sf.override_n_e(1e-4)
        sf.solve(deposition_ev_per_s_per_cm3=100)
        assert sf.get_n_e() == 1e-4

    # the loss function itself also rejects a non-positive density
    with pytest.raises(ValueError, match="positive finite free electron density"):
        pynonthermal.electronlossfunction(100.0, 0.0)


def test_deposition_rate_validated() -> None:
    # every fraction and rate coefficient is divided by the deposition rate density. Zero raised a bare
    # ZeroDivisionError from inside the analysis, and a negative value was accepted outright: it flipped
    # the sign of yvec and of every rate coefficient while still leaving the fractions summing to one,
    # so the conservation diagnostic never noticed.
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=300, npts=100) as sf:
        sf.add_ionisation(8, 2, n_ion=1e8)
        for badvalue in (0.0, -100.0):
            with pytest.raises(ValueError, match="deposition_ev_per_s_per_cm3 must be greater than zero"):
                sf.solve(deposition_ev_per_s_per_cm3=badvalue)

        sf.solve(deposition_ev_per_s_per_cm3=1e8)
        assert (sf.yvec > 0.0).all()
        assert sf.get_ionisation_ratecoeff(8, 2) > 0.0


def test_solve_inputs_reject_nonfinite() -> None:
    # nan slipped past the old "<= 0.0" guards because nan comparisons are always False, and inf
    # passed them outright; either then contaminated the solution without any exception being raised
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=300, npts=100) as sf:
        sf.add_ionisation(8, 2, n_ion=1e8)
        for badvalue in (math.nan, math.inf):
            with pytest.raises(ValueError, match="deposition_ev_per_s_per_cm3 must be greater than zero and finite"):
                sf.solve(deposition_ev_per_s_per_cm3=badvalue)
            with pytest.raises(ValueError, match="override_n_e must be greater than zero and finite"):
                sf.override_n_e(badvalue)


def test_solve_upper_triangular_guards() -> None:
    # scipy.linalg.solve_triangular raised on non-finite entries and zero diagonals; the numpy
    # back-substitution that replaced it must do the same rather than return nan/inf solutions
    a = np.triu(np.ones((3, 3)))
    b = np.ones(3)
    assert np.allclose(spencerfano.solve_upper_triangular(a, b), [0.0, 0.0, 1.0])

    a_nonfinite = a.copy()
    a_nonfinite[0, 2] = math.nan
    with pytest.raises(ValueError, match="finite"):
        spencerfano.solve_upper_triangular(a_nonfinite, b)

    with pytest.raises(ValueError, match="finite"):
        spencerfano.solve_upper_triangular(a, np.array([1.0, math.inf, 1.0]))

    a_singular = a.copy()
    a_singular[1, 1] = 0.0
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        spencerfano.solve_upper_triangular(a_singular, b)


def test_excitation_inputs_validated() -> None:
    npts = 100
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=300, npts=npts) as sf:
        sf.add_ionisation(8, 2, n_ion=1e8)
        xs_vec = np.full(npts, 1e-16)

        # a non-positive transition energy puts entries below the diagonal, where the triangular
        # solve silently discards them, and the solution comes out looking plausible
        for badvalue in (-50.0, 0.0):
            with pytest.raises(ValueError, match="epsilon_trans_ev must be greater than zero"):
                sf.add_excitation(8, 2, levelnumberdensity=1e10, xs_vec=xs_vec, epsilon_trans_ev=badvalue)

        # above the top of the grid there is no electron that could drive the transition
        with pytest.raises(ValueError, match="above the top of the energy grid"):
            sf.add_excitation(8, 2, levelnumberdensity=1e10, xs_vec=xs_vec, epsilon_trans_ev=301.0)

        # negative populations and cross sections give a negative excitation fraction
        with pytest.raises(ValueError, match="levelnumberdensity must be non-negative"):
            sf.add_excitation(8, 2, levelnumberdensity=-1e10, xs_vec=xs_vec, epsilon_trans_ev=20.0)
        with pytest.raises(ValueError, match="xs_vec must be non-negative"):
            sf.add_excitation(8, 2, levelnumberdensity=1e10, xs_vec=-xs_vec, epsilon_trans_ev=20.0)

        # a transition exactly at the top of the grid is still legal
        sf.add_excitation(8, 2, levelnumberdensity=1e10, xs_vec=xs_vec, epsilon_trans_ev=300.0)
        assert np.array_equal(sf.sfmatrix, np.triu(sf.sfmatrix))


def test_N_e_excitation_above_grid() -> None:
    # eq. 6 evaluates y and the cross section at energy_ev + epsilon_trans_ev. Above the top of the
    # grid no such electron exists, but get_energyindex_lteq clamps to the last bin, so the term came
    # out equal to its value at emax_ev instead of zero. This is the same clamp that the second
    # ionisation integral is already guarded against.
    npts, emax = 300, 300.0
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=emax, npts=npts) as sf:
        epsilon_trans_ev = 299.5
        sf.add_excitation(
            26,
            2,
            levelnumberdensity=1e8,
            xs_vec=np.where(sf.engrid_mid >= epsilon_trans_ev, 1e-16, 0.0),
            epsilon_trans_ev=epsilon_trans_ev,
        )
        sf.override_n_e(1e8)
        sf.solve(deposition_ev_per_s_per_cm3=1e8)

        # at 0.5 eV the excited electron is exactly at emax_ev and the term is real
        assert sf.calculate_N_e(emax - epsilon_trans_ev) > 0.0
        # above that it would have to have come from an electron the solver never had
        for energy_ev in (0.75, 1.0):
            assert energy_ev + epsilon_trans_ev > emax
            assert sf.calculate_N_e(energy_ev) == 0.0


def test_override_n_e_not_confused_with_cache() -> None:
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        sf.add_ionisation(8, 2, n_ion=1e8)
        sf.add_ionisation(8, 3, n_ion=1e8)
        assert sf.calculate_free_electron_density() == 3e8

        sf.override_n_e(1e6)
        sf.solve(deposition_ev_per_s_per_cm3=1e8)
        assert sf.get_n_e() == 1e6

        # clearing the override falls back to the ion populations
        sf.override_n_e(None)
        sf.solve(deposition_ev_per_s_per_cm3=1e8)
        assert sf.get_n_e() == 3e8


def test_n_e_cache_invalidated_by_later_adds() -> None:
    # reading n_e between add_* calls must not freeze it at the earlier value
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        sf.set_temperature(3000)
        sf.set_atomic_data(use_collstrengths=False)
        sf.add_ionisation(8, 2, n_ion=1e8)
        assert sf.get_n_e() == 1e8
        sf.add_ionisation(8, 3, n_ion=1e8)
        assert sf.get_n_e() == 3e8
        sf.add_ion_excitation(26, 3, n_ion=5e7)
        assert sf.get_n_e() == 4e8


def test_N_e_empty_second_integral_domain() -> None:
    # KF92 eq. 6 has an integral over [2E + I, E_max]. When 2E + I lies beyond the
    # top of the grid that domain is empty, but get_energyindex_lteq clamps to the last bin,
    # which used to add one spurious term. Na XI's single shell (I = 1465 eV) on a 1400-3000 eV
    # grid makes the domain empty for every E > 767.5 eV. The spurious term is in proportion to the square
    # of the grid spacing. With 150 points it is 2.4e-3 of N(E), which is 20 times the quadrature error of
    # the sub-grid of the solver.
    with pynonthermal.SpencerFanoSolver(emin_ev=1400, emax_ev=3000, npts=150) as sf:
        sf.add_ionisation(11, 10, n_ion=1e8)
        sf.solve(deposition_ev_per_s_per_cm3=1e8)

        shell = (
            pynonthermal.collion.read_colliondata()
            .filter((pl.col("Z") == 11) & (pl.col("ion_stage") == 10))
            .to_dicts()[0]
        )
        ionpot_ev = float(shell["ionpot_ev"])
        J = pynonthermal.collion.get_J(11, 10, ionpot_ev)

        energy_ev = 1200.0
        assert 2 * energy_ev + ionpot_ev > sf.engrid[-1] + sf.deltaen  # second domain is empty

        # reference: eq. 6 with only the first integral, epsilon in [I, min(E_max - E, E + I)],
        # integrated on a grid far finer than the solver's own
        enlambda = min(float(sf.engrid[-1]) - energy_ev, energy_ev + ionpot_ev)
        arr_epsilon: npt.NDArray[np.float64] = np.linspace(ionpot_ev, enlambda, num=20001)
        arr_e_p: npt.NDArray[np.float64] = energy_ev + arr_epsilon
        arr_y: npt.NDArray[np.float64] = np.interp(arr_e_p, sf.engrid_mid, sf.yvec, left=0.0, right=0.0)
        arr_xs = pynonthermal.collion.get_arxs_array_shell(arr_e_p, shell)
        # the closed form of Psecondary, written out so that the reference does not depend on the
        # implementation it is checking
        arr_p = 1.0 / J / np.arctan((arr_e_p - ionpot_ev) / 2 / J) / (1 + ((arr_epsilon - ionpot_ev) / J) ** 2)
        expected = 1e8 * float(np.trapezoid(arr_y * arr_xs * arr_p, arr_epsilon))

        # the tolerance covers the solver's own coarser sub-grids for these integrals (up to 2e-4 for
        # this integral, which crosses the source region), and is tight enough to notice the empty-domain
        # guard going away (asserted just below)
        got = sf.calculate_N_e(energy_ev)
        assert math.isclose(got, expected, rel_tol=5e-4)

        # reproduce the term the old clamping added, and check the result would no longer pass
        clamped_index = sf.get_energyindex_lteq(2 * energy_ev + ionpot_ev)
        spurious = 1e8 * float(
            sf.deltaen
            * sf.yvec[clamped_index]
            * pynonthermal.collion.get_arxs_array_shell(sf.engrid, shell)[clamped_index]
            * pynonthermal.collion.Psecondary_vec(
                e_p=float(sf.engrid[clamped_index]), e_s=energy_ev, ionpot_ev=ionpot_ev, J=J
            )
        )
        assert spurious > 2e-3 * expected
        assert not math.isclose(got + spurious, expected, rel_tol=5e-4)


def test_N_e_second_integral_ends_at_emax() -> None:
    # The second integral of KF92 eq. 6 runs over [2E + I, E_max]. It becomes empty at E = (E_max - I) / 2, so
    # N(E) is continuous there. The solution is at the middle of each grid interval, and the last of these
    # energies is half an interval above E_max. With that energy as the end of the integral, N(E) had a step
    # of 1.2 times the term below, which is half an interval at E_max.
    with pynonthermal.SpencerFanoSolver(emin_ev=1400, emax_ev=3000, npts=150) as sf:
        sf.add_ionisation(11, 10, n_ion=1e8)
        sf.solve(deposition_ev_per_s_per_cm3=1e8)
        channel = sf._ionisation_channels[(11, 10)][0]
        e_max = float(sf.engrid[-1])
        energy_empty_ev = (e_max - channel.ionpot_ev) / 2
        half_interval_term = 1e8 * float(
            0.5
            * sf.deltaen
            * sf.yvec[-1]
            * channel.evaluate_xs(np.array([e_max]))[0]
            * pynonthermal.collion.Psecondary_vec(
                e_p=e_max, e_s=energy_empty_ev, ionpot_ev=channel.ionpot_ev, J=channel.J_ev
            )
        )
        assert half_interval_term > 1e-4 * sf.calculate_N_e(energy_empty_ev)
        step = sf.calculate_N_e(energy_empty_ev - 1e-7) - sf.calculate_N_e(energy_empty_ev + 1e-7)
        assert abs(step) < 1e-3 * half_interval_term


def test_N_e_epsilon_integral_not_keyed_to_the_grid() -> None:
    # The epsilon integral of eq. 6 spans at most emin_ev, so it is narrower than one cell of a
    # typical energy grid and needs a sub-grid of its own. Evaluating it on the solver's own grid
    # instead made the answer depend on where the grid points happened to fall relative to the
    # ionisation threshold rather than on the resolution. Al I has a 6.0 eV shell, and npts of 600,
    # 1200, 1800 and 2400 each put a grid point within 10 meV of it on a 1-3000 eV grid, where the
    # Psecondary normalisation 1/atan((e_p - I) / 2J) diverges. That gave frac_sum of 1.86, 1.37,
    # 1.15 and 1.07 at those four npts while the next npts up gave 0.99-1.00 each time.
    # assert the premise rather than trusting the hard-coded npts: each of the first of these pairs
    # must put a grid point within a few meV of an Al I threshold, and each second one must not
    ionpots = [
        float(row["ionpot_ev"])
        for row in pynonthermal.collion.read_colliondata()
        .filter((pl.col("Z") == 13) & (pl.col("ion_stage") == 1))
        .to_dicts()
    ]

    def distance_to_threshold(npts: int) -> float:
        engrid = np.linspace(1.0, 3000.0, num=npts)
        return min(float(np.abs(engrid - ionpot).min()) for ionpot in ionpots)

    pairs = ((600, 700), (1200, 1300), (1800, 1900), (2400, 2900))
    for npts_onthreshold, npts_neighbour in pairs:
        assert distance_to_threshold(npts_onthreshold) < 0.01, (
            f"npts={npts_onthreshold} is no longer on an Al I threshold; this test needs one that is"
        )
        assert distance_to_threshold(npts_neighbour) > 0.05, (
            f"npts={npts_neighbour} is itself on an Al I threshold, so it is not a clean control"
        )

    fracsums: dict[int, float] = {}
    for npts in [npts for pair in pairs for npts in pair]:
        with (
            pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=npts) as sf,
            # the coarsest grids here are genuinely under-resolved, so the conservation diagnostic
            # fires as designed. This test is about the spikes, not about the residual error.
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", UserWarning)
            sf.add_ionisation(13, 1, n_ion=1.0)
            sf.override_n_e(1e-4)
            sf.solve(deposition_ev_per_s_per_cm3=100)
            fracsums[npts] = sf.get_frac_sum()

    for npts_onthreshold, npts_neighbour in pairs:
        assert math.isclose(fracsums[npts_onthreshold], fracsums[npts_neighbour], abs_tol=0.03), (
            f"npts={npts_onthreshold} gives frac_sum={fracsums[npts_onthreshold]} but the neighbouring"
            f" npts={npts_neighbour} gives {fracsums[npts_neighbour]}"
        )

    # the terms of the matrix give the energy below E_0, so the sum is one to rounding on each grid
    for npts, fracsum in fracsums.items():
        assert math.isclose(fracsum, 1.0, abs_tol=1e-9), npts


def test_N_e_epsilon_integral_value_on_a_narrow_domain() -> None:
    # The checks above pin the symptom (no spikes) rather than the value, so they would still pass if
    # the epsilon integral returned zero. Pin the value itself, in the regime the fix is for: emin_ev=1
    # makes the domain [I, E + I] at most 1 eV wide, narrower than the 0.75 eV grid spacing here.
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=4000) as sf:
        sf.add_ionisation(13, 1, n_ion=1.0)
        sf.override_n_e(1e-4)
        sf.solve(deposition_ev_per_s_per_cm3=100)

        shells = (
            pynonthermal.collion.read_colliondata().filter((pl.col("Z") == 13) & (pl.col("ion_stage") == 1)).to_dicts()
        )
        energy_ev = 0.7
        emax = float(sf.engrid[-1])

        def psecondary(
            arr_e_p: npt.NDArray[np.float64], e_s: npt.NDArray[np.float64] | float, ionpot_ev: float, J: float
        ) -> npt.NDArray[np.float64]:
            # the closed form, so the reference does not depend on the implementation it is checking
            return 1.0 / J / np.arctan((arr_e_p - ionpot_ev) / 2 / J) / (1 + (np.asarray(e_s) / J) ** 2)

        # both eq. 6 integrals over every shell, each on a grid far finer than the solver's own
        expected = 0.0
        narrow_domain_seen = False
        for shell in shells:
            ionpot_ev = float(shell["ionpot_ev"])
            J = pynonthermal.collion.get_J(13, 1, ionpot_ev)

            enlambda = min(emax - energy_ev, energy_ev + ionpot_ev)
            if enlambda > ionpot_ev:
                narrow_domain_seen |= (enlambda - ionpot_ev) < sf.deltaen
                arr_epsilon = np.linspace(ionpot_ev, enlambda, num=20001)
                arr_e_p = arr_epsilon + energy_ev
                arr_y = np.interp(arr_e_p, sf.engrid_mid, sf.yvec, left=0.0, right=0.0)
                arr_xs = pynonthermal.collion.get_arxs_array_shell(arr_e_p, shell)
                arr_p = psecondary(arr_e_p, arr_epsilon - ionpot_ev, ionpot_ev, J)
                expected += float(np.trapezoid(arr_y * arr_xs * arr_p, arr_epsilon))

            en_lower2 = 2 * energy_ev + ionpot_ev
            if en_lower2 < emax:
                arr_e_p2 = np.linspace(en_lower2, emax, num=40001)
                arr_y2 = np.interp(arr_e_p2, sf.engrid_mid, sf.yvec, left=0.0, right=0.0)
                arr_xs2 = pynonthermal.collion.get_arxs_array_shell(arr_e_p2, shell)
                arr_p2 = psecondary(arr_e_p2, energy_ev, ionpot_ev, J)
                expected += float(np.trapezoid(arr_y2 * arr_xs2 * arr_p2, arr_e_p2))

        # the epsilon domain really is narrower than one grid cell here, the regime the sub-grid is for
        assert narrow_domain_seen
        assert expected > 0.0
        assert math.isclose(sf.calculate_N_e(energy_ev), expected, rel_tol=0.02)


def test_quadrature_resolution_constants_are_adequate() -> None:
    # The constants of the sub-grids of calculate_N_e() (NPTS_SUBGRID_MIN,
    # NPTS_SUBGRID_MAX, NPTS_SUBGRID_PER_J, and NCELLS_SECOND_INTEGRAL_SUBGRID) carry accuracy claims in their
    # comments. The test compares N(E) at the configured values with N(E) at much finer values.
    # Then a lower value that buys speed cannot silently degrade N(E).
    def n_e_below_e0(emin_ev: float, Z: int) -> float:
        with pynonthermal.SpencerFanoSolver(emin_ev=emin_ev, emax_ev=3000, npts=2000) as sf:
            sf.add_ionisation(Z, 1, n_ion=0.99)
            sf.add_ionisation(Z, 2, n_ion=0.01)
            sf.solve(deposition_ev_per_s_per_cm3=1e6)
            return sf.calculate_N_e(0.5 * emin_ev)

    # emin_ev has to cover the large values add_ionisation's own error message steers users towards
    # ("set emin_ev <= {lowest ionpot} eV"), not just the default grid
    for emin_ev, Z in ((7.9, 26), (15.7, 18), (24.5, 2)):
        asconfigured = n_e_below_e0(emin_ev, Z)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(spencerfano, "NPTS_SUBGRID_MIN", 257)
            mp.setattr(spencerfano, "NPTS_SUBGRID_MAX", 4097)
            mp.setattr(spencerfano, "NPTS_SUBGRID_PER_J", 128)
            mp.setattr(spencerfano, "NCELLS_SECOND_INTEGRAL_SUBGRID", 8)
            converged = n_e_below_e0(emin_ev, Z)
        assert math.isclose(asconfigured, converged, rel_tol=1e-3), (
            f"emin_ev={emin_ev}: the configured resolution gives N(E)={asconfigured} against a converged {converged}"
        )


def test_lotz_xs_relativistic() -> None:
    # the Lotz/Axelrod cross section must fall off smoothly rather than dropping to zero at the
    # 255 keV energy where the classical beta^2 = 2E/mc^2 reaches one
    shell = (
        pynonthermal.collion.read_colliondata()
        .filter((pl.col("Z") == 56) & (pl.col("ion_stage") == 2) & (pl.col("n") < 0))
        .to_dicts()[0]
    )
    arr_en_ev = np.array([1e2, 1e3, 1e4, 1e5, 2.54e5, 2.6e5, 5.11e5, 1e6])
    xs_vec = pynonthermal.collion.get_arxs_array_shell(arr_en_ev, shell)

    assert (xs_vec > 0.0).all()
    assert np.isfinite(xs_vec).all()
    # the cross section must fall monotonically through the former 255 keV cliff
    assert (np.diff(xs_vec) < 0.0).all()


def test_excitation_xs_zero_above_grid() -> None:
    # a transition that no electron on the grid can drive must have a zero cross section
    # everywhere, rather than a spurious (and for the E1 branch, negative) value at the top point
    engrid = np.linspace(1.0, 100.0, 200)
    for row in (
        {"collstr": -1, "epsilon_trans_ev": 500.0, "forbidden": 0, "lower_g": 1, "upper_g": 3, "A": 1e8},
        {"collstr": 1.0, "epsilon_trans_ev": 500.0, "forbidden": 1, "lower_g": 1, "upper_g": 3, "A": 0.0},
    ):
        xs_vec = pynonthermal.excitation.get_xs_excitation_vector(engrid, row)
        assert not xs_vec.any()

    # a transition right at the top of the grid is still open
    row_attop = {"collstr": -1, "epsilon_trans_ev": 100.0, "forbidden": 0, "lower_g": 1, "upper_g": 3, "A": 1e8}
    assert pynonthermal.excitation.get_xs_excitation_vector(engrid, row_attop)[-1] > 0.0

    # no cross section anywhere may be negative for real atomic data on a grid far below the
    # highest transition energy
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=5.0, npts=300) as sf:
        sf.set_temperature(3000)
        # the collision-strength cross sections of LHD12, which no other test reaches
        sf.set_atomic_data(use_collstrengths=True)
        sf.add_ionisation(26, 3, n_ion=0.7)
        sf.add_ion_excitation(26, 3, n_ion=0.7)
        for transitions in sf.excitationlists.values():
            for trans in transitions.values():
                assert trans.epsilon_trans_ev <= sf.engrid[-1]
                assert (trans.xs_vec >= 0.0).all()


def test_ionisation_ratecoeff_matches_eff_ionpot_form() -> None:
    # the rate coefficient is the direct integral of y(E) sigma(E) dE over the shells. For a
    # populated ion that equals KF92 equation 13 with the effective potential.
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=400) as sf:
        sf.add_ionisation(8, 1, n_ion=9e8)
        sf.add_ionisation(8, 2, n_ion=1e8)
        sf.add_ionisation(26, 2, n_ion=2e7)
        sf.solve(deposition_ev_per_s_per_cm3=1e9)
        n_ion_tot = sf.get_n_ion_tot()
        for Z, ion_stage in sf.ionpopdict:
            ratecoeff = sf.get_ionisation_ratecoeff(Z, ion_stage)
            assert ratecoeff > 0.0
            assert math.isclose(ratecoeff, 1e9 / n_ion_tot / sf.get_eff_ionpot(Z, ion_stage), rel_tol=1e-12)


def test_excitation_only_ion_counted() -> None:
    # an ion given an excitation channel but no ionisation channel still spends deposited energy,
    # so its excitation fraction has to appear in the totals
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=500) as sf:
        sf.set_temperature(3000)
        sf.set_atomic_data(use_collstrengths=False)
        sf.add_ionisation(8, 2, n_ion=1e8)
        xs_vec = np.where(sf.engrid_mid >= 20.0, 1e-16, 0.0)
        sf.add_excitation(26, 2, levelnumberdensity=1e8, xs_vec=xs_vec, epsilon_trans_ev=20.0)
        sf.solve(deposition_ev_per_s_per_cm3=1e8)

        assert sf.get_frac_excitation_tot() > 0.3
        assert math.isclose(sf.get_frac_sum(), 1.0, abs_tol=0.01)


def test_ltepopexcitation_registers_population() -> None:
    # add_ion_excitation() without add_ionisation() must still contribute the ion's free
    # electrons and nuclei, and must not invent an ionisation channel for it
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=600) as sf:
        sf.set_temperature(3000)
        sf.set_atomic_data(use_collstrengths=False)
        sf.add_ionisation(8, 2, n_ion=1e8)
        sf.add_ion_excitation(26, 3, n_ion=5e7)

        assert sf.ionpopdict[(26, 3)] == 5e7
        assert sf.get_n_e() == 1e8 * 1 + 5e7 * 2
        assert sf.get_n_ion_tot() == 1.5e8

        sf.solve(deposition_ev_per_s_per_cm3=1e8)
        assert sf.get_frac_excitation_tot() > 0.0
        assert sf.get_ionisation_ratecoeff(26, 3) == 0.0
        assert sf.get_eff_ionpot(26, 3) == float("inf")
        assert math.isclose(sf.get_frac_sum(), 1.0, abs_tol=0.01)

    # a negative population is rejected on the excitation path too
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        sf.set_temperature(3000)
        sf.set_atomic_data(use_collstrengths=False)
        with pytest.raises(ValueError, match="non-negative"):
            sf.add_ion_excitation(26, 3, n_ion=-1.0)


def test_population_mismatch_checked_before_atomic_data() -> None:
    # a population that conflicts with an earlier registration is rejected before the level data is
    # read, so the failed call neither caches a level table nor changes the populations
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        sf.set_temperature(3000)
        sf.set_atomic_data(use_collstrengths=False)
        sf.add_ionisation(8, 2, n_ion=1e8)
        with pytest.raises(ValueError, match="different populations"):
            sf.add_ion_excitation(8, 2, n_ion=5e7)
        assert sf.adata_polars is None
        assert sf.ionpopdict == {(8, 2): 1e8}
        assert not sf.excitationlists


def test_energy_conservation_on_coarse_grid() -> None:
    # The matrix and the analysis count each energy loss with the same weights, so the energy fractions sum
    # to one also on a grid that is too coarse. With the ionisation terms at the grid energies alone, the sum
    # of this helium plasma was 6.5 % from one at npts=100.
    x_e = 1e-4
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=100) as sf:
        sf.set_temperature(3000)
        sf.set_atomic_data(use_collstrengths=False)
        for Z, ion_stage, n_ion in ((2, 1, 1.0 - x_e), (2, 2, x_e)):
            sf.add_ionisation(Z, ion_stage, n_ion=n_ion)
            sf.add_ion_excitation(Z, ion_stage, n_ion=n_ion)
        sf.solve(deposition_ev_per_s_per_cm3=100)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            frac_sum = sf.get_frac_sum()
        assert math.isclose(frac_sum, 1.0, abs_tol=1e-9)


def test_conservation_warning() -> None:
    # the warning shows a defect of the energy count. Here the test removes the heating to make one.
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=100) as sf:
        sf.add_ionisation(2, 1, n_ion=1.0)
        sf.override_n_e(1e-4)
        sf.solve(deposition_ev_per_s_per_cm3=100)
        sf._frac_heating = 0.0
        with pytest.warns(UserWarning, match="energy fractions sum to"):
            sf.analyse_ntspectrum()


def test_opal_J_applies_to_every_shell_of_its_ion() -> None:
    # The measured secondary-electron widths of OPB71 are whole-atom values dominated by
    # valence-shell ionisation, but ARTIS applies each to every shell of its ion and pynonthermal
    # deliberately matches that, so Ne I's 48.5 eV 2s shell gets the same J as its 21.6 eV 2p shell.
    assert pynonthermal.collion.get_J(2, 1, 24.6) == 15.8  # He I, its only shell
    for ionpot_ev in (21.6, 48.5):  # Ne I 2p and 2s
        assert pynonthermal.collion.get_J(10, 1, ionpot_ev) == 24.2
    for ionpot_ev in (15.8, 29.2):  # Ar I 3p and 3s
        assert pynonthermal.collion.get_J(18, 1, ionpot_ev) == 10.0

    # every other ion, including the ionised stages of those three, uses the general rule
    for Z, ion_stage, ionpot_ev in ((26, 2, 16.18), (2, 2, 54.4), (10, 2, 41.0), (18, 2, 27.6)):
        assert pynonthermal.collion.get_J(Z, ion_stage, ionpot_ev) == 0.6 * ionpot_ev


def test_psecondary_requires_an_energy() -> None:
    # this guard was an assert, so it vanished under python -O and left the sentinel default of -1
    # to be used as a real secondary energy
    with pytest.raises(ValueError, match="e_s or the primary energy loss epsilon"):
        pynonthermal.collion.Psecondary(e_p=100.0, ionpot_ev=10.0, J=6.0)

    # the scalar and vectorised forms have to agree, since the solver uses both
    e_s = np.array([0.0, 5.0, 20.0])
    arr = pynonthermal.collion.Psecondary_vec(e_p=100.0, e_s=e_s, ionpot_ev=10.0, J=6.0)
    for i, value in enumerate(e_s):
        assert math.isclose(
            float(arr[i]), pynonthermal.collion.Psecondary(e_p=100.0, e_s=float(value), ionpot_ev=10.0, J=6.0)
        )

    # below threshold there are no secondaries
    assert not pynonthermal.collion.Psecondary_vec(e_p=np.array([5.0, 10.0]), e_s=0.0, ionpot_ev=10.0, J=6.0).any()


def test_n_e_nt_relativistic() -> None:
    # the electron speed used to be the classical sqrt(2E/m_e), which exceeds c above 511 keV and is
    # already 2.5 per cent fast at the 16 keV emax_ev that the iron benchmark uses
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=16000, npts=400) as sf:
        sf.add_ionisation(26, 2, n_ion=1.0)
        sf.solve(deposition_ev_per_s_per_cm3=1e8)
        n_e_nt = sf.get_n_e_nt()
        assert n_e_nt > 0.0

        # the classical speed exceeds the relativistic one, so it gives a lower density. Comparing
        # against it pins the physics without restating the implementation.
        # yvec[j] is the solution at the middle of grid interval j
        arr_velocity_classical = np.sqrt(2 * sf.engrid_mid * pynonthermal.constants.EV / pynonthermal.constants.ME)
        assert n_e_nt > float(np.sum(sf.yvec / arr_velocity_classical) * sf.deltaen)

        # and no electron may travel faster than light, which the classical form allows above 511 keV
        assert (pynonthermal.base.get_betasq(sf.engrid_mid) < 1.0).all()
        assert n_e_nt > float(np.sum(sf.yvec) * sf.deltaen / pynonthermal.constants.CLIGHT)


def test_lossfunction_plasma_energy_guard() -> None:
    # the high-energy branch is ln(2E / hbar*omega_p), which is zero or negative once the plasma
    # energy reaches 2E. That was an assert, so under python -O it returned a non-positive loss rate.
    with pytest.raises(ValueError, match="not valid at"):
        pynonthermal.electronlossfunction(100.0, 1e30)

    # for densities the solver is actually used at, the loss rate is positive and falls with energy
    arr_loss = [pynonthermal.electronlossfunction(en_ev, 1e8) for en_ev in (1.0, 5.0, 14.0, 100.0, 3000.0)]
    assert all(loss > 0.0 for loss in arr_loss)
    assert all(b < a for a, b in itertools.pairwise(arr_loss))


def _reference_excitation_fill(
    sf: pynonthermal.SpencerFanoSolver,
    levelnumberdensity: float,
    xs_vec: npt.NDArray[np.float64],
    epsilon_trans_ev: float,
) -> npt.NDArray[np.float64]:
    # straightforward row-by-row construction of one transition's contribution: row i covers k
    # full-width bins from max(i, threshold index) plus a partial final bin of weight frac,
    # truncated at the top of the grid where every remaining bin is full width. Kept independent
    # of the production code so that a regression in the banded fill cannot hide here.
    engrid = sf.engrid
    npts = len(engrid)
    deltaen = sf.deltaen
    contribution = np.zeros((npts, npts))
    vec = levelnumberdensity * deltaen * xs_vec
    # the threshold bin is the first bin whose middle energy reaches the transition energy, and no bin
    # below it contributes, also not as the partial bin of a row
    xsstartindex = int(np.searchsorted(sf.engrid_mid, epsilon_trans_ev))
    vec[:xsstartindex] = 0.0
    k = int(epsilon_trans_ev / deltaen)
    frac = epsilon_trans_ev / deltaen - k
    for i in range(npts):
        startindex = max(i, xsstartindex)
        if i + k < npts:
            contribution[i, startindex : i + k] += vec[startindex : i + k]
            contribution[i, i + k] += vec[i + k] * frac
        else:
            contribution[i, startindex:] += vec[startindex:]
    return contribution


def test_excitation_band_fill_matches_rowwise() -> None:
    # add_excitation writes each transition as one windowed band plus a partial-bin diagonal
    # (every row repeats the same vec[j] values over the overlap of its window with the next
    # row's). The result must be bit-identical to the row-by-row fill, including the rows
    # truncated by the excitation threshold and the rows clipped at the top of the grid.
    rng = np.random.default_rng(42)
    npts = 157
    deltaen = (300.0 - 1.0) / (npts - 1)  # approximate spacing, only used to place the cases below
    cases = [
        (1.0, 300.0, 0.5 * deltaen),  # narrower than one grid cell: band width zero, partial bins only
        (1.0, 300.0, 5 * deltaen),  # within rounding of an exact multiple of the grid spacing
        (1.0, 300.0, 13.3),  # a typical transition energy, not aligned with the grid
        (1.0, 300.0, 299.0),  # band nearly as wide as the whole grid
        (1.0, 300.0, 300.0),  # at the top of the grid: every row's window is clipped
        (200.0, 300.0, 250.0),  # band wider than the whole grid: every row is clipped and truncated
    ]
    for emin, emax, epsilon_trans_ev in cases:
        with pynonthermal.SpencerFanoSolver(emin_ev=emin, emax_ev=emax, npts=npts) as sf:
            # nonzero cross sections below threshold too: the solver zeroes them in the stored
            # array and in the matrix, and the partial-bin term is written for every row
            xs_vec = rng.random(npts) * 1e-16
            sf.add_excitation(8, 2, levelnumberdensity=1e5, xs_vec=xs_vec, epsilon_trans_ev=epsilon_trans_ev)
            stored = sf.excitationlists[(8, 2)][0].xs_vec
            assert not stored[: int(np.searchsorted(sf.engrid_mid, epsilon_trans_ev))].any()
            expected = _reference_excitation_fill(sf, 1e5, xs_vec, epsilon_trans_ev)
            assert sf.sfmatrix.tobytes() == expected.tobytes(), f"mismatch for {epsilon_trans_ev=}"


def _reference_ionisation_fill(
    sf: pynonthermal.SpencerFanoSolver, n_ion: float, channel: pynonthermal.IonisationChannel
) -> npt.NDArray[np.float64]:
    # the matrix terms of one channel from a full array for each row, with no column ranges. Each term is the
    # mean over one grid interval of the probability that the ionisation moves an electron across an energy.
    engrid = sf.engrid
    npts = len(engrid)
    deltaen = sf.deltaen
    ionpot_ev = channel.ionpot_ev
    J = channel.J_ev
    atan_epsilon = np.arctan((engrid - ionpot_ev) / 2.0 / J)
    with np.errstate(divide="ignore", invalid="ignore"):
        prefactors = np.divide(
            n_ion * channel.xs_grid * deltaen, atan_epsilon, out=np.zeros(npts), where=atan_epsilon != 0.0
        )
    prefactors[engrid < ionpot_ev] = 0.0
    epsilon_uppers = np.minimum((engrid + ionpot_ev) / 2, engrid)
    atan_uppers = np.arctan((epsilon_uppers - ionpot_ev) / J)

    def atan_integral(epsilon: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # the integral from ionpot_ev to epsilon of atan[(x - ionpot_ev) / J]. Above the upper limit of
        # epsilon of each column, the function keeps its value at the limit.
        u = np.maximum(np.minimum(epsilon, epsilon_uppers) - ionpot_ev, 0.0) / J
        integral = J * (u * np.arctan(u) - 0.5 * np.log1p(u * u))
        return integral + atan_uppers * np.maximum(epsilon - epsilon_uppers, 0.0)

    def crossing_probability(epsilon_lower: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # times the atan norm: the mean over [epsilon_lower, epsilon_lower + deltaen] of the integral of the
        # distribution from that limit to the upper limit
        atan_mean = (atan_integral(epsilon_lower + deltaen) - atan_integral(epsilon_lower)) / deltaen
        return np.where(epsilon_lower < epsilon_uppers, atan_uppers - atan_mean, 0.0)

    contribution = np.zeros((npts, npts))
    for i, en in enumerate(engrid):
        columns = np.arange(npts) >= i
        contribution[i] += np.where(columns, prefactors * crossing_probability(engrid - en), 0.0)
        contribution[i] -= prefactors * crossing_probability(np.full(npts, en + ionpot_ev))
    return contribution


def test_ionisation_fill_holds_the_energy_of_each_ionisation() -> None:
    # The sum of the rows of a column times deltaen is the energy that the matrix takes from the heating for
    # one ionisation in that column: the loss of the primary less the energy of the secondary above E_0. Its
    # mean is ionpot_ev plus the mean of min(E_s, E_0). The test gets that mean from the distribution of KF92
    # equation 4. With the terms at the grid energies alone, the 3d shell of Fe I had 0.17 eV less than
    # ionpot_ev on the default grid, and the energy fractions summed to more than one.
    for emin, emax, npts, Z, ion_stage in [
        (0.1, 16000.0, 1024, 26, 1),
        (1.0, 3000.0, 311, 8, 2),
        (7.9, 3000.0, 400, 26, 2),
    ]:
        with pynonthermal.SpencerFanoSolver(emin_ev=emin, emax_ev=emax, npts=npts) as sf:
            sf.add_ionisation(Z, ion_stage, n_ion=1.0)
            E_0 = float(sf.engrid[0])
            for channel in sf._ionisation_channels[(Z, ion_stage)]:
                matrix = np.zeros_like(sf.sfmatrix)
                sf._add_ionisation_channel_to_matrix(1.0, channel, matrix=matrix)
                ionpot_ev, J = channel.ionpot_ev, channel.J_ev
                columns = np.flatnonzero((channel.xs_grid > 0.0) & (sf.engrid > ionpot_ev))
                assert len(columns) > 10
                for j in columns[[0, 1, 2, len(columns) // 2, -1]]:
                    # the secondary has an energy E_s from zero to (E' - ionpot_ev) / 2. The mean of
                    # min(E_s, E_0) is the integral over x in [0, E_0] of the probability that E_s > x.
                    e_s_max = (sf.engrid[j] - ionpot_ev) / 2
                    arr_x = np.linspace(0.0, min(E_0, e_s_max), 20001)
                    arr_probability = 1.0 - np.arctan(arr_x / J) / math.atan(e_s_max / J)
                    mean_below_e0 = float(np.trapezoid(arr_probability, arr_x))
                    energy_ev = sf.deltaen * matrix[:, j].sum() / (channel.xs_grid[j] * sf.deltaen)
                    assert math.isclose(energy_ev, ionpot_ev + mean_below_e0, rel_tol=1e-7), (Z, channel.key, j)


def test_ltepop_excitation_grouped_fill() -> None:
    # add_ion_excitation writes the matrix once per distinct band width, with the vectors
    # of same-width transitions pre-summed. Rebuilding from the stored per-transition data must
    # agree to within summation-order rounding.
    npts = 400
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=npts) as sf:
        sf.set_temperature(3000)
        sf.set_atomic_data(use_collstrengths=False)
        sf.add_ion_excitation(2, 1, n_ion=1.0)
        assert len(sf.excitationlists[(2, 1)]) > 100  # the grouping must actually see many transitions
        expected = np.zeros((npts, npts))
        for trans in sf.excitationlists[(2, 1)].values():
            expected += _reference_excitation_fill(sf, trans.levelnumberdensity, trans.xs_vec, trans.epsilon_trans_ev)
        assert np.allclose(sf.sfmatrix, expected, rtol=1e-12, atol=0.0)


def test_ionisation_fill_matches_masked_reference() -> None:
    # _add_ionisation_channel_to_matrix writes only the column range of each integral in each row, and it
    # finds the ranges with pointers that only move forward. The matrix must agree with a fill from a full
    # array for each row. The tolerance is relative to the largest entry, because the two integrals cancel
    # in some entries.
    for emin, emax, npts, Z, ion_stage in [
        (1.0, 300.0, 193, 8, 2),
        (7.9, 16000.0, 217, 26, 2),  # emin well above 1 eV and a heavy ion with many shells
        (1.0, 3000.0, 149, 2, 1),  # helium: J takes the OPB71 special-case value
        (1.0, 3000.0, 300, 56, 2),  # Lotz shells only, some of them above the top of the grid
    ]:
        with pynonthermal.SpencerFanoSolver(emin_ev=emin, emax_ev=emax, npts=npts) as sf:
            n_ion = 1e5
            if Z == 56:
                with pytest.warns(pynonthermal.LotzApproximationWarning):
                    sf.add_ionisation(Z, ion_stage, n_ion=n_ion)
            else:
                sf.add_ionisation(Z, ion_stage, n_ion=n_ion)
            expected = np.zeros((npts, npts))
            for channel in sf._ionisation_channels[(Z, ion_stage)]:
                expected += _reference_ionisation_fill(sf, n_ion, channel)
            atol = 1e-12 * float(np.abs(expected).max())
            assert np.allclose(sf.sfmatrix, expected, rtol=1e-12, atol=atol), f"mismatch for {Z=} {ion_stage=}"


def test_solve_upper_triangular_diag_add() -> None:
    # solve() passes the free-electron loss function as a separate diagonal vector instead of
    # copying the whole matrix to add it, so the diag_add path must reproduce the added-in-place
    # result exactly
    rng = np.random.default_rng(7)
    n = 40
    a = np.triu(rng.random((n, n))) + np.diag(np.full(n, 2.0))
    b = rng.random(n)
    d = rng.random(n)
    expected = spencerfano.solve_upper_triangular(a + np.diag(d), b)
    assert np.array_equal(spencerfano.solve_upper_triangular(a, b, diag_add=d), expected)

    # the singularity check applies to the summed diagonal: a zero on the matrix diagonal is
    # fine when diag_add makes the sum non-zero...
    a_zerodiag = a.copy()
    a_zerodiag[3, 3] = 0.0
    assert np.isfinite(spencerfano.solve_upper_triangular(a_zerodiag, b, diag_add=d)).all()

    # ...and a cancellation to exactly zero is singular
    d_cancel = d.copy()
    d_cancel[3] = -a[3, 3]
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        spencerfano.solve_upper_triangular(a, b, diag_add=d_cancel)

    with pytest.raises(ValueError, match="finite"):
        spencerfano.solve_upper_triangular(a, b, diag_add=np.array([math.nan] + [0.0] * (n - 1)))


def _bethe_xs_grid(
    engrid: npt.NDArray[np.float64], ionpot_ev: float, scale_cm2_ev2: float = 3e-13
) -> npt.NDArray[np.float64]:
    # a plain Bethe-like cross section [cm^2] on the solver energy grid, which stands for a custom
    # cross section. It is zero at and below the ionisation potential, as every channel must be.
    out = np.zeros_like(engrid)
    above = engrid > ionpot_ev
    out[above] = scale_cm2_ev2 * np.log(engrid[above] / ionpot_ev) / (engrid[above] * ionpot_ev)
    return out


def test_custom_ionisation_channel_matches_the_builtin_path() -> None:
    # a custom channel must take exactly the same path as a shell of the built-in table. Adding the
    # built-in cross sections back one call at a time must give the same matrix, bit for bit, and
    # the same energy fractions after the solve.
    Z, ion_stage, n_ion = 8, 2, 1e8
    shells = (
        pynonthermal.collion.read_colliondata()
        .filter((pl.col("Z") == Z) & (pl.col("ion_stage") == ion_stage))
        .to_dicts()
    )
    assert len(shells) > 1  # the ion must have more than one shell for this to mean anything

    with (
        pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf_custom,
        pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf_builtin,
    ):
        sf_builtin.add_ionisation(Z, ion_stage, n_ion=n_ion)
        for shell in shells:
            sf_custom.add_ionisation_channel(
                Z,
                ion_stage,
                n_ion=n_ion,
                ionpot_ev=float(shell["ionpot_ev"]),
                xs_vec=pynonthermal.collion.get_arxs_array_shell(sf_custom.engrid_mid, shell),
            )

        assert sf_custom.sfmatrix.tobytes() == sf_builtin.sfmatrix.tobytes()

        for sf in (sf_custom, sf_builtin):
            sf.solve(deposition_ev_per_s_per_cm3=1e8)

        assert math.isclose(
            sf_custom.get_frac_ionisation_ion(Z, ion_stage), sf_builtin.get_frac_ionisation_ion(Z, ion_stage)
        )
        # frac_heating differs only through calculate_N_e, which needs the cross section between
        # the grid points and interpolates the array there instead of using the exact formula
        assert math.isclose(sf_custom.get_frac_heating(), sf_builtin.get_frac_heating(), rel_tol=1e-5)


def test_custom_ionisation_channel_conserves_energy() -> None:
    # the whole custom path, from the matrix fill to calculate_N_e, must still put every
    # deposited eV into heating, excitation, or ionisation
    ionpot_ev = 35.0
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=2000) as sf:
        sf.add_ionisation_channel(8, 2, n_ion=1e8, ionpot_ev=ionpot_ev, xs_vec=_bethe_xs_grid(sf.engrid_mid, ionpot_ev))
        sf.solve(deposition_ev_per_s_per_cm3=1e8)

        assert sf.get_frac_ionisation_tot() > 0.1  # the channel must carry a real share
        assert math.isclose(sf.get_frac_sum(), 1.0, abs_tol=0.01)

        # calculate_N_e integrates over a domain narrower than one grid cell just above the
        # ionisation potential, so the channel must give values between the grid points
        channel = sf._ionisation_channels[(8, 2)][0]
        off_grid = np.array([ionpot_ev * 1.001, ionpot_ev * 1.002, ionpot_ev * 1.003])
        assert not np.isin(off_grid, sf.engrid).any()
        assert (channel.xs(off_grid) > 0.0).all()
        # and it stays at zero below its own ionisation potential
        assert not channel.xs(np.array([ionpot_ev * 0.999, ionpot_ev])).any()


def test_custom_ionisation_channel_adds_to_the_builtin_shells() -> None:
    # add_ionisation() and add_ionisation_channel() may both be called for one ion
    Z, ion_stage, n_ion = 8, 2, 1e8
    with (
        pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf_builtin,
        pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf_extra,
    ):
        for sf in (sf_builtin, sf_extra):
            sf.add_ionisation(Z, ion_stage, n_ion=n_ion)
        sf_extra.add_ionisation_channel(
            Z,
            ion_stage,
            n_ion=n_ion,
            ionpot_ev=120.0,
            xs_vec=_bethe_xs_grid(sf_extra.engrid_mid, 120.0),
            channelkey="my inner shell",
        )

        assert (
            len(sf_extra._ionisation_channels[(Z, ion_stage)])
            == len(sf_builtin._ionisation_channels[(Z, ion_stage)]) + 1
        )
        for sf in (sf_builtin, sf_extra):
            sf.solve(deposition_ev_per_s_per_cm3=1e8)
        assert sf_extra.get_frac_ionisation_ion(Z, ion_stage) > sf_builtin.get_frac_ionisation_ion(Z, ion_stage)


def test_custom_ionisation_channel_validation() -> None:
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        good = _bethe_xs_grid(sf.engrid_mid, 35.0)

        # the cross section must be non-negative, finite, and on the whole grid
        with pytest.raises(ValueError, match="non-negative"):
            sf.add_ionisation_channel(8, 2, n_ion=1e8, ionpot_ev=35.0, xs_vec=-good)
        with pytest.raises(ValueError, match="finite"):
            sf.add_ionisation_channel(8, 2, n_ion=1e8, ionpot_ev=35.0, xs_vec=np.full_like(good, np.inf))
        with pytest.raises(ValueError, match="one value for each"):
            sf.add_ionisation_channel(8, 2, n_ion=1e8, ionpot_ev=35.0, xs_vec=good[:-1])

        # a cross section below the ionisation potential would count towards the ionisation
        # fraction without a matching loss term in the matrix
        with pytest.raises(ValueError, match="zero at and below its ionisation potential"):
            sf.add_ionisation_channel(8, 2, n_ion=1e8, ionpot_ev=35.0, xs_vec=np.full_like(good, 1e-18))

        # the ionisation potential must lie inside the energy grid
        with pytest.raises(ValueError, match="below emin_ev"):
            sf.add_ionisation_channel(8, 2, n_ion=1e8, ionpot_ev=0.5, xs_vec=_bethe_xs_grid(sf.engrid_mid, 0.5))
        with pytest.raises(ValueError, match="above the top of the energy grid"):
            sf.add_ionisation_channel(8, 2, n_ion=1e8, ionpot_ev=5000.0, xs_vec=np.zeros_like(good))
        with pytest.raises(ValueError, match="ionpot_ev must be greater than zero"):
            sf.add_ionisation_channel(8, 2, n_ion=1e8, ionpot_ev=-1.0, xs_vec=np.zeros_like(good))

        # a nan or infinite population used to reach ionpopdict and make every result nan
        for bad in (float("nan"), float("inf")):
            with pytest.raises(ValueError, match="non-negative and finite"):
                sf.add_ionisation_channel(8, 2, n_ion=bad, ionpot_ev=35.0, xs_vec=good)
            with pytest.raises(ValueError, match="non-negative and finite"):
                sf.add_ionisation(8, 2, n_ion=bad)

        # ion_stage below one gives a negative charge, which only a bare assert used to catch
        with pytest.raises(ValueError, match="ion_stage of Z=8 must be an integer between 1 and 9"):
            sf.add_ionisation_channel(8, 0, n_ion=1e8, ionpot_ev=35.0, xs_vec=good)
        with pytest.raises(ValueError, match="Z must be an integer of at least 1"):
            sf.add_ionisation_channel(0, 1, n_ion=1e8, ionpot_ev=35.0, xs_vec=good)
        # a stage above the bare nucleus, a non-integer stage, and the bare nucleus itself
        with pytest.raises(ValueError, match="between 1 and 9"):
            sf.add_ionisation_channel(8, 10, n_ion=1e8, ionpot_ev=35.0, xs_vec=good)
        with pytest.raises(ValueError, match="between 1 and 9"):
            sf.add_ionisation_channel(8, 2.5, n_ion=1e8, ionpot_ev=35.0, xs_vec=good)  # ty: ignore[invalid-argument-type]
        with pytest.raises(ValueError, match="bare nucleus"):
            sf.add_ionisation_channel(8, 9, n_ion=1e8, ionpot_ev=35.0, xs_vec=good)
        with pytest.raises(ValueError, match="bare nucleus"):
            sf.add_excitation(8, 9, levelnumberdensity=1e8, xs_vec=good, epsilon_trans_ev=35.0)

        assert not sf.ionpopdict
        assert not sf._ionisation_channels

        # a zero population registers the ion and keeps the checked channel with no matrix entries
        with pytest.raises(ValueError, match="non-negative"):
            sf.add_ionisation_channel(8, 3, n_ion=0.0, ionpot_ev=55.0, xs_vec=-_bethe_xs_grid(sf.engrid_mid, 55.0))
        sf.add_ionisation_channel(8, 3, n_ion=0.0, ionpot_ev=55.0, xs_vec=_bethe_xs_grid(sf.engrid_mid, 55.0))
        assert sf.ionpopdict[(8, 3)] == 0.0
        assert len(sf._ionisation_channels[(8, 3)]) == 1
        assert not sf.sfmatrix.any()

        # an omitted channelkey gets an automatic index, and a duplicate key is rejected
        sf.add_ionisation_channel(8, 2, n_ion=1e8, ionpot_ev=35.0, xs_vec=good)
        assert sf._ionisation_channels[(8, 2)][0].key == 0
        with pytest.raises(ValueError, match="already added"):
            sf.add_ionisation_channel(8, 2, n_ion=1e8, ionpot_ev=35.0, xs_vec=good, channelkey=0)
        # the population must agree with the one an earlier call for this ion gave
        with pytest.raises(ValueError, match="different populations"):
            sf.add_ionisation_channel(8, 2, n_ion=2e8, ionpot_ev=42.0, xs_vec=_bethe_xs_grid(sf.engrid_mid, 42.0))

        sf.solve(deposition_ev_per_s_per_cm3=1e8)
        with pytest.raises(RuntimeError, match="after solving"):
            sf.add_ionisation_channel(8, 2, n_ion=1e8, ionpot_ev=60.0, xs_vec=good)

    # numpy integers are accepted and stored as Python integers
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        good = _bethe_xs_grid(sf.engrid_mid, 35.0)
        Z_numpy, ion_stage_numpy = np.int64(8), np.int64(2)
        sf.add_ionisation_channel(Z_numpy, ion_stage_numpy, n_ion=1e8, ionpot_ev=35.0, xs_vec=good)  # ty: ignore[invalid-argument-type]
        assert all(type(key) is int for key in next(iter(sf.ionpopdict)))


def test_add_ionisation_is_atomic() -> None:
    # a key clash inside add_ionisation must leave the solver exactly as it was, so that the
    # caller can correct the key and try again
    Z, ion_stage = 8, 2
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=150) as sf:
        # "n 2 l 0" is the second built-in shell of O II, so the clash comes after the first
        sf.add_ionisation_channel(
            Z,
            ion_stage,
            n_ion=1e8,
            ionpot_ev=42.6,
            xs_vec=_bethe_xs_grid(sf.engrid_mid, 42.6),
            channelkey="n 2 l 0",
        )
        matrix_before = sf.sfmatrix.copy()

        with pytest.raises(ValueError, match="already added"):
            sf.add_ionisation(Z, ion_stage, n_ion=1e8)

        assert np.array_equal(sf.sfmatrix, matrix_before)
        assert [channel.key for channel in sf._ionisation_channels[(Z, ion_stage)]] == ["n 2 l 0"]

        # the ion is not poisoned: a call with a free key still works
        sf.add_ionisation_channel(Z, ion_stage, n_ion=1e8, ionpot_ev=35.1, xs_vec=_bethe_xs_grid(sf.engrid_mid, 35.1))
        assert len(sf._ionisation_channels[(Z, ion_stage)]) == 2


def test_cross_sections_are_copied_from_the_caller() -> None:
    # the solver keeps a read-only copy of every cross section, so a later write by the caller
    # cannot silently change the physics it already recorded
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf:
        xs_vec = np.where(sf.engrid_mid >= 20.0, 1e-16, 0.0)
        sf.add_excitation(8, 2, levelnumberdensity=1e8, xs_vec=xs_vec, epsilon_trans_ev=20.0)
        xs_vec[:] = 0.0
        assert sf.excitationlists[(8, 2)][0].xs_vec.max() > 0.0
        assert not sf.excitationlists[(8, 2)][0].xs_vec.flags.writeable

        ion_xs = _bethe_xs_grid(sf.engrid_mid, 35.0)
        sf.add_ionisation_channel(8, 2, n_ion=1e8, ionpot_ev=35.0, xs_vec=ion_xs)
        ion_xs[:] = 0.0
        channel = sf._ionisation_channels[(8, 2)][0]
        assert channel.xs_grid.max() > 0.0
        assert not channel.xs_grid.flags.writeable
        # the interpolation the channel uses off the grid kept the copy too
        assert channel.xs(np.array([100.5]))[0] > 0.0

    # the built-in shells of one ion must not share an array
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf_builtin:
        sf_builtin.add_ionisation(8, 2, n_ion=1e8)
        channels = sf_builtin._ionisation_channels[(8, 2)]
        assert len(channels) > 1
        assert not np.shares_memory(channels[0].xs_grid, channels[1].xs_grid)


def test_engrid_is_read_only() -> None:
    # deltaen is derived from engrid once, so a write in place would leave the two inconsistent
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=100) as sf:
        assert not sf.engrid.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            sf.engrid[0] = 2.0
        assert sf.engrid[0] == 1.0


def test_excitation_xs_must_be_finite() -> None:
    # an infinite cross section used to pass the non-negative test and put inf into the matrix
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=100) as sf:
        xs_vec = np.zeros(100)
        xs_vec[50:] = np.inf
        with pytest.raises(ValueError, match="finite"):
            sf.add_excitation(8, 2, levelnumberdensity=1e8, xs_vec=xs_vec, epsilon_trans_ev=20.0)
        assert not sf.sfmatrix.any()

        # a cross section of the wrong shape is named, rather than reaching the matrix band
        with pytest.raises(ValueError, match="one value for each"):
            sf.add_excitation(8, 2, levelnumberdensity=1e8, xs_vec=np.zeros((100, 1)), epsilon_trans_ev=20.0)
        assert not sf.excitationlists.get((8, 2), {})


def test_every_builtin_ion_builds_channels() -> None:
    # the checks in IonisationChannel.from_xs must accept every row of the built-in tables, on both
    # the Arnaud & Rothenflug fit branch and the Lotz branch
    engrid = np.linspace(1.0, 3000.0, num=64, dtype=np.float64)
    for collionfilename in ("collion.txt", "collion-AR1985.txt"):
        dfcollion = pynonthermal.collion.read_colliondata(collionfilename)
        n_lotz = 0
        n_fit = 0
        for Z, ion_stage in dfcollion.select(["Z", "ion_stage"]).unique().iter_rows():
            channels = pynonthermal.collion.get_ion_ionisation_channels(dfcollion, Z, ion_stage, engrid)
            assert channels
            for channel in channels:
                if str(channel.key).startswith("Lotz"):
                    n_lotz += 1
                else:
                    n_fit += 1
        assert n_lotz > 0
        assert n_fit > 0


def test_rejected_solve_keeps_the_last_solution() -> None:
    # every argument of solve() is checked before the solver takes any of them, so a rejected call
    # keeps the deposition rate density, the solution, and the analysis of the last solve
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        sf.add_ionisation(8, 2, n_ion=1e8)
        sf.solve(deposition_ev_per_s_per_cm3=100.0)
        assert sf.deposition_ev_per_s_per_cm3 == 100.0
        yvec = sf.yvec.copy()
        frac_heating = sf.get_frac_heating()

        bad_arguments: list[tuple[dict[str, t.Any], str]] = [
            ({"deposition_ev_per_s_per_cm3": 999.0, "balance_tol": 0.0}, "balance_tol"),
            ({"deposition_ev_per_s_per_cm3": 0.0}, "deposition_ev_per_s_per_cm3 must be"),
            ({}, "needs the deposition rate density"),
        ]
        for arguments, message in bad_arguments:
            with pytest.raises(ValueError, match=message):
                sf.solve(**arguments)
            # the getters still give the last solution, which the caller does not have to repeat
            assert sf.deposition_ev_per_s_per_cm3 == 100.0
            assert np.array_equal(sf.yvec, yvec)
            assert sf.get_frac_heating() == frac_heating
        # the deprecated override_n_e argument is checked there as well
        with (
            pytest.warns(DeprecationWarning, match="override_n_e"),
            pytest.raises(ValueError, match="override_n_e"),
        ):
            sf.solve(deposition_ev_per_s_per_cm3=999.0, override_n_e=-1.0)
        assert sf.deposition_ev_per_s_per_cm3 == 100.0


def test_solution_arrays_are_read_only() -> None:
    # the fractions and the rate coefficients integrate yvec, so an in-place write by the caller
    # would silently change every one of them. The write must raise instead.
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        sf.add_ionisation(8, 2, n_ion=1e8)
        sf.solve(deposition_ev_per_s_per_cm3=1e8)
        frac_heating = sf.get_frac_heating()

        for arr in (sf.yvec, sf.engrid):
            assert not arr.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                arr[0] = 1.0

        # a second solve gives a new array that is read-only as well
        sf.solve(deposition_ev_per_s_per_cm3=2e8)
        assert not sf.yvec.flags.writeable
        assert math.isclose(sf.get_frac_heating(), frac_heating, rel_tol=1e-12)


def test_lotz_constant_is_configurable() -> None:
    # every Lotz channel scales with lotz_a_cm2_ev2, the fit channels do not, and the solver warns
    # once per element about the ions that use the Lotz formula
    factor = 2.0
    with pytest.raises(ValueError, match="lotz_a_cm2_ev2"):
        pynonthermal.SpencerFanoSolver(lotz_a_cm2_ev2=0.0)
    with (
        pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf_default,
        pynonthermal.SpencerFanoSolver(
            emin_ev=1, emax_ev=3000, npts=300, lotz_a_cm2_ev2=factor * pynonthermal.axelrod.LOTZ_A_CM2_EV2
        ) as sf_scaled,
    ):
        # the constant is fixed for the life of the solver
        with pytest.raises(AttributeError):
            sf_scaled.lotz_a_cm2_ev2 = 1.0  # ty: ignore[invalid-assignment]
        for sf in (sf_default, sf_scaled):
            with warnings.catch_warnings():
                # O I has fits for every shell, so no warning
                warnings.simplefilter("error", UserWarning)
                sf.add_ionisation(8, 1, n_ion=1.0)
            with pytest.warns(pynonthermal.LotzApproximationWarning, match=r"Z=56 ion_stage 2: no fitted"):
                sf.add_ionisation(56, 2, n_ion=1.0)
            # one warning for an element with two Lotz stages
            with pytest.warns(pynonthermal.LotzApproximationWarning, match=r"Z=57 ion_stages \[2, 3\]") as record:
                sf.add_element(57, 1e8, recomb_ratecoeffs={3: 1e-12})
            assert len([w for w in record if issubclass(w.category, pynonthermal.LotzApproximationWarning)]) == 1

        # off-grid energies probe the function, and the grid probes the stored array
        arr_en_ev = np.linspace(1.5, 2999.5, 200)
        for (Z, ion_stage), scaled in (((56, 2), True), ((57, 2), True), ((8, 1), False)):
            channels_default = sf_default._ionisation_channels[(Z, ion_stage)]
            channels_scaled = sf_scaled._ionisation_channels[(Z, ion_stage)]
            expected_factor = factor if scaled else 1.0
            for channel_default, channel_scaled in zip(channels_default, channels_scaled, strict=True):
                assert channel_default.lotz is scaled
                assert np.allclose(
                    channel_scaled.xs(arr_en_ev), expected_factor * channel_default.xs(arr_en_ev), rtol=1e-12
                )
                assert np.allclose(channel_scaled.xs_grid, expected_factor * channel_default.xs_grid, rtol=1e-12)


def test_heating_of_an_excitation_below_emin() -> None:
    # The excitation term of N(E) needs y(E + eps), which is zero below emin_ev. For eps < emin_ev
    # the integrand of the heating below E_0 therefore has a step at E_0 - eps. Simpson's rule on
    # 9 nodes gave a heating 8 times too large and a fraction sum of 1.25 for the first plasma. The
    # energy of the excitation terms below E_0 comes from their bands in the matrix, so the sum is one
    # to rounding with a cross section that starts inside the range (the second plasma, where an
    # interpolation of N(E) gave 12.8), and for a transition energy near emax_ev (the third plasma).
    cases = (
        (100.0, 3000, 0.5, 0.5),
        (100.0, 500, 50.0, 130.3),
        (1.0, 200, 2999.5, 2999.5),
    )
    for emin_ev, npts, epsilon_trans_ev, xs_start_ev in cases:
        with pynonthermal.SpencerFanoSolver(emin_ev=emin_ev, emax_ev=3000, npts=npts) as sf:
            sf.add_excitation(
                1,
                1,
                1e9,
                lambda en_ev, xs_start_ev=xs_start_ev: np.where(en_ev >= xs_start_ev, 1e-16, 0.0),
                epsilon_trans_ev=epsilon_trans_ev,
            )
            sf.override_n_e(1e3)
            sf.solve(deposition_ev_per_s_per_cm3=1.0)
            assert math.isclose(sf.get_frac_sum(), 1.0, rel_tol=1e-9)


def test_energy_fractions_of_the_heating_only_case_sum_to_one() -> None:
    # the source normalisation and rhsvec must agree. They disagreed by deltaen per source
    # electron, which made every energy fraction too large by deltaen / E_init_ev.
    for npts in (30, 200, 4096):
        with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=npts) as sf:
            sf.override_n_e(1e6)
            sf.solve(deposition_ev_per_s_per_cm3=1.0)
            assert math.isclose(sf.get_frac_heating(), 1.0, rel_tol=1e-12)


def test_verbose_analysis_does_not_raise_where_the_quiet_analysis_succeeds() -> None:
    # the work function estimate of the verbose output needs the shell tables, which stop at Z=108,
    # and a total population above zero
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200, verbose=True) as sf:
        sf.add_ionisation_channel(110, 2, 1e8, 20.0, lambda en_ev: np.where(en_ev > 20.0, 1e-16, 0.0))
        sf.override_n_e(1e6)
        sf.solve(deposition_ev_per_s_per_cm3=1e8)
        sf.get_frac_sum()
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200, verbose=True) as sf:
        # add_ion_excitation() registers the zero population, so the ionisation channels go into the
        # solver and the verbose analysis reaches the work function estimate
        sf.set_temperature(6000)
        sf.add_ion_excitation(8, 2, n_ion=0.0)
        sf.add_ionisation(8, 2, None)
        sf.override_n_e(1e6)
        sf.solve(deposition_ev_per_s_per_cm3=1e8)
        sf.get_frac_sum()


def test_rejected_calls_leave_the_solver_unchanged() -> None:
    with pynonthermal.SpencerFanoSolver(npts=200) as sf:
        # an unhashable key raises before the first write
        with pytest.raises(TypeError):
            sf.add_excitation(26, 2, 1e5, np.zeros(200), 5.0, transitionkey=[0, 1])
        assert sf.excitationlists == {}

        # the deprecated method restores the temperature and the options of the solver
        with (
            pytest.warns(DeprecationWarning, match="add_ion_ltepopexcitation"),
            pytest.raises(ValueError, match="n_ion must be non-negative"),
        ):
            sf.add_ion_ltepopexcitation(26, 2, -1.0, maxnlevelslower=10)
        assert sf.temperature is None
        assert sf._maxnlevelslower == 5

        # a bad rate coefficient raises before the level data is read or the temperature is needed
        with pytest.raises(ValueError, match="recombination rate coefficient of Z=8 ion_stage 2"):
            sf.add_element(8, 1e8, recomb_ratecoeffs={2: math.nan}, excitation=True)
        assert sf.adata_polars is None

    with pynonthermal.SpencerFanoSolver(emin_ev=12, npts=200) as sf, warnings.catch_warnings():
        # the rate coefficient warning comes only after every check of the call
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="below emin_ev"):
            sf.add_element(26, 1e8, recomb_ratecoeffs={2: 1e-19, 3: 1e-12})


def test_ionisation_channel_checks_the_ion() -> None:
    # a charge in place of the ion stage gave a channel with the J of the wrong stage
    engrid = np.linspace(1.0, 3000.0, 200)
    with pytest.raises(ValueError, match="ion_stage is one more than the charge"):
        pynonthermal.IonisationChannel.from_xs(
            engrid, 2, 0, 24.59, lambda en_ev: np.where(en_ev > 24.59, 1e-16, 0.0), "k"
        )


def test_shared_tables_are_not_writable() -> None:
    # every solver and every channel share the cached tables, so a caller gets a copy of each
    nist = pynonthermal.collion.get_nist_ionisation_energies_ev()
    ionpot_ev = nist[(8, 1)]
    nist[(8, 1)] = 1.0
    assert pynonthermal.collion.get_nist_ionisation_energies_ev()[(8, 1)] == ionpot_ev
    assert pickle.loads(pickle.dumps(nist)) == nist  # noqa: S301
    with pynonthermal.SpencerFanoSolver(npts=100) as sf:
        sf.dfcollion[0, "A"] = 1e3
    with pynonthermal.SpencerFanoSolver(npts=100) as sf_later:
        assert sf_later.dfcollion[0, "A"] != 1e3
    assert pynonthermal.collion.read_colliondata()[0, "A"] != 1e3
    # the cross section of a built-in channel cannot change after the channel holds its grid values
    channel = pynonthermal.collion.get_ion_ionisation_channels(
        pynonthermal.collion.read_colliondata(), 8, 2, np.linspace(1.0, 3000.0, 100)
    )[0]
    with pytest.raises(TypeError):
        channel.xs.shell["A"] = 0.0  # ty: ignore[missing-attribute]


def test_solver_with_channels_can_be_pickled() -> None:
    # multiprocessing and joblib pickle a solver, so the cross sections of the channels must not be
    # closures, and the state must leave out the internal level database, whose polars Object columns
    # pickle cannot serialise
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        sf.set_temperature(6000)
        sf.add_ionisation(8, 2, 1e8)
        sf.add_ion_excitation(8, 2)
        sf.add_ionisation_channel(8, 2, None, 60.0, np.where(sf.engrid_mid > 60.0, 1e-18, 0.0))
        sf.solve(deposition_ev_per_s_per_cm3=1e3)
        sf_copy = pickle.loads(pickle.dumps(sf))  # noqa: S301
        assert sf_copy.get_frac_sum() == sf.get_frac_sum()
        assert sf.adata_polars is not None
        assert sf_copy.adata_polars is None


def test_population_of_a_numpy_float32_is_a_python_float() -> None:
    # a float32 population made the loss function float32, which underflowed to zero
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        sf.add_ionisation(26, 2, np.float32(1e-10))  # ty: ignore[invalid-argument-type]
        sf.solve(deposition_ev_per_s_per_cm3=1.0)
        assert type(sf.get_n_e()) is float


def test_auger_energy_warning_names_the_line_of_the_caller() -> None:
    # the warning must name this file on every path, whatever the depth of the call in the package
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        xs_vec = np.where(sf.engrid_mid > 40.0, 1e-17, 0.0)
        for xs in (xs_vec, lambda en_ev: np.where(en_ev > 40.0, 1e-17, 0.0)):
            with pytest.warns(UserWarning, match="less than the energy that the ion must keep") as record:
                sf.add_ionisation_channel(8, 1, 1e8, 40.0, xs, n_ejected=2)
            assert {warning.filename for warning in record} == {__file__}


def test_auger_energy_warning_comes_after_every_check() -> None:
    # a call that a check rejects gives only its error, also with warnings as errors
    xs = lambda en_ev: np.where(en_ev > 40.0, 1e-17, 0.0)  # noqa: E731
    with pynonthermal.SpencerFanoSolver(emin_ev=45, emax_ev=3000, npts=200) as sf, warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="below emin_ev"):
            sf.add_ionisation_channel(8, 1, 1e8, 40.0, xs, n_ejected=2)
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        with pytest.warns(UserWarning, match="less than the energy that the ion must keep"):
            sf.add_ionisation_channel(8, 1, 1e8, 40.0, xs, "double", n_ejected=2)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="already added"):
                sf.add_ionisation_channel(8, 1, 1e8, 40.0, xs, "double", n_ejected=2)
            with pytest.raises(ValueError, match="must be zero at and below"):
                sf.add_ionisation_channel(8, 1, 1e8, 40.0, np.full(200, 1e-17), "below", n_ejected=2)


def test_negative_level_weight_rejects_the_call() -> None:
    # a negative statistical weight gives a negative population fraction, which the check of each level
    # weight rejects before the first write, on the fixed and on the balanced path
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        adata = sf._get_adata_polars().filter((pl.col("Z") == 8) & (pl.col("ion_stage") == 2))
        levels = adata["levels"].item()
        levels = levels.with_columns(g=pl.when(pl.col("levelindex") == 1).then(-pl.col("g")).otherwise(pl.col("g")))
        adata = adata.with_columns(pl.Series("levels", [levels], dtype=pl.Object))
        sf.set_temperature(6000)
        sf.set_atomic_data(adata_polars=adata)
        with pytest.raises(ValueError, match="give no valid LTE populations"):
            sf.add_element(8, ion_densities={2: 1e8}, excitation=True)
        assert sf.ionpopdict == {}
        assert sf.excitationlists == {}
        with pytest.raises(ValueError, match="give no valid LTE populations"):
            sf.add_element(8, 1e8, recomb_ratecoeffs={3: 3e-12}, excitation=True)
        assert sf.ionpopdict == {}
        assert sf.excitationlists == {}


def test_heating_at_a_low_free_electron_fraction() -> None:
    # N(E) took y between the grid points from a linear interpolation, also from the grid point below
    # the threshold, where y holds only the losses to the free electrons. At a low free electron
    # fraction that y is much larger, and the heating was 2.6 times too large at x_e = 1e-6.
    for x_e in (1e-3, 1e-6):
        with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=1000) as sf:
            sf.add_element(26, ion_densities={1: 1e8 * (1 - x_e), 2: 1e8 * x_e})
            sf.solve(deposition_ev_per_s_per_cm3=1e3)
            assert math.isclose(sf.get_frac_sum(), 1.0, abs_tol=0.03)


def test_numpy_scalar_inputs_give_python_floats() -> None:
    # a numpy float16 overflowed in the products of the analysis and of the balance
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf:
        sf.add_element(8, ion_densities={1: 0.99e8, 2: 0.01e8})
        sf.add_excitation(8, 1, 1e5, np.full(300, 1e-17), np.float16(5.0))  # ty: ignore[invalid-argument-type]
        sf.solve(deposition_ev_per_s_per_cm3=np.float16(1000.0))  # ty: ignore[invalid-argument-type]
        assert type(sf.deposition_ev_per_s_per_cm3) is float
        assert math.isfinite(sf.get_frac_sum())
        assert 10.0 < sf.get_eff_ionpot(8, 1) < 1e3
        # every scalar result is a Python float. The grid spacing was a numpy float64, and it made some of them one.
        assert type(sf.deltaen) is float
        assert type(sf.E_init_ev) is float
        for name, value in {
            "get_frac_heating": sf.get_frac_heating(),
            "get_frac_ionisation_tot": sf.get_frac_ionisation_tot(),
            "get_frac_excitation_tot": sf.get_frac_excitation_tot(),
            "get_frac_sum": sf.get_frac_sum(),
            "get_frac_ionisation_ion": sf.get_frac_ionisation_ion(8, 1),
            "get_frac_excitation_ion": sf.get_frac_excitation_ion(8, 1),
            "get_eff_ionpot": sf.get_eff_ionpot(8, 1),
            "get_ionisation_ratecoeff": sf.get_ionisation_ratecoeff(8, 1),
            "get_ionisation_ratecoeff with n_ejected": sf.get_ionisation_ratecoeff(8, 1, n_ejected=1),
            "get_excitation_ratecoeff": sf.get_excitation_ratecoeff(8, 1, 0),
            "get_n_e": sf.get_n_e(),
            "get_n_e_nt": sf.get_n_e_nt(),
            "calculate_N_e": sf.calculate_N_e(0.5),
        }.items():
            assert type(value) is float, name
    assert math.isclose(
        pynonthermal.ionbalance.get_saha_factor(np.float16(1e4), 13.6, 2.0, 1.0),  # ty: ignore[invalid-argument-type]
        pynonthermal.ionbalance.get_saha_factor(1e4, 13.6, 2.0, 1.0),
        rel_tol=1e-12,
    )


def test_float16_energies_give_a_uniform_grid_and_an_exact_band() -> None:
    # np.linspace calculates in the type of its limits, so float16 limits gave a grid with repeated points
    sf = pynonthermal.SpencerFanoSolver(emin_ev=np.float16(0.1))  # ty: ignore[invalid-argument-type]
    assert len(np.unique(sf.engrid)) == len(sf.engrid)
    assert np.ptp(np.diff(sf.engrid)) < 1e-9

    # the matrix band of a float16 transition energy had float16 weights, and the energy sum was 3e-4 wrong
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf:
        epsilon_trans_ev = np.float16(79.9)
        xs_vec = np.where(sf.engrid_mid > float(epsilon_trans_ev), 1e-16, 0.0)
        sf.add_excitation(8, 1, 1e8, xs_vec, epsilon_trans_ev)  # ty: ignore[invalid-argument-type]
        sf.override_n_e(1e4)
        sf.solve(1e5)
        assert math.isclose(sf.get_frac_sum(), 1.0, abs_tol=1e-12)


def test_getters_and_flags_check_their_arguments() -> None:
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        sf.add_element(8, ion_densities={1: 0.99e8, 2: 0.01e8})
        sf.solve(deposition_ev_per_s_per_cm3=1e3)
        # the charge in place of the ion stage, and a string, give the ion check and not 0.0
        with pytest.raises(ValueError, match="ion_stage is one more than the charge"):
            sf.calculate_nt_frac_excitation_ion(8, 0)
        assert sf.calculate_nt_frac_excitation_ion(8, 1) == 0.0
        with pytest.raises(ValueError, match="Z must be an integer"):
            sf.get_frac_ionisation_ion("8", 2)  # ty: ignore[invalid-argument-type]
        with pytest.raises(ValueError, match="Z must be an integer"):
            sf.get_ion_fractions("8")  # ty: ignore[invalid-argument-type]
        with pytest.raises(ValueError, match="energy_ev must be at least zero"):
            sf.calculate_N_e(-1.0)
        assert isinstance(sf.get_d_etaheating_by_d_en_vec(), np.ndarray)
    # a string is true in a condition, so a flag must be a bool
    with pytest.raises(TypeError, match="heating_only_approximation must be True or False"):
        pynonthermal.SpencerFanoSolver(npts=100, heating_only_approximation="False")  # ty: ignore[invalid-argument-type]
    with pynonthermal.SpencerFanoSolver(npts=100) as sf:
        with pytest.raises(TypeError, match="builtin_channels must be True or False"):
            sf.add_element(8, ion_densities={2: 1e8}, builtin_channels="False")  # ty: ignore[invalid-argument-type]
        # a nan cutoff disabled the cutoff, and a cutoff below one dropped every transition
        for cutoff in (math.nan, 0, "5"):
            with pytest.raises(ValueError, match="maxnlevelslower must be None or an integer"):
                sf.set_atomic_data(maxnlevelslower=cutoff)  # ty: ignore[invalid-argument-type]
        assert sf._maxnlevelslower == 5
        with pytest.raises(TypeError, match="use_collstrengths must be True or False"):
            sf.set_atomic_data(use_collstrengths="False")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="ion_stage is one more than the charge"):
        pynonthermal.axelrod.get_shell_occupancies(26, 0)


def test_plot_spec_channels_on_an_unsolved_solver_opens_no_figure() -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    figures_before = len(plt.get_fignums())
    with pynonthermal.SpencerFanoSolver(npts=100) as sf, pytest.raises(RuntimeError, match="must be solved first"):
        sf.plot_spec_channels()
    assert len(plt.get_fignums()) == figures_before


def test_plot_spec_channels_draws_into_the_axes_of_the_caller(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200)
    sf.add_element(8, ion_densities={2: 1e8})
    sf.solve(1e8)
    fig, axes = plt.subplots(nrows=2, sharex=True)
    figures_before = plt.get_fignums()
    # the caller owns the figure, so the method neither makes a figure nor saves one
    sf.plot_spec_channels(axes=axes, xscalelog=True, outputfilename=tmp_path / "unused.pdf")
    assert plt.get_fignums() == figures_before
    assert not (tmp_path / "unused.pdf").exists()
    assert len(axes[0].lines) == 1
    assert {line.get_label() for line in axes[1].lines} == {"Ionisation", "Heating"}
    assert axes[1].get_xscale() == "log"
    plt.close(fig)


def test_ionbalance_checks_and_edge_cases() -> None:
    ionbalance = pynonthermal.ionbalance
    # an iterator gave one stage, because the check used it up
    assert ionbalance.get_ion_fractions(iter([1e3, 1e2]), 1e2) == ionbalance.get_ion_fractions([1e3, 1e2], 1e2)  # ty: ignore[invalid-argument-type]
    # a first lower bracket that underflows gives the message of the module
    with pytest.raises(ValueError, match="below the range of a double precision number"):
        ionbalance.solve_charge_neutral_n_e_ratios(0.0, [(1e-270, 1, [1e-3])])
    with pytest.raises(ValueError, match="the lowest ion stage must be an integer"):
        ionbalance.solve_charge_neutral_n_e_ratios(0.0, [(1e6, 1.5, [1e5])])  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="ratio coefficients must be non-negative"):
        ionbalance.solve_charge_neutral_n_e_ratios(0.0, [(1e6, 1, [-1.0])])


def test_balance_with_a_low_free_electron_density_converges() -> None:
    # with a free electron density far below the ion charges, the ions take most of the energy, and the
    # mixing alone needed more than BALANCE_MAXITER iterations. Anderson acceleration fixes that.
    with (
        pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf,
        warnings.catch_warnings(),
    ):
        # the top stage of this short chain takes a part of the ionisation, which is not the subject here
        warnings.simplefilter("ignore", UserWarning)
        sf.add_element(8, 1e8, recomb_ratecoeffs={2: 1e-11, 3: 3e-11, 4: 1e-10})
        sf.override_n_e(1.0)
        sf.solve(deposition_ev_per_s_per_cm3=0.1)
        assert sf.balance_iterations < spencerfano.BALANCE_MAXITER


def test_balance_without_any_ionisation_names_the_cause() -> None:
    # every threshold of O I is above emax_ev, so no ion leaves the neutral stage
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=12, npts=200) as sf:
        sf.add_element(8, 1e8, recomb_ratecoeffs={2: 1e-12, 3: 3e-12})
        with pytest.raises(ValueError, match="zero at every energy of the grid"):
            sf.solve(deposition_ev_per_s_per_cm3=1e8)


def test_charge_neutral_root_find_checks_the_charge_density() -> None:
    # a charge density outside the bounds, or nan, gave a wrong root with no error
    for charge_density, bounds in ((lambda _n_e: 5.0, (0.0, 1.0)), (lambda _n_e: math.nan, (0.0, 10.0))):
        with pytest.raises(ValueError, match="which is not between charge_density_min"):
            pynonthermal.ionbalance.solve_charge_neutral_n_e(0.0, charge_density, *bounds)


def test_pickle_protocol_4_keeps_the_arrays_read_only() -> None:
    # numpy keeps the read-only flag only with protocol 5, and multiprocessing can use protocol 4
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        sf.add_element(8, ion_densities={1: 1e5, 2: 1e4})
        sf.add_ionisation_channel(8, 2, None, 60.0, np.where(sf.engrid_mid > 60.0, 1e-18, 0.0))
        sf.solve(deposition_ev_per_s_per_cm3=1e3)
        sf_copy = pickle.loads(pickle.dumps(sf, protocol=4))  # noqa: S301
        for array in (
            sf_copy.engrid,
            sf_copy.engrid_mid,
            sf_copy.yvec,
            sf_copy._ionisation_channels[(8, 2)][0].xs_grid,
        ):
            with pytest.raises(ValueError, match="read-only"):
                array[0] = 0.0

        # a state from a release before engrid_mid gets the attribute
        state = sf.__getstate__()
        del state["engrid_mid"]
        sf_old = pynonthermal.SpencerFanoSolver.__new__(pynonthermal.SpencerFanoSolver)
        sf_old.__setstate__(state)
        assert np.array_equal(sf_old.engrid_mid, sf.engrid_mid)
        assert not sf_old.engrid_mid.flags.writeable


def test_pickle_protocol_4_keeps_the_cached_fills_read_only() -> None:
    # the cached fill of a balanced element is shared by each later call for the channel. A copy of the fill
    # arrays got the read-only flag, and the arrays of the fill stayed writable.
    sf = _balanced_oxygen()
    with pytest.warns(UserWarning, match="top stage"):
        sf.solve(deposition_ev_per_s_per_cm3=1e3)
    assert sf._channel_fills
    sf_copy = pickle.loads(pickle.dumps(sf, protocol=4))  # noqa: S301
    assert sf_copy._channel_fills
    for fill in sf_copy._channel_fills.values():
        for array in (fill.unit_prefactors, fill.atan_mean_offsets, fill.partial_rows, fill.partial_unit_terms):
            assert not array.flags.writeable


def test_default_key_does_not_collide_with_a_named_key() -> None:
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        sf.add_excitation(8, 2, 1e3, np.full(200, 1e-18), 10.0, transitionkey=1)
        sf.add_excitation(8, 2, 1e3, np.full(200, 1e-18), 12.0)
        assert set(sf.excitationlists[(8, 2)]) == {1, 2}
        xs = np.where(sf.engrid_mid > 60.0, 1e-18, 0.0)
        sf.add_ionisation_channel(8, 2, 1e4, 60.0, xs, channelkey=1)
        sf.add_ionisation_channel(8, 2, 1e4, 70.0, np.where(sf.engrid_mid > 70.0, 1e-18, 0.0))
        assert [channel.key for channel in sf._ionisation_channels[(8, 2)]] == [1, 2]


def test_population_arguments_check_their_types() -> None:
    with pynonthermal.SpencerFanoSolver(npts=100) as sf:
        with pytest.raises(ValueError, match="n_ion must be non-negative and finite but is True"):
            sf.add_ionisation(8, 1, True)
        with pytest.raises(TypeError, match="ion_fractions must be a mapping"):
            sf.add_element(8, 1e8, ion_fractions=[0.3, 0.7])  # ty: ignore[invalid-argument-type]


def test_iron_notebook_deposition_is_independent_of_the_selected_epoch() -> None:
    # The deposition function scaled the constants of 250 days with the selected epoch T_DAYS.
    notebook = json.loads((Path(__file__).parents[2] / "fe_ionbalance_sn1a.ipynb").read_text(encoding="utf-8"))
    sources = ["".join(cell["source"]) for cell in notebook["cells"]]
    namespace: dict[str, t.Any] = {"math": math}
    # the source is the notebook of this repository
    exec(next(source for source in sources if "def get_deposition_rate_density" in source), namespace)  # noqa: S102
    deposition = namespace["get_deposition_rate_density"](500.0)
    namespace["T_DAYS"] = 500.0
    assert namespace["get_deposition_rate_density"](500.0) == deposition


def test_verbose_output_gives_the_version(capsys: pytest.CaptureFixture[str]) -> None:
    # a log of a result must record the version of the package that made it
    assert pynonthermal.__version__
    pynonthermal.SpencerFanoSolver(npts=100, verbose=True)
    assert f"(pynonthermal {pynonthermal.__version__})" in capsys.readouterr().out


def test_plot_prints_the_path_only_with_verbose(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for verbose in (False, True):
        sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200, verbose=verbose)
        sf.add_element(8, ion_densities={2: 1e8})
        sf.solve(1e8)
        capsys.readouterr()
        sf.plot_yspectrum(outputfilename=tmp_path / "yspectrum.pdf")
        assert ("Saving" in capsys.readouterr().out) == verbose


def test_ion_results_agree_with_the_getters() -> None:
    sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300)
    sf.set_temperature(6000)
    sf.add_element(26, ion_densities={2: 1e5, 3: 1e5})
    sf.add_element(8, ion_densities={1: 1e8, 2: 1e6}, excitation=True)
    with pytest.raises(RuntimeError, match=r"Call solve\(\)"):
        sf.get_ion_results()
    sf.solve(1e8)

    table = sf.get_ion_results()
    assert table.columns == [
        "Z",
        "ion_stage",
        "n_ion",
        "frac_ionisation",
        "frac_excitation",
        "ionisation_ratecoeff",
        "eff_ionpot_ev",
    ]
    # sorted by Z and ion_stage, not in the order of the add_element() calls
    assert table.select("Z", "ion_stage").rows() == [(8, 1), (8, 2), (26, 2), (26, 3)]
    for row in table.iter_rows(named=True):
        Z, ion_stage = row["Z"], row["ion_stage"]
        assert row["n_ion"] == sf.ionpopdict[(Z, ion_stage)]
        assert row["frac_ionisation"] == sf.get_frac_ionisation_ion(Z, ion_stage)
        assert row["frac_excitation"] == sf.get_frac_excitation_ion(Z, ion_stage)
        assert row["ionisation_ratecoeff"] == sf.get_ionisation_ratecoeff(Z, ion_stage)
        assert row["eff_ionpot_ev"] == sf.get_eff_ionpot(Z, ion_stage)
    assert table["frac_ionisation"].sum() == pytest.approx(sf.get_frac_ionisation_tot(), rel=1e-12)
    assert table["frac_excitation"].sum() == pytest.approx(sf.get_frac_excitation_tot(), rel=1e-12)

    # the table must not give the rows of a solution that a later call discarded
    sf.override_n_e(1e7)
    with pytest.raises(RuntimeError, match="discarded the last solution"):
        sf.get_ion_results()


def test_ion_results_give_null_for_an_unknown_population() -> None:
    # an excitation with only a level density gives the solver no population for its ion. The table
    # must not report zero for that ion, because the ion has at least the level density.
    sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200)
    sf.add_element(8, ion_densities={2: 1e8})
    sf.add_excitation(
        Z=26,
        ion_stage=2,
        levelnumberdensity=1e8,
        xs_vec=np.where(sf.engrid_mid >= 5.0, 1e-16 * 5.0 / sf.engrid_mid, 0.0),
        epsilon_trans_ev=5.0,
        transitionkey=(0, 1),
    )
    sf.solve(1e8)
    table = sf.get_ion_results()
    fe_ii = table.filter((pl.col("Z") == 26) & (pl.col("ion_stage") == 2))
    assert fe_ii["n_ion"].to_list() == [None]
    assert fe_ii["frac_excitation"].item() > 0.0
    assert table.filter(pl.col("Z") == 8)["n_ion"].to_list() == [1e8]


def test_repr_gives_the_state_without_a_calculation() -> None:
    sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200)
    assert repr(sf) == "SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200, ions=[], not solved)"
    sf.add_element(8, ion_densities={2: 1e8})
    assert repr(sf).endswith("ions=[(8, 2)], not solved)")
    sf.solve(1e8)
    # the solve() above did not analyse the solution, and repr() must not analyse it either
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert repr(sf).endswith("ions=[(8, 2)], solved)")
    assert not sf._analysed
    assert repr(pickle.loads(pickle.dumps(sf))) == repr(sf)  # noqa: S301
    sf.override_n_e(1e7)
    assert repr(sf).endswith(", solution discarded)")
    # a solver that __init__() did not complete still has a repr
    assert repr(object.__new__(pynonthermal.SpencerFanoSolver)).startswith(
        "<pynonthermal.spencerfano.SpencerFanoSolver"
    )


@contextlib.contextmanager
def _fail_from_call(target: object, name: str, failing_call: int, exception: type[BaseException]) -> Generator[None]:
    # patch the function target.name, so that each call from the call with the number failing_call raises the
    # exception
    function = getattr(target, name)
    calls = 0

    def wrapper(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls >= failing_call:
            raise exception
        return function(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(target, name, wrapper)
        yield


def _balanced_oxygen() -> pynonthermal.SpencerFanoSolver:
    sf = pynonthermal.SpencerFanoSolver(npts=300)
    sf.add_element(8, n_elem=1e6, recomb_ratecoeffs={2: 3e-13, 3: 3e-12, 4: 1e-11})
    return sf


@pytest.mark.parametrize("exception", [KeyboardInterrupt, MemoryError])
@pytest.mark.parametrize("failing_call", [1, 2, 4])
def test_solve_after_an_interrupted_population_update(exception: type[BaseException], failing_call: int) -> None:
    # an interrupt or an error part-way through a matrix update of the balance left the matrix and ionpopdict
    # out of step. The next update adds only the change of each population, so a later solve() gave wrong
    # populations with no warning.
    reference = _balanced_oxygen()
    reference.solve(1e2)
    sf = _balanced_oxygen()
    with _fail_from_call(spencerfano, "_add_scaled_upper_triangle", failing_call, exception), pytest.raises(exception):
        sf.solve(1e2)
    assert sf._sfmatrix_stale
    sf.solve(1e2)
    assert not sf._sfmatrix_stale
    assert sf.get_ion_fractions(8) == pytest.approx(reference.get_ion_fractions(8), rel=1e-4, abs=0.0)


def _balanced_oxygen_with_excitation() -> pynonthermal.SpencerFanoSolver:
    # most transitions have one energy, so that the bands of a fixed ion and of two balanced stages have one
    # index. Then a fill of the matrix adds the bands together. The last transition has a band of its own.
    sf = _balanced_oxygen()
    sf.add_ionisation(1, 1, 1e5)
    sf.add_excitation(1, 1, None, np.where(sf.engrid_mid > 120.0, 1e-16, 0.0), 120.0, levelpopfrac=1.0)
    for ion_stage, transitionkey, epsilon_trans_ev in (
        (1, "a", 120.0),
        (1, "b", 120.0),
        (2, "a", 120.0),
        (2, "b", 250.0),
    ):
        xs_vec = np.where(sf.engrid_mid > epsilon_trans_ev, 1e-16, 0.0)
        sf.add_excitation(8, ion_stage, None, xs_vec, epsilon_trans_ev, transitionkey, levelpopfrac=0.5)
    return sf


@pytest.mark.parametrize("next_call", ["solve", "add"])
def test_matrix_fill_after_an_interrupt_keeps_the_excitation_bands(next_call: str) -> None:
    # the fill after an interrupted population update adds the bands of the fixed transitions and the unit
    # bands of the balanced stages. The next solve() or the next add call starts the fill.
    reference = _balanced_oxygen_with_excitation()
    sf = _balanced_oxygen_with_excitation()
    with (
        _fail_from_call(spencerfano, "_add_scaled_upper_triangle", 2, KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        sf.solve(1e2)
    assert sf._sfmatrix_stale
    if next_call == "add":
        sf.add_ionisation(2, 1, 1e4)
        reference.add_ionisation(2, 1, 1e4)
        assert not sf._sfmatrix_stale
        assert np.allclose(sf.sfmatrix, reference.sfmatrix, rtol=1e-12, atol=1e-12 * np.abs(reference.sfmatrix).max())
    sf.solve(1e2)
    reference.solve(1e2)
    assert math.isclose(sf.get_frac_excitation_tot(), reference.get_frac_excitation_tot(), rel_tol=1e-4)
    assert sf.get_ion_fractions(8) == pytest.approx(reference.get_ion_fractions(8), rel=1e-4, abs=0.0)


def test_failed_outer_transaction_removes_the_inner_add_call() -> None:
    # an add call inside another add call. The marks are sizes, so the failure of the outer call also removes
    # the records of the inner call.
    sf = _balanced_oxygen()

    @pynonthermal.SpencerFanoSolver._add_transaction
    def add_in_an_outer_transaction(solver: pynonthermal.SpencerFanoSolver) -> None:
        solver.add_ionisation(1, 1, 1e5)
        assert (1, 1) in solver.ionpopdict
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        add_in_an_outer_transaction(sf)
    assert (1, 1) not in sf.ionpopdict
    assert (1, 1) not in sf._ionisation_channels
    assert sf._sfmatrix_stale


def test_failed_solve_after_a_solution_keeps_it_with_a_stale_matrix() -> None:
    # a MemoryError in a matrix update of a second solve(). The restore applied the change of each population
    # to a matrix that the update had changed only in part. Now the restore does not write to the stale matrix,
    # so a second MemoryError cannot stop it. An update makes 4 calls, so the failure is in the second update.
    # Then the restore has a change of the populations to write.
    sf = _balanced_oxygen()
    sf.solve(1e1)
    frac_heating = sf.get_frac_heating()
    populations = dict(sf.ionpopdict)
    with _fail_from_call(spencerfano, "_add_scaled_upper_triangle", 7, MemoryError), pytest.raises(MemoryError):
        sf.solve(1e2)
    assert sf.get_frac_heating() == frac_heating
    assert sf.ionpopdict == populations
    sf.solve(1e2)
    reference = _balanced_oxygen()
    reference.solve(1e2)
    assert sf.get_ion_fractions(8) == pytest.approx(reference.get_ion_fractions(8), rel=1e-4, abs=0.0)


def test_interrupted_solve_keeps_the_last_solution() -> None:
    # solve() caught only an Exception, so a KeyboardInterrupt (an interrupt of the kernel in Jupyter) lost
    # the last solution and opened the add methods for the solved plasma
    sf = _balanced_oxygen()
    sf.solve(1e1)
    frac_heating = sf.get_frac_heating()
    populations = dict(sf.ionpopdict)
    with (
        _fail_from_call(pynonthermal.SpencerFanoSolver, "_solve_matrix", 2, KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        sf.solve(1e2)
    assert sf.get_frac_heating() == frac_heating
    assert sf.ionpopdict == populations
    with pytest.raises(RuntimeError, match="after solving"):
        sf.add_ionisation(1, 1, 1e5)


def test_interrupted_verbose_output_keeps_the_last_solution() -> None:
    # the verbose output of solve() came after the guard. An interrupt of the output, for example a
    # KeyboardInterrupt or a BrokenPipeError, left the solver with no solution and no restore.
    sf = _balanced_oxygen()
    sf.solve(1e1)
    frac_heating = sf.get_frac_heating()
    populations = dict(sf.ionpopdict)

    def print_that_fails(*args: object, **_kwargs: object) -> None:
        # only the summary at the end of solve() fails, and not an earlier output of the balance
        if args and str(args[0]).startswith(" n_ion_tot:"):
            raise BrokenPipeError

    sf.verbose = True
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(spencerfano, "print", print_that_fails, raising=False)
        with pytest.raises(BrokenPipeError):
            sf.solve(1e2)
    assert sf.get_frac_heating() == frac_heating
    assert sf.ionpopdict == populations
    with pytest.raises(RuntimeError, match="after solving"):
        sf.add_ionisation(1, 1, 1e5)

    sf.verbose = False
    sf.solve(1e2)
    reference = _balanced_oxygen()
    reference.solve(1e2)
    assert sf.get_ion_fractions(8) == pytest.approx(reference.get_ion_fractions(8), rel=1e-4, abs=0.0)


def test_copy_with_a_read_only_matrix_can_solve_and_add() -> None:
    # joblib gives a large array of a pickled object to a worker as a read-only memory map. The solver writes
    # to its matrix in place, so such a copy could not solve a balanced element or add an ion.
    sf = _balanced_oxygen()
    read_only = pickle.loads(pickle.dumps(sf))  # noqa: S301
    read_only.sfmatrix.flags.writeable = False
    # protocol 5 keeps the read-only flag of an array
    copy = pickle.loads(pickle.dumps(read_only, protocol=5))  # noqa: S301
    assert copy.sfmatrix.flags.writeable
    copy.add_ionisation(1, 1, 1e5)
    copy.solve(1e2)
    reference = _balanced_oxygen()
    reference.add_ionisation(1, 1, 1e5)
    reference.solve(1e2)
    assert copy.get_ion_fractions(8) == pytest.approx(reference.get_ion_fractions(8), rel=1e-4, abs=0.0)


def test_failed_matrix_fill_leaves_the_add_call_out() -> None:
    # an add call registered the population and stored the channel before the matrix fill. A fill that failed
    # left both in the solver with no matrix entries, and a later solve() was silently wrong.
    with pynonthermal.SpencerFanoSolver(npts=300) as sf:
        sf.add_element(8, ion_densities={1: 1e8, 2: 1e7})
        sf.sfmatrix.flags.writeable = False
        with pytest.raises(ValueError, match="read-only"):
            sf.add_ionisation(26, 2, 1e4)
        assert (26, 2) not in sf.ionpopdict
        assert (26, 2) not in sf._ionisation_channels
        sf.solve(1e8)
        with pynonthermal.SpencerFanoSolver(npts=300) as reference:
            reference.add_element(8, ion_densities={1: 1e8, 2: 1e7})
            reference.solve(1e8)
            assert np.allclose(sf.yvec, reference.yvec, rtol=1e-12, atol=0.0)

    # a call that a check rejects writes nothing, so the matrix is not filled again
    with pynonthermal.SpencerFanoSolver(emin_ev=12.0, emax_ev=3000, npts=300) as sf:
        with pytest.raises(ValueError, match="below emin_ev"):
            sf.add_element(26, 1e8, ion_fractions={2: 0.5, 1: 0.5})
        assert not sf._sfmatrix_stale


def test_override_with_the_density_in_use_keeps_the_solution() -> None:
    # the check compared the value only with the earlier override. An override with the density from the ion
    # charges discarded a valid solution and said that the density changed.
    with pynonthermal.SpencerFanoSolver(npts=300) as sf:
        sf.add_element(8, ion_densities={1: 1e8, 2: 1e8})
        sf.solve(1e8)
        frac_heating = sf.get_frac_heating()
        sf.override_n_e(sf.get_n_e())
        assert sf.get_frac_heating() == frac_heating
        sf.override_n_e(None)
        assert sf.get_frac_heating() == frac_heating
        sf.override_n_e(2e8)
        with pytest.raises(RuntimeError, match="which changed the free electron density, discarded"):
            sf.get_frac_heating()

    # with a balanced element, a given density replaces charge neutrality, so the rule of the balance changes
    sf = _balanced_oxygen()
    sf.solve(1e2)
    sf.override_n_e(sf.get_n_e())
    with pytest.raises(RuntimeError, match="ionisation balance to a given density"):
        sf.get_frac_heating()
