"""Tests of the declarative interface: Plasma, Element, and solve_spencerfano()."""

import math
import typing as t

import numpy as np
import numpy.typing as npt
import pytest

import pynonthermal

HELIUM_ALPHAS = {2: 4e-13, 3: 2e-12}


def test_solve_spencerfano_matches_the_solver() -> None:
    # the declarative call gives the same solution as the low-level solver that it drives
    plasma = pynonthermal.Plasma(
        temperature=6000,
        use_collstrengths=False,
        elements=[
            pynonthermal.Element(8, 1e9, pynonthermal.Fixed({1: 0.9, 2: 0.1}), excitation=True),
            pynonthermal.Element(2, 1e8, pynonthermal.IonBalance(HELIUM_ALPHAS)),
        ],
    )
    result = pynonthermal.solve_spencerfano(plasma, 1e8, emin_ev=1, emax_ev=3000, npts=300)

    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf:
        sf.set_temperature(6000)
        sf.set_atomic_data(use_collstrengths=False)
        sf.add_element(8, 1e9, pynonthermal.Fixed({1: 0.9, 2: 0.1}), excitation=True)
        sf.add_element(2, 1e8, pynonthermal.IonBalance(HELIUM_ALPHAS))
        sf.solve(deposition_ev_per_s_per_cm3=1e8)

        assert np.array_equal(result.yvec, sf.yvec)
        assert np.array_equal(result.engrid, sf.engrid)
        assert result.n_e == sf.get_n_e()
        assert result.n_ion_tot == sf.get_n_ion_tot()
        assert result.n_e_nt == sf.get_n_e_nt()
        assert result.frac_heating == sf.get_frac_heating()
        assert result.frac_ionisation == sf.get_frac_ionisation_tot()
        assert result.frac_excitation == sf.get_frac_excitation_tot()
        assert result.frac_sum == sf.get_frac_sum()
        assert result.frac_ionisation_ion(8, 1) == sf.get_frac_ionisation_ion(8, 1)
        assert result.frac_excitation_ion(8, 1) == sf.get_frac_excitation_ion(8, 1)
        assert result.ionisation_ratecoeff(8, 1) == sf.get_ionisation_ratecoeff(8, 1)
        assert result.eff_ionpot(8, 1) == sf.get_eff_ionpot(8, 1)
        assert result.ion_populations(2) == {(2, stage)[1]: sf.ionpopdict[(2, stage)] for stage in (1, 2, 3)}
        assert result.ion_fractions(2) == sf.get_ion_fractions(2)
        assert result.balance_iterations == sf.balance_iterations
        assert result.temperature == 6000
        assert result.deposition_ev_per_s_per_cm3 == 1e8

        transitionkey = result.transitionkeys(8, 1)[0]
        assert result.excitation_ratecoeff(8, 1, transitionkey) == sf.get_excitation_ratecoeff(8, 1, transitionkey)


def test_result_is_read_only() -> None:
    plasma = pynonthermal.Plasma(elements=[pynonthermal.Element(8, 1e9, pynonthermal.Fixed({1: 0.9, 2: 0.1}))])
    result = pynonthermal.solve_spencerfano(plasma, 1e8, emin_ev=1, emax_ev=3000, npts=200)
    for array in (result.yvec, result.engrid):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 1.0
    # the plasma that the solution used cannot change either
    with pytest.raises(AttributeError):
        setattr(plasma, "temperature", 5000)  # noqa: B010


def test_plasma_validation() -> None:
    element = pynonthermal.Element(8, 1e9, pynonthermal.Fixed({1: 1.0}))
    with pytest.raises(ValueError, match="at least one element"):
        pynonthermal.Plasma(elements=[])
    with pytest.raises(ValueError, match="own atomic number"):
        pynonthermal.Plasma(elements=[element, element])
    not_an_element: t.Any = [{"Z": 8}]
    with pytest.raises(TypeError, match="must be an Element"):
        pynonthermal.Plasma(elements=not_an_element)
    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="temperature"):
            pynonthermal.Plasma(elements=[element], temperature=bad)
        with pytest.raises(ValueError, match="free_electron_density"):
            pynonthermal.Plasma(elements=[element], free_electron_density=bad)

    # a Saha element needs the temperature of the plasma
    saha = pynonthermal.Plasma(elements=[pynonthermal.Element(8, 1e9, pynonthermal.Saha([1, 2]))])
    with pytest.raises(ValueError, match="Call set_temperature"):
        pynonthermal.solve_spencerfano(saha, 1e8, emin_ev=1, emax_ev=3000, npts=200)

    # a balanced element sets the free electron density itself
    balanced = pynonthermal.Plasma(
        elements=[pynonthermal.Element(2, 1e8, pynonthermal.IonBalance(HELIUM_ALPHAS))],
        free_electron_density=1e5,
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        pynonthermal.solve_spencerfano(balanced, 1e8, emin_ev=1, emax_ev=3000, npts=200)


def test_custom_cross_sections_of_a_plasma() -> None:
    # a cross section given as a function needs no energy grid, so the plasma is grid independent
    def channel_xs(en_ev: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.where(en_ev > 60.0, 2e-17, 0.0)

    def excitation_xs(en_ev: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.where(en_ev > 21.0, 1e-17, 0.0)

    plasma = pynonthermal.Plasma(
        elements=[
            pynonthermal.Element(
                2,
                1e8,
                pynonthermal.IonBalance(HELIUM_ALPHAS),
                builtin_channels=False,
                ionisation_channels=[
                    pynonthermal.CustomChannel(1, 24.6, channel_xs, key="He I custom"),
                    pynonthermal.CustomChannel(2, 54.4, channel_xs, key="He II custom"),
                ],
                excitations=[pynonthermal.CustomExcitation(1, 0.25, 21.0, excitation_xs, key="He I custom")],
            )
        ]
    )

    for npts in (200, 300):
        result = pynonthermal.solve_spencerfano(plasma, 1e8, emin_ev=1, emax_ev=3000, npts=npts, balance_tol=1e-6)
        # the balance holds with the custom channels driving it
        n_e = result.n_e
        for upper, alpha in HELIUM_ALPHAS.items():
            populations = result.ion_populations(2)
            rate_ionisation = populations[upper - 1] * result.ionisation_ratecoeff(2, upper - 1)
            assert math.isclose(rate_ionisation, populations[upper] * n_e * alpha, rel_tol=2e-6)
        assert result.excitation_ratecoeff(2, 1, "He I custom") > 0.0
        assert result.transitionkeys(2, 1) == ["He I custom"]


def test_deprecated_deposition_argument() -> None:
    # the former name of the deposition rate density still works on the low-level solver
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        sf.add_ionisation(8, 2, n_ion=1e8)
        with pytest.warns(DeprecationWarning, match="deposition_ev_per_s_per_cm3"):
            sf.solve(depositionratedensity_ev=1e8)
        assert sf.deposition_ev_per_s_per_cm3 == 1e8
        with pytest.warns(DeprecationWarning, match="deposition_ev_per_s_per_cm3"):
            assert sf.depositionratedensity_ev == 1e8

        both: t.Any = {"deposition_ev_per_s_per_cm3": 1e8, "depositionratedensity_ev": 1e8}
        with (
            pytest.warns(DeprecationWarning, match="deposition_ev_per_s_per_cm3"),
            pytest.raises(ValueError, match="once"),
        ):
            sf.solve(**both)
        with pytest.raises(ValueError, match="needs the deposition rate density"):
            sf.solve()
