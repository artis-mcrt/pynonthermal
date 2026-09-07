"""Tests of add_element() and its population rules."""

import math

import numpy as np
import numpy.typing as npt
import pytest

import pynonthermal

HELIUM_ALPHAS = {2: 4e-13, 3: 2e-12}


def test_ion_densities_rule() -> None:
    # ion_densities gives the number densities directly, and n_elem is their sum
    densities = {1: 9.9e9, 2: 1e8}
    with (
        pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf_density,
        pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf_fraction,
    ):
        for sf in (sf_density, sf_fraction):
            sf.set_temperature(6000)
            sf.set_atomic_data(use_collstrengths=False)
        sf_density.add_element(8, ion_densities=densities, excitation=True)
        sf_fraction.add_element(8, 1e10, ion_fractions={1: 0.99, 2: 0.01}, excitation=True)
        for sf in (sf_density, sf_fraction):
            sf.solve(deposition_ev_per_s_per_cm3=1e8)

        # the densities are kept exactly, and the two rules describe the same gas
        assert {stage: sf_density.ionpopdict[(8, stage)] for stage in (1, 2)} == densities
        assert sf_density.get_n_ion_tot() == sum(densities.values())
        assert np.array_equal(sf_density.yvec, sf_fraction.yvec)
        assert sf_density.ionpopdict == sf_fraction.ionpopdict


def test_population_rule_validation() -> None:
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        # exactly one rule
        with pytest.raises(ValueError, match="exactly one of ion_densities"):
            sf.add_element(8, 1e9)
        with pytest.raises(ValueError, match="exactly one of ion_densities"):
            sf.add_element(8, 1e9, ion_fractions={1: 1.0}, recomb_ratecoeffs={2: 1e-12})
        with pytest.raises(ValueError, match="exactly one of ion_densities"):
            sf.add_element(8, ion_densities={1: 1e9}, ion_fractions={1: 1.0})

        # n_elem is the sum of ion_densities, and every other rule needs it
        with pytest.raises(ValueError, match="n_elem is the sum of ion_densities"):
            sf.add_element(8, 1e10, ion_densities={1: 1e9})
        with pytest.raises(ValueError, match="n_elem is required with ion_fractions"):
            sf.add_element(8, ion_fractions={1: 1.0})
        with pytest.raises(ValueError, match="n_elem is required with recomb_ratecoeffs"):
            sf.add_element(8, recomb_ratecoeffs={2: 1e-12})
        for bad in (0.0, -1.0, math.nan, math.inf):
            with pytest.raises(ValueError, match="n_elem must be greater than zero"):
                sf.add_element(8, bad, ion_fractions={1: 1.0})

        # the densities and the fractions of one element
        with pytest.raises(ValueError, match="at least one ion density"):
            sf.add_element(8, ion_densities={})
        with pytest.raises(ValueError, match="must sum to a number greater than zero"):
            sf.add_element(8, ion_densities={1: 0.0, 2: 0.0})
        for bad in (-1.0, math.nan, math.inf):
            with pytest.raises(ValueError, match="must be non-negative and finite"):
                sf.add_element(8, ion_densities={1: bad})
        with pytest.raises(ValueError, match="at least one ion fraction"):
            sf.add_element(8, 1e10, ion_fractions={})
        with pytest.raises(ValueError, match="must sum to one"):
            sf.add_element(8, 1e10, ion_fractions={1: 0.5, 2: 0.4})
        with pytest.raises(ValueError, match="between 0 and 1"):
            sf.add_element(8, 1e10, ion_fractions={1: 1.5, 2: -0.5})
        with pytest.raises(ValueError, match="between 1 and 9"):
            sf.add_element(8, ion_densities={10: 1e9})

        # every rejected call leaves the solver unchanged
        assert not sf.ionpopdict
        assert not sf.sfmatrix.any()

        # a zero-density stage is registered without channels, so n_ion=None works for it later
        sf.add_element(8, ion_densities={1: 1e9, 2: 0.0})
        assert sf.ionpopdict == {(8, 1): 1e9, (8, 2): 0.0}
        assert (8, 2) not in sf._ionisation_channels
        sf.add_ionisation(8, 2, None)
        assert (8, 2) in sf._ionisation_channels


def test_cross_sections_can_be_functions() -> None:
    # a cross section given as a function needs no energy grid, so the same one works on any grid
    def channel_xs(en_ev: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.where(en_ev > 60.0, 2e-17, 0.0)

    def excitation_xs(en_ev: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.where(en_ev > 21.0, 1e-17, 0.0)

    for npts in (200, 300):
        with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=npts) as sf:
            sf.add_element(2, 1e8, recomb_ratecoeffs=HELIUM_ALPHAS, builtin_channels=False)
            sf.add_ionisation_channel(2, 1, None, 24.6, channel_xs, "He I custom")
            sf.add_ionisation_channel(2, 2, None, 54.4, channel_xs, "He II custom")
            sf.add_excitation(2, 1, None, excitation_xs, 21.0, "He I custom", levelpopfrac=0.25)
            sf.solve(deposition_ev_per_s_per_cm3=1e8, balance_tol=1e-6)

            # the balance holds with the custom channels driving it
            n_e = sf.get_n_e()
            for upper, alpha in HELIUM_ALPHAS.items():
                rate_ionisation = sf.ionpopdict[(2, upper - 1)] * sf.get_ionisation_ratecoeff(2, upper - 1)
                assert math.isclose(rate_ionisation, sf.ionpopdict[(2, upper)] * n_e * alpha, rel_tol=2e-6)
            assert sf.get_excitation_ratecoeff(2, 1, "He I custom") > 0.0
            assert list(sf.excitationlists[(2, 1)]) == ["He I custom"]


def test_deprecated_deposition_argument() -> None:
    # the former name of the deposition rate density still works
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        sf.add_ionisation(8, 2, n_ion=1e8)
        with pytest.warns(DeprecationWarning, match="deposition_ev_per_s_per_cm3"):
            sf.solve(depositionratedensity_ev=1e8)
        assert sf.deposition_ev_per_s_per_cm3 == 1e8
        with pytest.warns(DeprecationWarning, match="deposition_ev_per_s_per_cm3"):
            assert sf.depositionratedensity_ev == 1e8

        with (
            pytest.warns(DeprecationWarning, match="deposition_ev_per_s_per_cm3"),
            pytest.raises(ValueError, match="once"),
        ):
            sf.solve(1e8, depositionratedensity_ev=1e8)
        with pytest.raises(ValueError, match="needs the deposition rate density"):
            sf.solve()


def test_messages_name_the_likely_mistake() -> None:
    # the messages of the mistakes that a new user makes must say what to do instead
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        # ion_stage 0 means that the caller used the charge
        with pytest.raises(ValueError, match="ion_stage is one more than the charge"):
            sf.add_element(26, ion_densities={0: 3e5, 1: 7e5})
        with pytest.raises(ValueError, match="ion_stage is one more than the charge"):
            sf.add_ionisation(26, 0, n_ion=1e6)

        # recomb_ratecoeffs keyed by the stage that ionises, not by the stage that recombines
        with pytest.raises(ValueError, match="Each key is the ion stage that recombines"):
            sf.add_element(8, 1e10, recomb_ratecoeffs={1: 3e-13, 2: 3e-12})

        # the message names what needs the temperature
        with pytest.raises(ValueError, match="the LTE population of each lower level"):
            sf.add_element(8, 1e10, ion_fractions={1: 1.0}, excitation=True)


def test_result_getters_name_the_ions_that_the_solver_holds() -> None:
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        sf.set_temperature(6000)
        sf.add_element(8, ion_densities={1: 1e9, 2: 1e8}, excitation=True)
        sf.solve(deposition_ev_per_s_per_cm3=1e8)

        # an element that the solver does not hold, and a stage that it does not hold
        for getter in (sf.get_frac_ionisation_ion, sf.get_frac_excitation_ion, sf.get_eff_ionpot):
            with pytest.raises(ValueError, match=r"holds no ion of Z=26, but holds the elements \[8\]"):
                getter(26, 2)
        with pytest.raises(ValueError, match=r"Z=8 has the ion stages \[1, 2\]"):
            sf.get_ionisation_ratecoeff(8, 5)

        # the transitions of an ion, and the ions that have transitions
        with pytest.raises(ValueError, match="has no transition"):
            sf.get_excitation_ratecoeff(8, 2, (0, 9999))
        with pytest.raises(ValueError, match="Its ions with excitations are"):
            sf.get_excitation_ratecoeff(26, 2, (0, 1))

        # a changed free electron density says which call discarded the solution
        sf.override_n_e(1e7)
        with pytest.raises(RuntimeError, match=r"override_n_e\(\).*call solve\(\) again"):
            sf.get_frac_heating()
        sf.solve(deposition_ev_per_s_per_cm3=1e8)
        assert sf.get_frac_heating() > 0.0
