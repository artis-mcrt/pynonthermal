"""Check that the excitations from the level data include the transitions below emin_ev."""

import math

import numpy as np
import polars as pl

import pynonthermal


def _level_table() -> pl.DataFrame:
    # the transition from level 0 to level 1 has 0.5 eV, below the emin_ev of 1 eV of the tests
    levels = pl.DataFrame({"levelindex": [0, 1, 2], "energy_ev": [0.0, 0.5, 4.0], "g": [1.0, 3.0, 5.0]})
    transitions = pl.DataFrame(
        {
            "lower": [0, 0, 1],
            "upper": [1, 2, 2],
            "A": [1e6, 1e8, 1e8],
            "collstr": [-1.0, -1.0, -1.0],
            "forbidden": [0, 0, 0],
            "epsilon_trans_ev": [0.5, 4.0, 3.5],
            "lower_g": [1.0, 1.0, 3.0],
            "upper_g": [3.0, 5.0, 5.0],
        }
    )
    return pl.DataFrame(
        {
            "Z": [8],
            "ion_stage": [2],
            "levels": pl.Series([levels], dtype=pl.Object),
            "transitions": pl.Series([transitions], dtype=pl.Object),
        }
    )


def _get_solver() -> pynonthermal.SpencerFanoSolver:
    sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=100, npts=100)
    sf.override_n_e(1e6)
    return sf


def test_level_data_includes_a_transition_below_emin() -> None:
    # A 100 eV electron can excite a 0.5 eV transition on a grid that starts at 1 eV. add_excitation()
    # accepts the transition, and the level data left it out.
    sf_levels = _get_solver()
    sf_levels.set_temperature(6000)
    sf_levels.set_atomic_data(adata_polars=_level_table())
    sf_levels.add_ion_excitation(8, 2, n_ion=1e8)
    transitions = sf_levels.excitationlists[8, 2]
    assert transitions.keys() == {(0, 1), (0, 2), (1, 2)}
    assert transitions[0, 1].epsilon_trans_ev == 0.5
    assert transitions[0, 1].xs_vec.all()

    # the same transitions through add_excitation(), with the same cross sections and level populations
    sf_direct = _get_solver()
    for transitionkey, transition in transitions.items():
        sf_direct.add_excitation(
            8,
            2,
            transition.levelnumberdensity,
            transition.xs_vec,
            transition.epsilon_trans_ev,
            transitionkey=transitionkey,
        )

    sf_levels.solve(deposition_ev_per_s_per_cm3=1e5)
    sf_direct.solve(deposition_ev_per_s_per_cm3=1e5)
    np.testing.assert_allclose(sf_levels.sfmatrix, sf_direct.sfmatrix, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(sf_levels.yvec, sf_direct.yvec, rtol=1e-12)
    assert math.isclose(sf_levels.get_frac_excitation_tot(), sf_direct.get_frac_excitation_tot(), rel_tol=1e-12)
    assert math.isclose(sf_levels.get_frac_heating(), sf_direct.get_frac_heating(), rel_tol=1e-12)
    assert math.isclose(sf_levels.get_frac_sum(), 1.0, rel_tol=1e-12)
    for transitionkey in transitions:
        ratecoeff = sf_levels.get_excitation_ratecoeff(8, 2, transitionkey)
        assert ratecoeff > 0.0
        assert math.isclose(ratecoeff, sf_direct.get_excitation_ratecoeff(8, 2, transitionkey), rel_tol=1e-12)


def test_bundled_level_data_includes_the_transitions_below_emin() -> None:
    # Fe II at 6000 K has 44 active transitions below 1 eV in the bundled table, with the default level limits.
    # Without them, the excitation fraction was 0.26615.
    sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=1000, npts=512)
    sf.set_temperature(6000)
    sf.add_ionisation(26, 2, 1e10)
    sf.add_ion_excitation(26, 2)
    sf.override_n_e(1e7)
    sf.solve(deposition_ev_per_s_per_cm3=1e5)
    epsilons = [transition.epsilon_trans_ev for transition in sf.excitationlists[26, 2].values()]
    assert sum(epsilon < 1.0 for epsilon in epsilons) == 44
    assert math.isclose(sf.get_frac_excitation_tot(), 0.27355, abs_tol=1e-5)
    assert math.isclose(sf.get_frac_sum(), 1.0, rel_tol=1e-12)
