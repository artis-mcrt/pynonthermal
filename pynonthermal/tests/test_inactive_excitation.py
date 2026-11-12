"""Check the exclusion of inactive transitions from the level data."""

import numpy as np
import polars as pl
import pytest

import pynonthermal


def _level_table() -> pl.DataFrame:
    levels = pl.DataFrame({"levelindex": [0, 1, 2, 3], "energy_ev": [0.0, 2.0, 4.0, 6.0], "g": [1.0, 3.0, 5.0, 7.0]})
    transitions = pl.DataFrame(
        {
            "lower": [0, 0, 0],
            "upper": [1, 2, 3],
            "A": [1e8, 0.0, 0.0],
            "collstr": [-1.0, 1.0, -1.0],
            "forbidden": [0, 1, 0],
            "epsilon_trans_ev": [2.0, 4.0, 6.0],
            "lower_g": [1.0, 1.0, 1.0],
            "upper_g": [3.0, 5.0, 7.0],
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


@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("use_collstrengths", [False, True])
def test_forbidden_transitions_require_collision_strengths(balanced: bool, use_collstrengths: bool) -> None:
    sf = pynonthermal.SpencerFanoSolver(npts=100, emax_ev=100)
    sf.set_temperature(6000)
    sf.set_atomic_data(adata_polars=_level_table(), use_collstrengths=use_collstrengths)
    if balanced:
        sf.add_element(8, n_elem=1e8, recomb_ratecoeffs={3: 1e-12}, excitation=True)
    else:
        sf.add_ion_excitation(8, 2, n_ion=1e8)

    expected_keys = {(0, 1), (0, 2)} if use_collstrengths else {(0, 1)}
    assert sf.excitationlists[(8, 2)].keys() == expected_keys
    assert all(transition.xs_vec.any() for transition in sf.excitationlists[(8, 2)].values())


def test_excluded_transitions_do_not_change_active_results() -> None:
    def solve(include_zero: bool) -> pynonthermal.SpencerFanoSolver:
        sf = pynonthermal.SpencerFanoSolver(npts=100, emax_ev=100)
        sf.set_temperature(6000)
        sf.set_atomic_data(adata_polars=_level_table(), use_collstrengths=False)
        sf.add_ion_excitation(8, 2, n_ion=1e8)
        if include_zero:
            sf.add_excitation(8, 2, 1e8, np.zeros(100), 4.0, transitionkey=(0, 2))
            sf.add_excitation(8, 2, 1e8, np.zeros(100), 6.0, transitionkey=(0, 3))
        sf.solve(1e8)
        return sf

    compact = solve(False)
    reference = solve(True)
    np.testing.assert_array_equal(compact.sfmatrix, reference.sfmatrix)
    np.testing.assert_array_equal(compact.yvec, reference.yvec)
    assert compact.get_excitation_ratecoeff(8, 2, (0, 1)) == reference.get_excitation_ratecoeff(8, 2, (0, 1))
    assert compact.get_frac_excitation_tot() == reference.get_frac_excitation_tot()
    assert compact.get_frac_heating() == reference.get_frac_heating()
