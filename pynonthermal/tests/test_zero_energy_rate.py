"""Check the electron source at zero energy."""

import numpy as np
import pytest

import pynonthermal


def test_zero_energy_rate_agrees_with_positive_energy_limit() -> None:
    sf = pynonthermal.SpencerFanoSolver(npts=100, emax_ev=500)
    sf.add_element(1, ion_densities={1: 1e7, 2: 1e7})
    sf.solve(1e8)

    rate = sf.calculate_N_e(0.0)
    assert rate > 0.0
    assert rate == pytest.approx(sf.calculate_N_e(1e-6), rel=1e-4)
    assert rate == pytest.approx(sf.calculate_N_e(1e-4), rel=1e-4)


def test_zero_energy_excitation_source() -> None:
    sf = pynonthermal.SpencerFanoSolver(npts=100, emax_ev=100)
    threshold = float(sf.engrid[20])
    sf.add_excitation(1, 1, 1e7, np.full(100, 1e-17), threshold)
    sf.override_n_e(1e7)
    sf.solve(1e8)

    expected = 1e7 * 1e-17 * sf.yvec[20]
    assert expected > 0.0
    assert sf.calculate_N_e(0.0) == pytest.approx(expected)


def test_heating_integral_excludes_zero_energy_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    sf = pynonthermal.SpencerFanoSolver(npts=100, emax_ev=500)
    sf.add_element(1, ion_densities={1: 1e7, 2: 1e7})
    sf.solve(1e8)
    expected = sf.get_frac_heating()
    sf.reset_solution_analysis()
    original = sf._calculate_N_e

    def check_energy(energy_ev: float, include_excitation_terms: bool) -> float:
        assert energy_ev > 0.0
        return original(energy_ev, include_excitation_terms)

    monkeypatch.setattr(sf, "_calculate_N_e", check_energy)
    assert sf.get_frac_heating() == expected


def test_zero_energy_rate_with_array_threshold_between_grid_points() -> None:
    # The interpolation of the array gave a positive cross section at the threshold. The secondary
    # distribution diverges there, so the rate depended on the sub-grid and grew without limit at small energies.
    sf = pynonthermal.SpencerFanoSolver(npts=200, emax_ev=500)
    ionpot_ev = float(sf.engrid[10]) + 0.5 * sf.deltaen
    xs_vec = np.where(sf.engrid > ionpot_ev, 1e-17 * np.log(np.maximum(sf.engrid / ionpot_ev, 1.0)), 0.0)
    sf.add_ionisation_channel(1, 1, 1e7, ionpot_ev, xs_vec)
    sf.override_n_e(1e7)
    sf.solve(1e8)

    channel = sf._ionisation_channels[(1, 1)][0]
    # the cross section between the grid points goes to zero at the threshold and keeps the grid values
    assert channel.xs(np.array([ionpot_ev, ionpot_ev + 1e-9 * sf.deltaen])) == pytest.approx([0.0, 0.0], abs=1e-25)
    assert np.array_equal(channel.xs(sf.engrid), xs_vec)
    en_first = float(sf.engrid[11])
    assert channel.xs(np.array([0.5 * (ionpot_ev + en_first)]))[0] == pytest.approx(0.5 * xs_vec[11])

    rate = sf.calculate_N_e(0.0)
    assert rate > 0.0
    for energy_ev in (1e-4, 1e-6, 1e-10):
        assert rate == pytest.approx(sf.calculate_N_e(energy_ev), rel=1e-4)


@pytest.mark.parametrize("exponent", [0.5, 1.0])
def test_zero_energy_rate_with_function_that_is_zero_at_threshold(exponent: float) -> None:
    sf = pynonthermal.SpencerFanoSolver(npts=200, emax_ev=500)
    ionpot_ev = float(sf.engrid[10]) + 0.5 * sf.deltaen
    sf.add_ionisation_channel(
        1, 1, 1e7, ionpot_ev, lambda en_ev: 1e-17 * np.clip(en_ev / ionpot_ev - 1.0, 0.0, None) ** exponent
    )
    sf.override_n_e(1e7)
    sf.solve(1e8)

    rate = sf.calculate_N_e(0.0)
    assert 0.0 < rate < np.inf
    assert rate == pytest.approx(sf.calculate_N_e(1e-2), rel=1e-3)
