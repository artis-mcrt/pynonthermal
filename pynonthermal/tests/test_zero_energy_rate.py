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
