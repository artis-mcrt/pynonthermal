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


@pytest.mark.parametrize("slope", [0.0, 1000.0])
def test_zero_energy_rate_rejects_function_with_step_at_threshold(slope: float) -> None:
    # The secondary distribution is 2 / (E' - I) near the threshold. With a step in the cross section, the
    # rate at zero energy is infinite, and the midpoint sum gave a finite value that grew with the sub-grid.
    # With the slope, the step is small in relation to the cross section at the first node of the sub-grid.
    sf = pynonthermal.SpencerFanoSolver(npts=200, emax_ev=500)
    ionpot_ev = float(sf.engrid[10]) + 0.5 * sf.deltaen
    sf.add_ionisation_channel(
        1,
        1,
        1e7,
        ionpot_ev,
        lambda en_ev: np.where(en_ev > ionpot_ev, 1e-17 * (1.0 + slope * (en_ev / ionpot_ev - 1.0)), 0.0),
    )
    sf.override_n_e(1e7)
    sf.solve(1e8)

    with pytest.raises(ValueError, match="does not go to zero continuously at ionpot_ev"):
        sf.calculate_N_e(0.0)
    # the rate above zero energy and the heating fraction do not use the rate at zero energy
    assert 0.0 < sf.calculate_N_e(0.5) < np.inf
    assert 0.0 < sf.get_frac_heating() < 1.0


@pytest.mark.parametrize("step", [True, False])
@pytest.mark.parametrize("spacings_below_emax", [6, 2**24])
def test_zero_energy_rate_step_check_with_threshold_near_emax(step: bool, spacings_below_emax: int) -> None:
    # With a threshold very near to emax_ev, the two energies of the check were equal to the threshold in
    # floating point. The check then did not find the step. With a threshold some spacings of the
    # floating-point numbers below emax_ev, one energy of the check was above the grid, where this function is zero.
    sf = pynonthermal.SpencerFanoSolver(npts=200, emax_ev=500)
    emax_ev = float(sf.engrid[-1])
    ionpot_ev = emax_ev - spacings_below_emax * float(np.spacing(emax_ev))

    def xs(en_ev: np.ndarray) -> np.ndarray:
        in_range = (en_ev > ionpot_ev) & (en_ev <= emax_ev)
        return np.where(in_range, 1e-17 if step else 1e-17 * (en_ev / ionpot_ev - 1.0), 0.0)

    sf.add_ionisation_channel(1, 1, 1e7, ionpot_ev, xs)
    sf.override_n_e(1e7)
    sf.solve(1e8)

    if step:
        with pytest.raises(ValueError, match="does not go to zero continuously at ionpot_ev"):
            sf.calculate_N_e(0.0)
    else:
        assert 0.0 < sf.calculate_N_e(0.0) < np.inf


@pytest.mark.parametrize("exponent", [0.5, 1.0])
def test_zero_energy_rate_accepts_function_that_is_zero_at_threshold(exponent: float) -> None:
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


def test_zero_energy_rate_accepts_step_above_threshold_and_empty_ion() -> None:
    # A step above the threshold is not at the singular point. An ion with no population gives no rate.
    sf = pynonthermal.SpencerFanoSolver(npts=200, emax_ev=500)
    ionpot_ev = float(sf.engrid[10]) + 0.5 * sf.deltaen
    sf.add_ionisation_channel(1, 1, 1e7, ionpot_ev, lambda en_ev: np.where(en_ev > ionpot_ev + 5.0, 1e-17, 0.0))
    sf.add_ionisation_channel(8, 2, 0.0, 40.0, lambda en_ev: np.where(en_ev > 40.0, 1e-18, 0.0))
    sf.override_n_e(1e7)
    sf.solve(1e8)

    assert 0.0 < sf.calculate_N_e(0.0) < np.inf
