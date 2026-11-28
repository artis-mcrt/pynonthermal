"""Tests of the Auger data of the built-in channels (use_km1993_auger=True).

The option follows the method of ARTIS (Shingles et al. 2020, MNRAS, 492, 2029-2043,
doi:10.1093/mnras/stz3412) with the data of Kaastra & Mewe (1993, A&AS, 97, 443-482, ADS
1993A&AS...97..443K). These tests check:

- the reader of the table;
- the channels of a subshell with Auger data;
- the source term in the matrix;
- the energy fractions and the rate coefficients;
- the top stage of an ionisation balance;
- the default, which has no Auger data.
"""

import math

import numpy as np
import pytest

import pynonthermal

# number densities [cm^-3] of Fe I to Fe IV. The values are only examples.
IRON_DENSITIES = {1: 1e3, 2: 5e4, 3: 4e4, 4: 1e3}


def solve_iron(use_km1993_auger: bool = False) -> pynonthermal.SpencerFanoSolver:
    sf = pynonthermal.SpencerFanoSolver(npts=1500, use_km1993_auger=use_km1993_auger)
    sf.add_element(Z=26, ion_densities=IRON_DENSITIES)
    sf.override_n_e(1.4e5)
    sf.solve(deposition_ev_per_s_per_cm3=1e3)
    return sf


def test_read_auger_data() -> None:
    auger_data = pynonthermal.collion.read_auger_data()

    # the K shell of Na I is one row of the table: E_A = 953.6 eV, epsilon = 0.010, and I = 1075 eV
    en_auger_ev, probabilities = auger_data[(11, 1, 1, 0)]
    assert math.isclose(en_auger_ev, 953.6 - 0.010 * 1075.0, rel_tol=1e-12)
    assert probabilities[:3] == (0.026, 0.6342, 0.3398)
    assert len(probabilities) == pynonthermal.collion.KM93_MAX_AUGER_ELECTRONS + 1

    # the 3p subshell of Fe I is the mean of the M2 and M3 rows, with the weights 2 and 4
    en_auger_ev, probabilities = auger_data[(26, 1, 3, 1)]
    en_m2_ev = 43.0 - 0.101 * 60.9
    en_m3_ev = 41.1 - 0.104 * 59.0
    assert math.isclose(en_auger_ev, (2 * en_m2_ev + 4 * en_m3_ev) / 6, rel_tol=1e-12)
    assert math.isclose(sum(probabilities), 1.0, abs_tol=1e-3)

    # the correction of the 3s subshell of Fe III is larger than E_A, so the energy is zero
    assert auger_data[(26, 3, 3, 0)][0] == 0.0

    # a write to the result does not change the table of the next caller
    auger_data.clear()
    assert (11, 1, 1, 0) in pynonthermal.collion.read_auger_data()


def test_default_channels_have_no_auger_data() -> None:
    sf = pynonthermal.SpencerFanoSolver(npts=500)
    sf.add_element(Z=26, ion_densities=IRON_DENSITIES)
    for channels in sf._ionisation_channels.values():
        for channel in channels:
            assert channel.n_ejected == 1
            assert channel.auger_electron_energy_ev == 0.0
            assert not channel.mean_auger_electron


def test_auger_channels_of_a_subshell() -> None:
    engrid = np.linspace(0.1, 16000.0, 2000)
    dfcollion = pynonthermal.collion.read_colliondata()
    auger_data = pynonthermal.collion.read_auger_data()
    default_channels = pynonthermal.collion.get_ion_ionisation_channels(dfcollion, 26, 3, engrid)
    channels = pynonthermal.collion.get_ion_ionisation_channels(
        dfcollion, 26, 3, engrid, use_km1993_auger=True, max_auger_electrons=2
    )

    # the 3d subshell of Fe III has no Auger data, so its channel does not change
    assert channels[0].key == default_channels[0].key == "n 3 l 2"
    assert not channels[0].mean_auger_electron
    assert np.array_equal(channels[0].xs_grid, default_channels[0].xs_grid)

    for default_channel, n, l in ((default_channels[1], 3, 1), (default_channels[2], 3, 0)):  # noqa: E741
        en_auger_ev, probabilities = auger_data[(26, 3, n, l)]
        shell_channels = [channel for channel in channels if str(channel.key).startswith(f"n {n} l {l},")]
        assert shell_channels
        for channel in shell_channels:
            assert channel.mean_auger_electron
            assert channel.n_auger_electrons == 1
            assert channel.ionpot_ev == default_channel.ionpot_ev
            assert channel.J_ev == default_channel.J_ev
            assert channel.auger_electron_energy_ev == en_auger_ev
            probability = probabilities[channel.n_ejected - 1] / sum(probabilities[:3])
            assert np.allclose(channel.xs_grid, probability * default_channel.xs_grid, rtol=1e-13, atol=0.0)
            # the function between the grid points has the same factor
            en_ev = np.array([150.0, 333.3])
            assert np.allclose(channel.xs(en_ev), probability * default_channel.xs(en_ev), rtol=1e-13, atol=0.0)
        # the channels of the subshell keep its total cross section
        xs_sum = np.sum([channel.xs_grid for channel in shell_channels], axis=0)
        assert np.allclose(xs_sum, default_channel.xs_grid, rtol=1e-13, atol=0.0)

    # the 3s subshell of Fe III emits two Auger electrons with the probability 0.9783. A limit of one
    # Auger electron moves that probability to n_ejected=2.
    capped = pynonthermal.collion.get_ion_ionisation_channels(
        dfcollion, 26, 3, engrid, use_km1993_auger=True, max_auger_electrons=1
    )
    assert max(channel.n_ejected for channel in capped) == 2
    assert [channel.n_ejected for channel in capped if str(channel.key).startswith("n 3 l 0,")] == [2]


def test_auger_source_term_in_the_matrix() -> None:
    # With the option, the matrix of an ion is the default matrix minus one source electron for
    # each ionisation of a subshell with Auger data. Each row below the mean Auger energy gets the
    # full source, and the last row gets the weight that gives the electron its true energy.
    n_ion = 1e6
    matrices = {}
    for use_km1993_auger in (False, True):
        sf = pynonthermal.SpencerFanoSolver(npts=800, use_km1993_auger=use_km1993_auger)
        sf.add_ionisation(Z=26, ion_stage=2, n_ion=n_ion)
        matrices[use_km1993_auger] = sf.sfmatrix.copy()
    default_channels = pynonthermal.collion.get_ion_ionisation_channels(sf.dfcollion, 26, 2, sf.engrid)

    expected = np.zeros_like(sf.sfmatrix)
    n_sources = 0
    for channel in default_channels:
        n, l = (int(part) for part in str(channel.key).replace("n ", "").split(" l "))  # noqa: E741
        en_auger_ev = pynonthermal.collion.read_auger_data().get((26, 2, n, l), (0.0, ()))[0]
        if en_auger_ev > 0.0:
            n_sources += 1
            row_weights = np.clip((en_auger_ev - sf.engrid) / sf.deltaen, 0.0, 1.0)
            expected -= np.outer(row_weights, n_ion * channel.xs_grid * sf.deltaen)
    assert n_sources > 0
    difference = matrices[True] - matrices[False]
    assert np.allclose(difference, expected, rtol=1e-9, atol=1e-12 * np.abs(expected).max())


def test_auger_energy_fractions_and_rate_coefficients() -> None:
    sf_default = solve_iron()
    sf = solve_iron(use_km1993_auger=True)

    # the Auger electrons return energy to the plasma, so the ionisation fraction falls and the
    # heating fraction increases. The energy stays conserved.
    assert sf.get_frac_ionisation_tot() < 0.95 * sf_default.get_frac_ionisation_tot()
    assert sf.get_frac_heating() > sf_default.get_frac_heating()
    assert math.isclose(sf.get_frac_sum(), sf_default.get_frac_sum(), abs_tol=1e-4)
    assert math.isclose(sf.get_frac_sum(), 1.0, abs_tol=5e-3)

    for ion_stage in (1, 2, 3):
        ratecoeff = sf.get_ionisation_ratecoeff(26, ion_stage)
        # the cross sections do not change, and the solution changes only a little
        assert math.isclose(ratecoeff, sf_default.get_ionisation_ratecoeff(26, ion_stage), rel_tol=0.02)
        by_n_ejected = [sf.get_ionisation_ratecoeff(26, ion_stage, n_ejected) for n_ejected in (1, 2, 3)]
        assert math.isclose(sum(by_n_ejected), ratecoeff, rel_tol=1e-12)
        # an ionisation of the 3p subshell emits one Auger electron
        assert by_n_ejected[1] > 0.0
        assert sf_default.get_ionisation_ratecoeff(26, ion_stage, 2) == 0.0


def test_auger_option_with_an_ionisation_balance() -> None:
    # A stage of the balance cannot go above the top stage of the chain, so the ion of an ionisation
    # that passes the top stage ends in the top stage. The channel keeps the electron count of its
    # Auger data.
    sf = pynonthermal.SpencerFanoSolver(npts=800, use_km1993_auger=True)
    sf.add_element(Z=26, n_elem=1e5, recomb_ratecoeffs={2: 3e-12, 3: 5e-12, 4: 1e-11})
    # Fe III to Fe V and Fe III to Fe VI pass the top stage Fe IV
    assert {channel.n_ejected for channel in sf._ionisation_channels[(26, 3)]} == {1, 2, 3}
    assert any(channel.n_ejected == 2 for channel in sf._ionisation_channels[(26, 1)])
    # the channels of Fe III keep the source term of the Auger electrons
    assert any(channel.mean_auger_electron for channel in sf._ionisation_channels[(26, 3)])

    with pytest.warns(UserWarning, match="top stage"):
        sf.solve(deposition_ev_per_s_per_cm3=1e3)
    assert math.isclose(sum(sf.get_ion_fractions(26).values()), 1.0, rel_tol=1e-9)
    assert math.isclose(sf.get_frac_sum(), 1.0, abs_tol=5e-3)


def test_balance_keeps_the_electron_count_of_each_auger_event() -> None:
    # The balance and a solve with its populations as fixed inputs have the same matrix. Thus they
    # give the same rate coefficient for each number of ejected electrons. With the stage limit in the
    # channels, the balance gave zero for the double and the triple ionisation of Fe III.
    recomb_ratecoeffs = {2: 3e-12, 3: 5e-12, 4: 1e-11}
    sf = pynonthermal.SpencerFanoSolver(npts=200, use_km1993_auger=True)
    sf.add_element(Z=26, n_elem=1e5, recomb_ratecoeffs=recomb_ratecoeffs)
    with pytest.warns(UserWarning, match="leave the chain") as records:
        sf.solve(deposition_ev_per_s_per_cm3=1e3)
    populations = {ion_stage: sf.ionpopdict[(26, ion_stage)] for ion_stage in (1, 2, 3, 4)}

    sf_fixed = pynonthermal.SpencerFanoSolver(npts=200, use_km1993_auger=True)
    sf_fixed.add_element(Z=26, ion_densities=populations)
    sf_fixed.solve(deposition_ev_per_s_per_cm3=1e3)

    np.testing.assert_allclose(sf.yvec, sf_fixed.yvec, rtol=1e-12)
    for ion_stage in (1, 2, 3):
        for n_ejected in (None, 1, 2, 3):
            assert math.isclose(
                sf.get_ionisation_ratecoeff(26, ion_stage, n_ejected),
                sf_fixed.get_ionisation_ratecoeff(26, ion_stage, n_ejected),
                rel_tol=1e-12,
            )
    assert sf.get_ionisation_ratecoeff(26, 3, 2) == pytest.approx(6.96e-6, rel=2e-3)
    assert sf.get_ionisation_ratecoeff(26, 3, 3) == pytest.approx(9.82e-7, rel=2e-3)

    # The populations balance the rates with the ions of the passing events in the top stage: the rate
    # across the cut below Fe IV is equal to the recombination rate of Fe IV.
    n_e = sf.get_n_e()
    rate_up = sum(
        populations[ion_stage] * sf.get_ionisation_ratecoeff(26, ion_stage, n_ejected)
        for ion_stage in (1, 2, 3)
        for n_ejected in (1, 2, 3)
        if ion_stage + n_ejected > 3
    )
    assert math.isclose(rate_up, populations[4] * n_e * recomb_ratecoeffs[4], rel_tol=1e-5)

    # The events that leave the chain are the ionisations of Fe IV and the ionisations from the lower
    # stages that pass Fe IV. The warning gives the sum of their rates, and the stage that holds them all.
    rate_top = populations[4] * sf.get_ionisation_ratecoeff(26, 4)
    rate_passing = sum(
        populations[ion_stage] * sf.get_ionisation_ratecoeff(26, ion_stage, n_ejected)
        for ion_stage in (1, 2, 3)
        for n_ejected in (1, 2, 3)
        if ion_stage + n_ejected > 4
    )
    assert rate_passing > 0.0
    message = next(str(record.message) for record in records if "leave the chain" in str(record.message))
    assert f"is {rate_top + rate_passing:.2e} /s/cm3" in message
    assert f"top stage give {rate_top:.2e} /s/cm3" in message
    # Fe III with three ejected electrons ends in Fe VI
    assert "up to ion stage 6" in message


def test_auger_option_validation() -> None:
    # the function of collion also checks the limit, because a negative limit counted from the end
    dfcollion = pynonthermal.collion.read_colliondata()
    engrid = np.linspace(0.1, 16000.0, 200)
    for invalid in (-1, 1.5, True):
        with pytest.raises(ValueError, match="max_auger_electrons"):
            pynonthermal.collion.get_ion_ionisation_channels(
                dfcollion,
                26,
                3,
                engrid,
                use_km1993_auger=True,
                max_auger_electrons=invalid,  # ty: ignore[invalid-argument-type]
            )
    with pytest.raises(ValueError, match="max_auger_electrons"):
        pynonthermal.SpencerFanoSolver(npts=100, use_km1993_auger=True, max_auger_electrons=-1)
    with pytest.raises(ValueError, match="max_auger_electrons"):
        pynonthermal.SpencerFanoSolver(npts=100, max_auger_electrons=1.5)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="use_km1993_auger"):
        pynonthermal.SpencerFanoSolver(npts=100, use_km1993_auger=1)  # ty: ignore[invalid-argument-type]
