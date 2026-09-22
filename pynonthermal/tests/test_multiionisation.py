"""Tests of ionisation channels that remove more than one electron (n_ejected > 1).

These check the cut coefficients of the ionisation balance, the source term of the extra
electrons, the energy accounting, and the error paths.
"""

import math

import numpy as np
import numpy.typing as npt
import pytest

import pynonthermal

# illustrative recombination rate coefficients [cm^3 s^-1] keyed by the recombining ion stage
HELIUM_ALPHAS = {2: 4e-13, 3: 2e-12}
OXYGEN_ALPHAS = {2: 3e-13, 3: 3e-12}
STRONTIUM_ALPHAS = {2: 3e-13, 3: 1e-12, 4: 3e-12}


def lotz_like_xs(ionpot_ev: float, scale_cm2: float) -> pynonthermal.CrossSectionFunc:
    # a made-up cross section with the shape of the Lotz formula. It is zero at and below ionpot_ev.
    def xs(en_ev: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        u = np.maximum(np.asarray(en_ev, dtype=np.float64) / ionpot_ev, 1.0)
        return np.where(u > 1.0, scale_cm2 * np.log(u) / u, 0.0)

    return xs


def test_ion_fractions_cuts_single_ionisation_matches_chain() -> None:
    # with only the last value of each cut, the cut recurrence is the ratio chain to the last bit
    rng = np.random.default_rng(11)
    for _ in range(200):
        nstages = int(rng.integers(2, 8))
        ratio_coeffs = [float(10 ** rng.uniform(-10, 10)) for _ in range(nstages - 1)]
        cut_coeffs = [[0.0] * j + [c] for j, c in enumerate(ratio_coeffs)]
        n_e = float(10 ** rng.uniform(-5, 12))
        assert pynonthermal.ionbalance.get_ion_fractions_cuts(
            cut_coeffs, n_e
        ) == pynonthermal.ionbalance.get_ion_fractions(ratio_coeffs, n_e)


def test_ion_fractions_cuts_three_stages() -> None:
    # stages 1, 2, and 3 with the rates Gamma_11, Gamma_12 (1 to 3), and Gamma_21 against the direct formula
    gamma_11, gamma_12, gamma_21 = 3e-5, 1e-5, 2e-6
    alpha_2, alpha_3 = 4e-13, 2e-12
    n_e = 1e7
    cut_coeffs = [[(gamma_11 + gamma_12) / alpha_2], [gamma_12 / alpha_3, gamma_21 / alpha_3]]
    n_1 = 1.0
    n_2 = n_1 * (gamma_11 + gamma_12) / (n_e * alpha_2)
    n_3 = (n_1 * gamma_12 + n_2 * gamma_21) / (n_e * alpha_3)
    total = n_1 + n_2 + n_3
    fractions = pynonthermal.ionbalance.get_ion_fractions_cuts(cut_coeffs, n_e)
    assert np.allclose(fractions, [n_1 / total, n_2 / total, n_3 / total], rtol=1e-14)

    # a jump over a stage fills the stage above it, also with no ionisation out of the stage between
    fractions = pynonthermal.ionbalance.get_ion_fractions_cuts([[1e8], [1e8, 0.0]], n_e)
    assert fractions[2] > 0.0

    # a cut that no ionisation crosses makes every stage above it zero. A jump from a lower stage past
    # the next cut also crosses this cut, so only the stage just below the next cut has a term there.
    fractions = pynonthermal.ionbalance.get_ion_fractions_cuts([[1e8], [0.0, 0.0], [0.0, 0.0, 1e30]], n_e)
    assert fractions[2] == 0.0
    assert fractions[3] == 0.0

    with pytest.raises(ValueError, match="n_e"):
        pynonthermal.ionbalance.get_ion_fractions_cuts([[1.0]], 0.0)
    with pytest.raises(ValueError, match="cut coefficients"):
        pynonthermal.ionbalance.get_ion_fractions_cuts([[-1.0]], n_e)
    with pytest.raises(ValueError, match="must have 2 values"):
        pynonthermal.ionbalance.get_ion_fractions_cuts([[1.0], [1.0]], n_e)


def test_charge_neutral_n_e_cuts_is_unique() -> None:
    # With a multiple ionisation, the mean charge of an element can increase with n_e. The bisection
    # needs only that the mean charge divided by n_e decreases, so that the residual changes sign once.
    # Examine this condition for random rate coefficients of chains of up to 6 stages, with the multiple
    # ionisation larger than the single ionisation in some cases. The condition is not true for every
    # chain: a jump of about 14 stages can break it (see solve_charge_neutral_n_e_cuts()).
    rng = np.random.default_rng(5)
    ln_n_e = np.linspace(math.log(1e-8), math.log(1e12), 600)
    charge_increases = False
    for _ in range(300):
        nstages = int(rng.integers(3, 7))
        # gammas[i][k - 1] is the rate coefficient of stage i with n_ejected = k
        gammas = [[float(10 ** rng.uniform(-8, 2)) for _ in range(1, nstages - i)] for i in range(nstages - 1)]
        alphas = 10 ** rng.uniform(-3, 3, size=nstages)
        cut_coeffs = [
            [sum(gammas[i][k - 1] for k in range(j - i + 1, nstages - i)) / alphas[j + 1] for i in range(j + 1)]
            for j in range(nstages - 1)
        ]
        mean_charge = np.array(
            [
                np.dot(pynonthermal.ionbalance.get_ion_fractions_cuts(cut_coeffs, math.exp(x)), np.arange(nstages))
                for x in ln_n_e
            ]
        )
        charge_increases = charge_increases or bool(np.any(np.diff(mean_charge) > 1e-6))
        # the mean charge divided by n_e decreases, to the rounding of the fractions
        ln_charge_over_n_e = np.log(np.maximum(mean_charge, 1e-300)) - ln_n_e
        assert np.all(np.diff(ln_charge_over_n_e)[mean_charge[1:] > 1e-12] < 1e-9)

        n_elem = float(10 ** rng.uniform(0, 10))
        n_e = pynonthermal.ionbalance.solve_charge_neutral_n_e_cuts(0.0, [(n_elem, 1, cut_coeffs)])
        fractions = pynonthermal.ionbalance.get_ion_fractions_cuts(cut_coeffs, n_e)
        assert math.isclose(n_e, n_elem * float(np.dot(fractions, np.arange(nstages))), rel_tol=1e-9)

    # the test must include the cases that a condition on the mean charge alone would reject
    assert charge_increases


def test_helium_double_ionisation_balance() -> None:
    # He I to He III in one ionisation. The ionpot of the channel is inside MULTIPLE_IONPOT_REL_TOL of the
    # sum of the NIST potentials, so the extra electron gets no energy.
    n_helium = 1e8
    deposition = 1e8
    balance_tol = 1e-6
    ionpot_double_ev = 79.0

    def solve_helium(double: bool) -> pynonthermal.SpencerFanoSolver:
        sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300)
        sf.add_element(2, n_helium, recomb_ratecoeffs=HELIUM_ALPHAS)
        if double:
            sf.add_ionisation_channel(
                2, 1, None, ionpot_double_ev, lotz_like_xs(ionpot_double_ev, 5e-18), "double", n_ejected=2
            )
        sf.solve(deposition_ev_per_s_per_cm3=deposition, balance_tol=balance_tol)
        return sf

    sf = solve_helium(double=True)
    channel = sf._ionisation_channels[(2, 1)][-1]
    assert channel.n_ejected == 2
    assert channel.extra_electron_energy_ev == 0.0

    gamma_11 = sf.get_ionisation_ratecoeff(2, 1, n_ejected=1)
    gamma_12 = sf.get_ionisation_ratecoeff(2, 1, n_ejected=2)
    gamma_21 = sf.get_ionisation_ratecoeff(2, 2, n_ejected=1)
    assert gamma_12 > 0.0
    assert math.isclose(gamma_11 + gamma_12, sf.get_ionisation_ratecoeff(2, 1), rel_tol=1e-12)
    assert sf.get_ionisation_ratecoeff(2, 1, n_ejected=3) == 0.0

    # the flux across each cut is zero: the double ionisation crosses both cuts
    n_1, n_2, n_3 = (sf.ionpopdict[(2, ion_stage)] for ion_stage in (1, 2, 3))
    n_e = sf.get_n_e()
    assert math.isclose(n_1 * (gamma_11 + gamma_12), n_2 * n_e * HELIUM_ALPHAS[2], rel_tol=2 * balance_tol)
    assert math.isclose(n_1 * gamma_12 + n_2 * gamma_21, n_3 * n_e * HELIUM_ALPHAS[3], rel_tol=2 * balance_tol)
    assert math.isclose(n_e, n_2 + 2 * n_3, rel_tol=1e-12)
    assert math.isclose(sf.get_frac_sum(), 1.0, abs_tol=0.02)

    # the double ionisation moves ions to He III
    sf_single = solve_helium(double=False)
    assert sf.ionpopdict[(2, 3)] > sf_single.ionpopdict[(2, 3)]


def test_new_channel_after_a_failed_solve(monkeypatch: pytest.MonkeyPatch) -> None:
    # A solve() that raises keeps the rate coefficients of the balance, and it permits more channels.
    # A channel with a new n_ejected must not make the next solve() fail.
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf:
        sf.add_element(2, 1e8, recomb_ratecoeffs=HELIUM_ALPHAS)
        with monkeypatch.context() as patch:
            patch.setattr(pynonthermal.spencerfano, "BALANCE_MAXITER", 1)
            with pytest.raises(RuntimeError, match="did not converge"):
                sf.solve(deposition_ev_per_s_per_cm3=1e8, balance_tol=1e-12)
        assert sf._balanced_elements[2].ratecoeffs_per_deposition is not None

        sf.add_ionisation_channel(2, 1, None, 79.0, lotz_like_xs(79.0, 5e-18), "double", n_ejected=2)
        assert sf._balanced_elements[2].ratecoeffs_per_deposition is None
        sf.solve(deposition_ev_per_s_per_cm3=1e8)
        assert sf.get_ionisation_ratecoeff(2, 1, n_ejected=2) > 0.0


def test_direct_double_ionisation_has_the_matrix_of_one_secondary() -> None:
    # without energy for the extra electron, a channel with n_ejected=2 fills the matrix exactly as
    # a single-ionisation channel with the same potential and cross section
    matrices = []
    for n_ejected in (1, 2):
        with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf:
            sf.add_ionisation_channel(2, 1, 1e8, 79.0, lotz_like_xs(79.0, 5e-18), n_ejected=n_ejected)
            matrices.append(sf.sfmatrix.copy())
    assert np.array_equal(matrices[0], matrices[1])


def test_extra_electron_source_term() -> None:
    # A Ne I K-shell channel with n_ejected=3: the two extra (Auger) electrons share the energy that the
    # ion does not keep. Its matrix is that of a single-ionisation channel with the same potential,
    # minus the source term of the extra electrons in the rows below the energy of one extra electron.
    ionpot_ev = 870.0
    n_ion = 1e8
    nist = pynonthermal.collion.get_nist_ionisation_energies_ev()
    extra_ev = ionpot_ev - sum(nist[(10, ion_stage)] for ion_stage in (1, 2, 3))

    fills = {}
    for n_ejected in (1, 3):
        with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=400) as sf:
            sf.add_ionisation_channel(10, 1, n_ion, ionpot_ev, lotz_like_xs(ionpot_ev, 1e-19), n_ejected=n_ejected)
            fills[n_ejected] = sf.sfmatrix.copy()
            channel = sf._ionisation_channels[(10, 1)][0]
            if n_ejected == 3:
                assert math.isclose(channel.extra_electron_energy_ev, extra_ev, rel_tol=1e-12)
                engrid = sf.engrid
                deltaen = sf.deltaen
                xs_grid = channel.xs_grid

    expected = np.zeros_like(fills[1])
    for i, en_ev in enumerate(engrid):
        if en_ev < extra_ev / 2:
            for j in range(len(engrid)):
                if engrid[j] >= ionpot_ev:
                    expected[i, j] = -n_ion * xs_grid[j] * deltaen * 2
    assert np.any(expected != 0.0)
    assert np.allclose(fills[3] - fills[1], expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max())


def test_auger_channel_energy_accounting() -> None:
    # The extra electrons of an inner-shell ionisation return the energy that the ion does not keep. So,
    # compared with n_ejected=1, the ionisation fraction falls and the heating fraction increases. The
    # energy fractions sum to one in both cases.
    fractions = {}
    for n_ejected in (1, 2):
        with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=1000) as sf:
            sf.add_ionisation_channel(10, 1, 1e8, 21.6, lotz_like_xs(21.6, 3e-16), "valence")
            sf.add_ionisation_channel(10, 1, 1e8, 870.0, lotz_like_xs(870.0, 1e-17), "K", n_ejected=n_ejected)
            # the solver holds only neutral ions, so the free electrons come from the override
            sf.override_n_e(1e6)
            sf.solve(deposition_ev_per_s_per_cm3=1e8)
            fractions[n_ejected] = (sf.get_frac_ionisation_tot(), sf.get_frac_heating(), sf.get_frac_sum())

    assert fractions[2][0] < fractions[1][0]
    assert fractions[2][1] > fractions[1][1]
    for n_ejected in (1, 2):
        assert math.isclose(fractions[n_ejected][2], 1.0, abs_tol=0.01)


def test_extra_electrons_below_emin_are_heating() -> None:
    # An extra electron at or below emin_ev never enters the grid, so its energy is heating. The matrix does
    # not change, so the energy moves from the ionisation fraction to the heating fraction and nothing more.
    # The grid spacing near the thresholds sets the error of the sum, so the grid is fine.
    nist = pynonthermal.collion.get_nist_ionisation_energies_ev()
    ionpot_ev = nist[(10, 1)] + nist[(10, 2)] + 5.0
    results = {}
    for extra_ev in (0.0, 5.0):
        with pynonthermal.SpencerFanoSolver(emin_ev=20, emax_ev=3000, npts=3000) as sf:
            sf.add_ionisation_channel(10, 1, 1e8, 21.6, lotz_like_xs(21.6, 3e-16), "valence")
            sf.add_ionisation_channel(
                10, 1, 1e8, ionpot_ev, lotz_like_xs(ionpot_ev, 3e-17), "double", n_ejected=2,
                extra_electron_energy_ev=extra_ev,
            )  # fmt: skip
            sf.override_n_e(1e6)
            sf.solve(deposition_ev_per_s_per_cm3=1e8)
            results[extra_ev] = (sf.sfmatrix.copy(), sf.get_frac_ionisation_tot(), sf.get_frac_heating())

    assert np.array_equal(results[0.0][0], results[5.0][0])
    ionisation_moved = results[0.0][1] - results[5.0][1]
    assert ionisation_moved > 0.0
    assert math.isclose(results[5.0][2] - results[0.0][2], ionisation_moved, rel_tol=1e-9)

    # the default comes from energy conservation, and it is the same 5 eV
    with pynonthermal.SpencerFanoSolver(emin_ev=20, emax_ev=3000, npts=400) as sf:
        sf.add_ionisation_channel(10, 1, 1e8, ionpot_ev, lotz_like_xs(ionpot_ev, 3e-17), n_ejected=2)
        assert math.isclose(sf._ionisation_channels[(10, 1)][0].extra_electron_energy_ev, 5.0, rel_tol=1e-9)


def test_strontium_multiple_ionisation_balance() -> None:
    # a heavy element with Lotz shells, as in a kilonova model: the double ionisation of Sr I and Sr II
    # moves the element to higher stages
    n_sr = 1e6

    def solve_strontium(double_scale_cm2: float) -> dict[int, float]:
        with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=400) as sf:
            with pytest.warns(pynonthermal.LotzApproximationWarning, match="Z=38"):
                sf.add_element(38, n_sr, recomb_ratecoeffs=STRONTIUM_ALPHAS)
            if double_scale_cm2 > 0.0:
                nist = pynonthermal.collion.get_nist_ionisation_energies_ev()
                for ion_stage in (1, 2):
                    ionpot_ev = nist[(38, ion_stage)] + nist[(38, ion_stage + 1)]
                    sf.add_ionisation_channel(
                        38, ion_stage, None, ionpot_ev, lotz_like_xs(ionpot_ev, double_scale_cm2), n_ejected=2
                    )
            sf.solve(deposition_ev_per_s_per_cm3=1.0, balance_tol=1e-6)
            assert math.isclose(sf.get_frac_sum(), 1.0, abs_tol=0.02)
            return sf.get_ion_fractions(38)

    fractions = {scale: solve_strontium(scale) for scale in (0.0, 1e-17, 1e-16)}
    mean_charge = {
        scale: sum((ion_stage - 1) * frac for ion_stage, frac in fracs.items()) for scale, fracs in fractions.items()
    }
    assert mean_charge[0.0] < mean_charge[1e-17] < mean_charge[1e-16]


def test_multiple_ionisation_validation() -> None:
    xs = lotz_like_xs(100.0, 1e-18)
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf:
        for n_ejected in (0, 1.5, True):
            with pytest.raises(ValueError, match="n_ejected must be an integer"):
                sf.add_ionisation_channel(8, 1, 1e8, 100.0, xs, n_ejected=n_ejected)  # ty: ignore[invalid-argument-type]
        with pytest.raises(ValueError, match="cannot remove"):
            sf.add_ionisation_channel(2, 1, 1e8, 100.0, xs, n_ejected=3)
        with pytest.raises(ValueError, match="less than the sum of the ground-state"):
            sf.add_ionisation_channel(8, 1, 1e8, 40.0, lotz_like_xs(40.0, 1e-18), n_ejected=2)
        with pytest.raises(ValueError, match="no extra electrons"):
            sf.add_ionisation_channel(8, 1, 1e8, 100.0, xs, extra_electron_energy_ev=10.0)
        # the ion must keep the sum of the NIST potentials (48.7 eV for O I to O III), also with a value
        # from the caller
        with pytest.raises(ValueError, match="Set extra_electron_energy_ev to at most"):
            sf.add_ionisation_channel(8, 1, 1e8, 100.0, xs, n_ejected=2, extra_electron_energy_ev=60.0)
        with pytest.raises(ValueError, match="less than the sum of the ground-state"):
            sf.add_ionisation_channel(
                8, 1, 1e8, 40.0, lotz_like_xs(40.0, 1e-18), n_ejected=2, extra_electron_energy_ev=0.0
            )
        # a rejected call leaves the solver unchanged
        assert not sf._ionisation_channels
        assert not sf.ionpopdict

        # a value from the caller that leaves the ion at least the NIST sum is kept as it is
        sf.add_ionisation_channel(8, 1, 1e8, 100.0, xs, "auger", n_ejected=2, extra_electron_energy_ev=30.0)
        assert sf._ionisation_channels[(8, 1)][0].extra_electron_energy_ev == 30.0

    # an ion that the NIST data does not hold needs the value from the caller
    nist = pynonthermal.collion.get_nist_ionisation_energies_ev()
    Z_missing, stage_missing = next(
        (Z, ion_stage) for Z in range(1, 111) for ion_stage in range(1, Z) if (Z, ion_stage) not in nist
    )
    ionpot_ev = 2000.0
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf:
        with pytest.raises(ValueError, match="Give extra_electron_energy_ev"):
            sf.add_ionisation_channel(
                Z_missing, stage_missing, 1e8, ionpot_ev, lotz_like_xs(ionpot_ev, 1e-20), n_ejected=2
            )
        # without the NIST data, the ion must still keep a positive energy
        with pytest.raises(ValueError, match="less than ionpot_ev"):
            sf.add_ionisation_channel(
                Z_missing, stage_missing, 1e8, ionpot_ev, lotz_like_xs(ionpot_ev, 1e-20), n_ejected=2,
                extra_electron_energy_ev=ionpot_ev,
            )  # fmt: skip
        sf.add_ionisation_channel(
            Z_missing, stage_missing, 1e8, ionpot_ev, lotz_like_xs(ionpot_ev, 1e-20), n_ejected=2,
            extra_electron_energy_ev=100.0,
        )  # fmt: skip
        assert sf._ionisation_channels[(Z_missing, stage_missing)][0].extra_electron_energy_ev == 100.0

    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf:
        sf.add_element(8, 1e8, recomb_ratecoeffs=OXYGEN_ALPHAS)
        # O I to O IV jumps past the top stage O III of the chain
        with pytest.raises(ValueError, match="top stage of the ionisation balance is 3"):
            sf.add_ionisation_channel(8, 1, None, 110.0, lotz_like_xs(110.0, 1e-18), n_ejected=3)
        # the top stage is a sink, so a multiple ionisation out of it is permitted, as a single one is
        sf.add_ionisation_channel(8, 3, None, 140.0, lotz_like_xs(140.0, 1e-18), "double", n_ejected=2)
