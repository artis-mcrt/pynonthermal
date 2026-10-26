"""Tests for the defects that a review of the whole library found.

Each test gives the input that failed before the fix and checks the behaviour after it.
"""

import pickle
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import pynonthermal


def _oxygen_solver(**kwargs: object) -> pynonthermal.SpencerFanoSolver:
    sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300, **kwargs)  # ty: ignore[invalid-argument-type]
    sf.add_element(8, ion_densities={1: 1e8, 2: 1e6})
    return sf


def _oxygen_level_table() -> pl.DataFrame:
    import artistools as at  # noqa: PLC0415

    table = at.atomic.get_levels(
        pynonthermal.DATADIR / "artis_files",
        get_transitions=True,
        derived_transitions_columns=list(pynonthermal.spencerfano.DERIVED_TRANSITIONS_COLUMNS),
    )
    return table.filter(pl.col("Z") == 8)


def _with_transitions(table: pl.DataFrame, ion_stage: int, transitions: pl.DataFrame) -> pl.DataFrame:
    # the table of one ion with its transitions replaced
    ion = table.filter(pl.col("ion_stage") == ion_stage)
    return ion.with_columns(pl.Series("transitions", [transitions], dtype=pl.Object))


def test_auger_advice_ends_for_a_huge_threshold() -> None:
    # the rounding loop of the advice did not end above 2^44 eV, where a step of 0.001 eV is lost
    auger_ev, message = pynonthermal.collion._get_auger_electron_energy_ev(26, 1, 1e14, 2, 1e14 - 1.0, 0.0)
    assert auger_ev == 1e14 - 1.0
    assert message is not None
    assert "at most" in message


def test_failed_deprecated_override_keeps_the_earlier_density() -> None:
    sf = _oxygen_solver()
    sf.solve(1e8)
    n_e_before = sf.get_n_e()
    with pytest.warns(DeprecationWarning, match="override_n_e"), pytest.raises(ValueError, match="not valid"):
        sf.solve(1e8, override_n_e=1e30)
    # the failed call kept the density of the plasma and the last solution
    assert sf.get_n_e() == n_e_before
    assert sf.get_frac_heating() > 0.0


def test_failed_solve_keeps_the_last_solution() -> None:
    sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300)
    sf.add_element(8, n_elem=5e19, recomb_ratecoeffs={2: 3e-13, 3: 3e-12})
    sf.solve(1e3)
    frac_heating = sf.get_frac_heating()
    populations = dict(sf.ionpopdict)
    yvec = sf.yvec.copy()
    matrix = sf.sfmatrix.copy()
    # the balance drives n_e above the limit of the loss function
    with pytest.raises(ValueError, match="not valid"):
        sf.solve(1e40)
    assert sf.get_frac_heating() == pytest.approx(frac_heating, rel=1e-12)
    assert sf.ionpopdict == pytest.approx(populations, rel=1e-12)
    np.testing.assert_allclose(sf.yvec, yvec, rtol=1e-12)
    np.testing.assert_allclose(sf.sfmatrix, matrix, rtol=1e-9, atol=1e-12 * np.abs(matrix).max())
    # the solver is still solved, so it accepts no more ions
    with pytest.raises(RuntimeError, match="after solving"):
        sf.add_ionisation(26, 2, 1e5)


def test_pickled_copy_does_not_replace_a_custom_table() -> None:
    sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300)
    sf.set_temperature(6000)
    sf.set_atomic_data(adata_polars=_oxygen_level_table())
    sf.add_element(8, ion_densities={1: 1e8, 2: 1e6}, excitation=True)
    sf.solve(1e8)
    copy = pickle.loads(pickle.dumps(sf))  # noqa: S301
    with pytest.warns(DeprecationWarning, match="override_n_e"), pytest.raises(ValueError, match="not valid"):
        copy.solve(1e8, override_n_e=1e30)
    # the failed solve keeps the solution, so the copy accepts no more ions
    with pytest.raises(RuntimeError, match="after solving"):
        copy.add_ion_excitation(26, 2, 1e4)
    # a copy without a solution must not read the internal database in place of the custom table
    copy._solved = False
    with pytest.raises(ValueError, match="pickle cannot keep the custom"):
        copy.add_ion_excitation(26, 2, 1e4)


def test_nulls_and_missing_levels_of_a_custom_table() -> None:
    table = _oxygen_level_table()
    transitions = table.filter(pl.col("ion_stage") == 2)["transitions"].item().lazy().collect()
    levels = table.filter(pl.col("ion_stage") == 2)["levels"].item()

    def solver_with(ion_table: pl.DataFrame) -> pynonthermal.SpencerFanoSolver:
        sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300)
        sf.set_temperature(6000)
        sf.set_atomic_data(adata_polars=ion_table)
        return sf

    # a null collision strength reached the cross section as None, and a null forbidden flag
    # dropped the permitted transitions with no message
    null_collstr = transitions.with_columns(
        pl.when(pl.col("forbidden") == 0).then(None).otherwise(pl.col("collstr")).alias("collstr")
    )
    with pytest.raises(ValueError, match=r"null values in the columns \['collstr'\]"):
        solver_with(_with_transitions(table, 2, null_collstr)).add_ion_excitation(8, 2, 1e6)
    null_forbidden = transitions.with_columns(pl.lit(None, dtype=pl.Int64).alias("forbidden"))
    with pytest.raises(ValueError, match=r"null values in the columns \['forbidden'\]"):
        solver_with(_with_transitions(table, 2, null_forbidden)).add_ion_excitation(8, 2, 1e6)

    # a lower level that the levels table does not hold gave float(None)
    ion = table.filter(pl.col("ion_stage") == 2).with_columns(
        pl.Series("levels", [levels.filter(pl.col("levelindex") != 1)], dtype=pl.Object)
    )
    with pytest.raises(ValueError, match="not in the levels table"):
        solver_with(ion).add_ion_excitation(8, 2, 1e6)


def test_bools_are_not_numbers() -> None:
    sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300)
    with pytest.raises(ValueError, match="is True"):
        sf.add_element(26, ion_densities={1: True, 2: 1e5})
    with pytest.raises(ValueError, match="is True"):
        sf.add_element(26, n_elem=True, ion_fractions={1: 0.5, 2: 0.5})
    with pytest.raises(ValueError, match="is True"):
        sf.set_temperature(True)
    with pytest.raises(ValueError, match="is True"):
        sf.override_n_e(True)
    sf.add_element(8, ion_densities={1: 1e8, 2: 1e6})
    with pytest.raises(ValueError, match="is True"):
        sf.solve(True)
    assert not sf.ionpopdict.keys() & {(26, 1), (26, 2)}


def test_grid_channel_copies_the_grid() -> None:
    grid = np.linspace(1.0, 100.0, 100)
    channel = pynonthermal.collion.IonisationChannel.from_xs_grid(
        grid, 8, 1, 20.0, np.where(grid > 20.0, 1e-18, 0.0), "k"
    )
    before = channel.xs(np.array([30.0]))
    grid *= 2.0
    np.testing.assert_array_equal(channel.xs(np.array([30.0])), before)


def test_shell_occupancies_check_every_ion() -> None:
    pynonthermal.axelrod.get_shell_occupancies(26, 2)
    # a float key equal to a cached integer key got the cached array with no check
    with pytest.raises(ValueError, match="must be an integer"):
        pynonthermal.axelrod.get_shell_occupancies(26.0, 2)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="up to Z=108"):
        pynonthermal.axelrod.get_workfn_ev(109, 1, ionpot_ev=5.0, Zbar=50.0)


def test_zbar_of_a_generator() -> None:
    populations = {(26, 1): 1.0, (26, 2): 2.0}
    assert pynonthermal.base.get_Zbar((ion for ion in populations), populations) == pytest.approx(26.0)


def test_zero_population_is_registered_by_every_method() -> None:
    sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300)
    sf.add_ionisation(8, 1, 0.0)
    assert sf.ionpopdict[(8, 1)] == 0.0
    # the registered zero lets a later call add the channels of the empty stage
    sf.add_ionisation(8, 1, None)
    assert sf._ionisation_channels[(8, 1)]
    sf.add_ionisation_channel(8, 2, 0.0, 40.0, lambda en_ev: np.where(en_ev > 40.0, 1e-18, 0.0))
    assert sf.ionpopdict[(8, 2)] == 0.0
    assert len(sf._ionisation_channels[(8, 2)]) == 1
    assert not sf.sfmatrix.any()


def test_stage_matrices_give_the_per_channel_result(monkeypatch: pytest.MonkeyPatch) -> None:
    def solved(budget: int) -> pynonthermal.SpencerFanoSolver:
        monkeypatch.setattr(pynonthermal.spencerfano, "BALANCE_STAGE_MATRIX_MAX_BYTES", budget)
        sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300)
        sf.add_element(8, n_elem=1e9, recomb_ratecoeffs={2: 3e-13, 3: 3e-12, 4: 1e-11})
        sf.solve(1e8)
        return sf

    per_channel = solved(0)
    unit_matrices = solved(2**30)
    assert not per_channel._stage_matrices
    assert len(unit_matrices._stage_matrices) == 4
    np.testing.assert_allclose(unit_matrices.yvec, per_channel.yvec, rtol=1e-12)
    assert unit_matrices.ionpopdict == pytest.approx(per_channel.ionpopdict, rel=1e-12)
    # pickle leaves the large unit matrices out
    assert not pickle.loads(pickle.dumps(unit_matrices))._stage_matrices  # noqa: S301


def test_import_loads_neither_artistools_nor_matplotlib() -> None:
    # the annotations of the plot methods still resolve at runtime, through a name that imports
    # matplotlib.axes only when the annotation is evaluated
    code = (
        "import sys, typing, pynonthermal; print('artistools' in sys.modules, 'matplotlib' in sys.modules);"
        " print(typing.get_type_hints(pynonthermal.SpencerFanoSolver.plot_channels)['axis'])"
    )
    repo = Path(pynonthermal.__file__).parent.parent
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, cwd=repo)  # noqa: S603
    assert result.stdout.split() == ["False", "False", "matplotlib.axes._axes.Axes", "|", "None"]


def test_inputs_with_a_bad_type_fail_with_a_named_message() -> None:
    with pytest.raises(TypeError, match="npts must be an integer"):
        pynonthermal.SpencerFanoSolver(npts=100.0)  # ty: ignore[invalid-argument-type]
    sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300)
    with pytest.raises(TypeError, match="Call collect"):
        sf.set_atomic_data(adata_polars=_oxygen_level_table().lazy())  # ty: ignore[invalid-argument-type]
    xs = np.where(sf.engrid > 40.0, 1e-18, 0.0)
    with pytest.raises(TypeError, match="channelkey must be hashable"):
        sf.add_ionisation_channel(8, 2, 1e6, 40.0, xs, [1])
    with pytest.raises(TypeError, match="transitionkey must be hashable"):
        sf.add_excitation(8, 2, 1e6, xs, 20.0, [1])
    # an integer transition energy works, and python -O no longer changes the check
    row = {"collstr": -1.0, "epsilon_trans_ev": 5, "lower_g": 1.0, "upper_g": 3.0, "A": 1e8, "forbidden": False}
    assert pynonthermal.excitation.get_xs_excitation_vector(sf.engrid, row).max() > 0.0
    # a bool is not an energy, and float(True) gave a transition of 1 eV
    with pytest.raises(ValueError, match="epsilon_trans_ev of a transition must be greater than zero"):
        pynonthermal.excitation.get_xs_excitation_vector(sf.engrid, row | {"epsilon_trans_ev": True})


def test_balanced_stage_without_transitions_has_no_entry() -> None:
    sf = pynonthermal.SpencerFanoSolver(emin_ev=0.5, emax_ev=0.6, npts=50)
    sf.set_temperature(6000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sf.add_element(8, n_elem=1e8, recomb_ratecoeffs={2: 3e-13}, excitation=True)
    assert not sf.excitationlists


def test_messages_name_the_cause() -> None:
    sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300)
    sf.add_element(8, ion_densities={1: 1e8, 2: 0.0})
    with pytest.raises(ValueError, match="every ionised stage that the solver holds has a population of zero"):
        sf.solve(1e3)
    sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300)
    sf.set_temperature(6000)
    sf.set_atomic_data(adata_polars=_oxygen_level_table())
    with pytest.raises(ValueError, match="in the level/transition table of set_atomic_data"):
        sf.add_ion_excitation(26, 2, 1e4)
