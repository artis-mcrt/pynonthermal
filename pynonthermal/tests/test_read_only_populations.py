"""Check that the public view of the ion populations cannot change the model."""

import math

import pytest

import pynonthermal


def _get_solver(n_ion: float) -> pynonthermal.SpencerFanoSolver:
    sf = pynonthermal.SpencerFanoSolver(emin_ev=0.1, emax_ev=1000, npts=256)
    sf.add_ionisation(8, 2, n_ion)
    sf.solve(deposition_ev_per_s_per_cm3=1e5)
    return sf


def test_public_populations_are_read_only() -> None:
    # A write of twice the population changed the reported density and left the old population in the
    # matrix. The energy sum of the next solve was then 1.06724.
    sf = _get_solver(1e8)
    with pytest.raises(TypeError):
        sf.ionpopdict[8, 2] = 2e8  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        del sf.ionpopdict[8, 2]  # ty: ignore[not-subscriptable]
    with pytest.raises(AttributeError):
        sf.ionpopdict = {(8, 2): 2e8}  # ty: ignore[invalid-assignment]

    assert sf.ionpopdict == {(8, 2): 1e8}
    assert dict(sf.ionpopdict) == {(8, 2): 1e8}
    assert sf.get_n_ion_tot() == 1e8
    sf.solve(deposition_ev_per_s_per_cm3=1e5)
    assert math.isclose(sf.get_frac_sum(), 1.0, rel_tol=1e-12)


def test_new_solver_gives_the_result_for_a_new_population() -> None:
    sf_double = _get_solver(2e8)
    assert sf_double.ionpopdict[8, 2] == 2e8
    assert math.isclose(sf_double.get_frac_sum(), 1.0, rel_tol=1e-12)
    assert not math.isclose(sf_double.get_frac_heating(), _get_solver(1e8).get_frac_heating(), rel_tol=1e-3)


def test_view_follows_the_populations_of_a_balance() -> None:
    sf = pynonthermal.SpencerFanoSolver(emin_ev=0.1, emax_ev=1000, npts=256)
    sf.add_element(Z=8, n_elem=1e8, recomb_ratecoeffs={2: 1e-12, 3: 3e-12, 4: 5e-12, 5: 1e-11})
    view = sf.ionpopdict
    sf.solve(deposition_ev_per_s_per_cm3=1e3)
    assert view is sf.ionpopdict
    assert math.isclose(sum(view.values()), 1e8, rel_tol=1e-12)
    assert view[8, 2] == sf.get_ion_fractions(8)[2] * 1e8
