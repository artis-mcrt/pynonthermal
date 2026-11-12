"""Check cross sections between the energy grid points."""

import numpy as np
import numpy.typing as npt
import pytest

import pynonthermal


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -1e-17])
def test_invalid_cross_section_between_grid_points(invalid_value: float) -> None:
    def xs(energy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        u = energy / 14.0
        values = np.where(energy > 14.0, 1e-17 * np.log(np.maximum(u, 1.0)) / u, 0.0)
        return np.where((energy > 14.0) & (energy < 14.2), invalid_value, values)

    sf = pynonthermal.SpencerFanoSolver(npts=100, emin_ev=0.1, emax_ev=500)
    sf.add_ionisation_channel(1, 1, 1e7, 14.0, xs, channelkey="custom")
    sf.override_n_e(1e7)
    sf.solve(1e8)

    with pytest.raises(ValueError, match=r"channel custom.*finite and non-negative.*14\..* eV"):
        sf.get_frac_heating()
    assert sf._frac_heating is None
    assert np.isfinite(sf.yvec).all()


def test_wrong_cross_section_shape_between_grid_points() -> None:
    def xs(energy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        values = np.where(energy > 14.0, 1e-17, 0.0)
        if ((energy > 14.0) & (energy < 14.2)).any():
            return values[:-1]
        return values

    sf = pynonthermal.SpencerFanoSolver(npts=100, emin_ev=0.1, emax_ev=500)
    sf.add_ionisation_channel(1, 1, 1e7, 14.0, xs, channelkey="custom")
    sf.override_n_e(1e7)
    sf.solve(1e8)

    with pytest.raises(ValueError, match="channel custom must have shape"):
        sf.calculate_N_e(0.1)
