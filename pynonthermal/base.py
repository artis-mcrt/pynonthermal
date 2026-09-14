#!/usr/bin/env python3
import math
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pynonthermal.constants import CLIGHT
from pynonthermal.constants import EV
from pynonthermal.constants import H
from pynonthermal.constants import ME
from pynonthermal.constants import QE

DATADIR = Path(__file__).absolute().parent / "data"

# ion_stage is one more than the charge. A caller who uses the charge gives a stage of 0 for a
# neutral atom, so the messages that reject a stage below 1 give this hint.
ION_STAGE_HINT: str = "ion_stage is one more than the charge, so a neutral atom is ion_stage 1"


def _is_integer(value: object) -> bool:
    # a Python or numpy integer, but not a bool, which is an int in Python
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _check_ion(Z: int, ion_stage: int, needs_electron: bool = False) -> tuple[int, int]:
    """Check an ion identity, and get Z and ion_stage as Python integers.

    ion_stage is one more than the charge, so it runs from 1 (neutral) to Z + 1 (the bare
    nucleus). With needs_electron, the bare nucleus is rejected. An ionisation or an excitation
    needs a bound electron.
    """
    if not _is_integer(Z) or Z < 1:
        msg = f"Z must be an integer of at least 1 but is {Z!r}"
        raise ValueError(msg)
    Z = int(Z)
    if not _is_integer(ion_stage) or not 1 <= ion_stage <= Z + 1:
        msg = f"ion_stage of Z={Z} must be an integer between 1 and {Z + 1} but is {ion_stage!r}"
        if _is_integer(ion_stage) and ion_stage < 1:
            msg = f"{msg}. {ION_STAGE_HINT}"
        raise ValueError(msg)
    ion_stage = int(ion_stage)
    if needs_electron and ion_stage == Z + 1:
        msg = f"Z={Z} ion_stage {ion_stage} is a bare nucleus, which has no bound electron"
        raise ValueError(msg)
    return Z, ion_stage


# A cross section sigma(E) [cm^2] at an array of electron energies [eV].
# An IonisationChannel holds one of these, because calculate_N_e() evaluates an ionisation cross
# section between the points of the solver energy grid. A channel built from an array on that grid
# interpolates the array there.
type CrossSectionFunc = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


def get_xs_on_grid(
    xs: npt.NDArray[np.float64] | CrossSectionFunc, engrid: npt.NDArray[np.float64], name: str
) -> npt.NDArray[np.float64]:
    """Get the cross sections [cm^2] of xs on the energy grid engrid [eV], and check them.

    xs is either an array on engrid or a function of an array of energies [eV]. name identifies
    xs in an error message. The result is a new read-only array. A later write to the array of
    the caller, or to a buffer that the function of the caller keeps, cannot change it.
    """
    # isinstance first, because callable() alone leaves the type checkers with an array that
    # might also be callable
    if isinstance(xs, np.ndarray):
        arr_xs = xs
    elif callable(xs):
        arr_xs = xs(engrid)
    else:
        msg = f"{name} must be a numpy array on the energy grid, but it is a {type(xs).__name__}"
        raise TypeError(msg)

    # np.array copies, so the checks below hold for the lifetime of the result
    xs_grid = np.array(arr_xs, dtype=np.float64)

    if xs_grid.shape != engrid.shape:
        msg = (
            f"{name} must give one value for each of the {len(engrid)} energies of engrid,"
            f" but it gave an array of shape {xs_grid.shape}"
        )
        raise ValueError(msg)

    if not np.isfinite(xs_grid).all():
        msg = f"{name} must be finite at every energy of the grid"
        raise ValueError(msg)

    if not (xs_grid >= 0.0).all():
        msg = f"{name} must be non-negative but its lowest value is {xs_grid.min()} cm^2"
        raise ValueError(msg)

    # handed on to the solver, so a write must raise at the mutation site
    xs_grid.flags.writeable = False

    return xs_grid


def get_betasq(en_ev: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64]:
    """Get (v/c)^2 for an electron of kinetic energy en_ev [eV], relativistically.

    The classical 2 * en_ev * EV / (ME * CLIGHT**2) reaches one at 255 keV and is already 5 per cent
    high at 16 keV, which is within the range of emax_ev this package is used over.
    """
    gamma = np.asarray(en_ev, dtype=np.float64) * EV / (ME * CLIGHT**2) + 1.0

    return 1.0 - 1.0 / gamma**2


def electronlossfunction(energy_ev: float, n_e_cgs: float) -> float:
    # the loss function L(E) in the Spencer-Fano equation: the energy loss rate - dE / dX
    # [eV / cm] of a non-thermal electron to the free thermal electrons, following Kozma &
    # Fransson 1992: their equation 1 above 14 eV and equation 2 below it, with the plasma
    # energy zeta_e of their equation 3 in the high-energy Coulomb logarithm
    # returns a positive number
    # the chained comparisons also reject nan, for which every comparison is False
    if not 0.0 < n_e_cgs < math.inf:
        # the plasma frequency would be zero, making the Coulomb logarithm infinite
        msg = f"the free-electron loss function requires a positive finite free electron density but n_e is {n_e_cgs}"
        raise ValueError(msg)
    if not 0.0 < energy_ev < math.inf:
        msg = f"the free-electron loss function requires a positive finite energy but energy_ev is {energy_ev}"
        raise ValueError(msg)

    n_e = n_e_cgs
    energy = energy_ev * EV  # convert eV to erg

    omegap = math.sqrt(4 * math.pi * n_e_cgs * QE**2 / ME)
    zetae = H * omegap / 2 / math.pi

    # Kozma & Fransson (1992) give separate Coulomb logarithms above and below 14 eV and the two do not
    # meet there: the branch below is lower by ln(hbar * v / (exp(eulergamma) * QE^2)), which is a step of
    # about 2-3 per cent in the loss rate for the free electron densities of interest. The step is part of
    # the published prescription (and of the ARTIS implementation this follows), so it is left in place
    # rather than smoothed with a formula that is nobody's.
    if energy_ev > 14:
        coulomblog_arg = 2 * energy / zetae
    else:
        v = math.sqrt(2 * energy / ME)  # velocity in cm/s (energy is in erg and ME in g, so cgs)
        # Kozma & Fransson (1992) eq. 2 describes the gamma in this Coulomb logarithm as "Euler's
        # constant (Schunk & Hays 1971)", but Schunk & Hays (1971, p. 114) define it by "ln gamma is
        # Euler's constant", i.e. gamma = exp(0.5772) = 1.781 rather than 0.5772 itself.
        exp_eulergamma = math.exp(np.euler_gamma)
        coulomblog_arg = ME * pow(v, 3) / (exp_eulergamma * pow(QE, 2) * omegap)

    # Both branches lose their meaning once the plasma is dense enough that the Coulomb logarithm
    # reaches zero, which would put a non-positive loss rate on the Spencer-Fano matrix diagonal. The
    # low-energy branch fails first, at n_e ~ 7e19 for a 1 eV electron against ~6e23 for the other.
    # The limit falls as the cube of the energy, so it is n_e ~ 7e16 at the 0.1 eV bottom of the
    # default grid of SpencerFanoSolver.
    if coulomblog_arg <= 1.0:
        msg = (
            f"the free-electron loss function is not valid at {energy_ev} eV for n_e = {n_e_cgs} cm^-3:"
            f" the plasma energy hbar*omega_p is {zetae / EV:.3g} eV, which is too close to the electron"
            " energy for the Coulomb logarithm to be positive."
        )
        raise ValueError(msg)

    lossfunc = n_e * 2 * math.pi * QE**4 / energy * math.log(coulomblog_arg)

    # lossfunc is now [erg / cm]
    return lossfunc / EV  # return as [eV / cm]


def get_n_tot(ions: Sequence[tuple[int, int]], ionpopdict: dict[tuple[int, int], float]) -> float:
    # total number density of all nuclei [cm^-3]
    n_tot = 0.0
    for Z, ion_stage in ions:
        n_tot += ionpopdict[(Z, ion_stage)]
    return n_tot


def get_Zbar(ions: Sequence[tuple[int, int]], ionpopdict: dict[tuple[int, int], float]) -> float:
    # number density-weighted average atomic number
    # i.e. protons per nucleus
    Zbar = 0.0
    n_tot = get_n_tot(ions, ionpopdict)
    for Z, ion_stage in ions:
        n_ion = ionpopdict[(Z, ion_stage)]
        Zbar += Z * n_ion / n_tot

    return Zbar


def _get_energyindex(en_ev: float, engrid: npt.NDArray[np.float64], round_up: bool) -> int:
    # index of the energy bin holding en_ev, clamped into the grid at both ends. A binary search
    # on the grid is exact at the grid points. A division by the grid spacing gave the next bin
    # for most of them.
    if math.isnan(en_ev):
        msg = "the energy must not be nan"
        raise ValueError(msg)
    if round_up:
        index = int(np.searchsorted(engrid, en_ev, side="left"))
    else:
        index = int(np.searchsorted(engrid, en_ev, side="right")) - 1

    return 0 if index < 0 else min(index, len(engrid) - 1)


def get_energyindex_lteq(en_ev: float, engrid: npt.NDArray[np.float64]) -> int:
    """Get the index of the energy bin whose lower boundary is less than or equal to en_ev."""
    return _get_energyindex(en_ev, engrid, round_up=False)


def get_energyindex_gteq(en_ev: float, engrid: npt.NDArray[np.float64]) -> int:
    """Get the index of the energy bin whose lower boundary is greater than or equal to en_ev."""
    return _get_energyindex(en_ev, engrid, round_up=True)
