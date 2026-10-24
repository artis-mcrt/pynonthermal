"""Collisional excitation of bound-bound transitions.

The comments and the docstrings of this module cite these publications by their labels:

- KF92: Kozma & Fransson 1992, ApJ, 390, 602-621, doi:10.1086/171311
- LHD12: Li, Hillier & Dessart 2012, MNRAS, 426, 1671-1686, doi:10.1111/j.1365-2966.2012.21198.x
- M72: Mewe 1972, A&A, 20, 215-221, ADS 1972A&A....20..215M (the journal gave no DOI)
- VR62: van Regemorter 1962, ApJ, 136, 906-915, doi:10.1086/147445
- S20: Shingles et al. 2020, MNRAS, 492, 2029-2043, doi:10.1093/mnras/stz3412
"""

import math
import typing as t
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

import pynonthermal
from pynonthermal.constants import CLIGHT
from pynonthermal.constants import EV
from pynonthermal.constants import H
from pynonthermal.constants import H_ionpot
from pynonthermal.constants import ME
from pynonthermal.constants import QE


@dataclass(frozen=True, slots=True, eq=False)
class ExcitationTransition:
    """One bound-bound excitation that the solver holds, as SpencerFanoSolver.add_excitation() records it.

    The cross section is an array on the energy grid of the solver, because each reader needs it
    only there. The readers are the matrix fill, the excitation fraction of KF92 equation 9, and
    the one interpolated point that calculate_N_e() takes at energy_ev + epsilon_trans_ev.
    """

    levelnumberdensity: float
    """The population density of the lower level [cm^-3]."""

    xs_vec: npt.NDArray[np.float64]
    """The cross section [cm^2] at every energy of SpencerFanoSolver.engrid."""

    epsilon_trans_ev: float
    """The transition energy [eV]."""


def get_xs_excitation_vector(
    engrid: npt.NDArray[np.float64], row: dict[str, t.Any], use_collstrengths: bool = True
) -> npt.NDArray[np.float64]:
    """Get an array that holds the excitation cross section in cm^2 at every energy of engrid (eV).

    This is the sigma of the excitation term in the degradation equation (KF92 equation 7). A
    transition with a collision strength gets the cross section of LHD12 equation 11. A permitted
    E1 transition without one gets the VR62 approximation from its oscillator strength. Its g-bar
    factor is g_bar = 0.15 + 0.28 ln(U), from the A and D ln(U) terms of the fit in M72 equation 5.
    S20 (section 2.5) calls these terms the "first two terms" of the formula.
    """
    A_naught_squared = 2.800285203e-17  # Bohr radius squared in cm^2
    npts = len(engrid)
    xs_excitation_vec = np.empty(npts)

    coll_str = row["collstr"]
    # a Python float, so that an integer or a numpy energy of a custom table works. A bare assert
    # rejected an integer with no message, and python -O removed it.
    epsilon_trans_ev = float(row["epsilon_trans_ev"])
    epsilon_trans = epsilon_trans_ev * EV

    if epsilon_trans_ev > engrid[-1]:
        # no electron on the grid has enough energy to drive this transition. Without this check,
        # get_energyindex_gteq clamps to the last grid point and leaves a spurious cross section
        # there, which for the E1 branch below comes out negative (g_bar < 0 for U < 0.585).
        return np.zeros(npts)

    startindex = pynonthermal.get_energyindex_gteq(en_ev=epsilon_trans_ev, engrid=engrid)
    xs_excitation_vec[:startindex] = 0.0

    if coll_str >= 0 and use_collstrengths:
        # collision strength is available, so use it
        # LHD12 equation 11: sigma = pi * a_0^2 * (H_ionpot / E) * coll_str / lower_g,
        # with k_i^2 = E / H_ionpot in units of the inverse Bohr radius squared
        constantfactor = H_ionpot / row["lower_g"] * coll_str * math.pi * A_naught_squared

        xs_excitation_vec[startindex:] = constantfactor / (engrid[startindex:] * EV)

    elif not row["forbidden"]:
        nu_trans = epsilon_trans / H
        g = row["upper_g"] / row["lower_g"]
        fij = g * ME * pow(CLIGHT, 3) / (8 * pow(QE * nu_trans * math.pi, 2)) * row["A"]
        # permitted E1 electric dipole transitions

        # M72 equation 5 fits g(U) = A + B/U + C/U^2 + D*ln(U); keep the A and D ln(U)
        # terms, with the D = sqrt(3)/(2 pi) that M72 recommends for all optically allowed
        # transitions rounded to 0.28 (S20, section 2.5, where this pair is
        # described as the formula's "first two terms")
        mewe_A = 0.15
        mewe_D = 0.28

        prefactor = 45.585750051
        # VR62 approximation with the g_bar below from M72
        constantfactor = prefactor * A_naught_squared * pow(H_ionpot / epsilon_trans, 2) * fij

        U = engrid[startindex:] / epsilon_trans_ev
        # g_bar = 0.2
        g_bar = mewe_D * np.log(U) + mewe_A

        xs_excitation_vec[startindex:] = constantfactor * g_bar / U
    else:
        xs_excitation_vec[startindex:] = 0.0

    return xs_excitation_vec
