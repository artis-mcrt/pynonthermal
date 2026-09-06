"""The description of a plasma that solve_spencerfano() takes.

A Plasma says what the gas contains and nothing about how to solve it: the elements, where the ion
populations of each come from (pynonthermal.populations), the temperature of the LTE level
populations, and the atomic data to use. It holds no energy grid and no deposition rate, so the
same Plasma can be solved on any grid and at any deposition rate.

Every object here is frozen, so a Plasma cannot change under a solution that used it.
"""

import dataclasses
import math
import typing as t
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import polars as pl

from pynonthermal.base import CrossSectionFunc
from pynonthermal.populations import PopulationModel

# a cross section [cm^2], either as a function of an array of energies [eV] or as an array on the
# energy grid of the solution. Prefer the function: the solver evaluates it between the grid points
# as well, where an array can only be interpolated, and it does not tie the plasma to one grid.
CrossSection = npt.NDArray[np.float64] | CrossSectionFunc


@dataclasses.dataclass(frozen=True, slots=True)
class CustomChannel:
    """One collisional ionisation channel of an ion, with a cross section that the caller gives.

    The channel enters the ionisation term of the degradation equation (Kozma & Fransson 1992
    equation 7) in the same way as a shell of the built-in table. To replace the built-in shells of
    an element instead of adding to them, give Element(builtin_channels=False).

    ion_stage:
        the ion stage that this channel ionises
    ionpot_ev:
        the ionisation potential of the channel in eV. It must lie on the energy grid of the
        solution, and the cross section must be zero at and below it.
    xs:
        the cross section in cm^2 (see CrossSection)
    key:
        a key that names the channel in its ion, for the verbose output. The default is the number
        of channels that the ion already has.
    """

    ion_stage: int
    ionpot_ev: float
    xs: CrossSection
    key: t.Any | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class CustomExcitation:
    """One bound-bound excitation of an ion, with a cross section that the caller gives.

    ion_stage:
        the ion stage that this transition excites
    levelpopfrac:
        the population of the lower level as a fraction of the ion population. The level population
        follows the ion population, so it is right whether the populations are fixed or come from
        a balance.
    epsilon_trans_ev:
        the transition energy in eV
    xs:
        the cross section in cm^2 (see CrossSection)
    key:
        a key that identifies the transition, to pass to SpencerFanoResult.excitation_ratecoeff().
        The default is the number of transitions that the ion already has.
    """

    ion_stage: int
    levelpopfrac: float
    epsilon_trans_ev: float
    xs: CrossSection
    key: t.Any | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Element:
    """One element of a plasma, with the model that gives its ion populations.

    Z:
        the atomic number
    n_elem:
        the number density of the element in cm^-3, summed over its ion stages
    populations:
        where the ion populations come from: Fixed, Saha, or IonBalance
        (see pynonthermal.populations)
    excitation:
        add the bound-bound excitations of every ion stage that has level data, with LTE level
        populations at the temperature of the plasma
    builtin_channels:
        give every ion stage the built-in ionisation cross sections. With False, the element is
        ionised only through the channels in ionisation_channels.
    ionisation_channels:
        ionisation channels with cross sections that the caller gives
    excitations:
        bound-bound excitations with cross sections that the caller gives
    """

    Z: int
    n_elem: float
    populations: PopulationModel
    excitation: bool = False
    builtin_channels: bool = True
    ionisation_channels: Sequence[CustomChannel] = ()
    excitations: Sequence[CustomExcitation] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class Plasma:
    """The elements of a plasma and the data that their populations need.

    elements:
        the elements of the plasma, one entry per atomic number
    temperature:
        the temperature in K of the LTE level populations and of the Saha equation. It is required
        if any element has a Saha population model or excitation=True.
    free_electron_density:
        the free electron density in cm^-3, in place of the one that the ion charges give. It
        cannot be combined with a Saha or IonBalance element, whose populations set the free
        electron density through charge neutrality.
    adata_polars:
        a levels/transitions table to use instead of the internal database (the CMFGEN-derived
        ARTIS atomic data), in the format that artistools.atomic.get_levels() returns with
        get_transitions=True: one row per ion with Z, ion_stage, and nested "levels" and
        "transitions" frames
    use_collstrengths:
        compute the excitation cross sections from tabulated collision strengths where available
        (Li et al. 2012 equation 11). Permitted transitions without one (or all permitted
        transitions, when False) instead use the oscillator strength via the van Regemorter
        approximation; forbidden transitions outside the collision-strength path get a zero cross
        section.
    maxnlevelslower, maxnlevelsupper:
        include only transitions whose lower level index is below maxnlevelslower and whose upper
        level index is below maxnlevelsupper; None disables that cutoff. The defaults match ARTIS.
    """

    elements: Sequence[Element]
    temperature: float | None = None
    free_electron_density: float | None = None
    adata_polars: pl.DataFrame | None = None
    use_collstrengths: bool = True
    maxnlevelslower: int | None = 5
    maxnlevelsupper: int | None = 250

    def __post_init__(self) -> None:
        """Check the parts of the plasma that do not need the energy grid of a solution."""
        if not self.elements:
            msg = "a plasma needs at least one element"
            raise ValueError(msg)
        for element in self.elements:
            if not isinstance(element, Element):
                msg = f"every entry of elements must be an Element but one is a {type(element).__name__}"
                raise TypeError(msg)
        atomic_numbers = [element.Z for element in self.elements]
        duplicates = sorted({Z for Z in atomic_numbers if atomic_numbers.count(Z) > 1})
        if duplicates:
            msg = f"every element of a plasma needs its own atomic number, but Z={duplicates} appear more than once"
            raise ValueError(msg)
        # the chained comparisons also reject nan
        if self.temperature is not None and not 0.0 < self.temperature < math.inf:
            msg = f"temperature must be greater than zero and finite but is {self.temperature}"
            raise ValueError(msg)
        if self.free_electron_density is not None and not 0.0 < self.free_electron_density < math.inf:
            msg = f"free_electron_density must be greater than zero and finite but is {self.free_electron_density}"
            raise ValueError(msg)
