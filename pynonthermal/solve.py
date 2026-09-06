"""solve_spencerfano(): the Spencer-Fano equation for a list of elements, and the result it returns.

An Element says what one element of the gas contains and where its ion populations come from. It
holds no energy grid and no deposition rate, so the same elements can be solved on any grid and at
any rate. Every object here is frozen, so nothing can change under a solution that used it.
"""

import dataclasses
import math
import typing as t
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path

import matplotlib.axes as mplax
import numpy as np
import numpy.typing as npt
import polars as pl

from pynonthermal.base import CrossSectionFunc
from pynonthermal.spencerfano import SpencerFanoSolver

# a cross section [cm^2], either as a function of an array of energies [eV] or as an array on the
# energy grid of the solution. Prefer the function: the solver evaluates it between the grid points
# as well, where an array can only be interpolated, and it does not tie the element to one grid.
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
    """One element of the gas, with the rule that gives its ion populations.

    Give exactly one of ion_fractions, saha_ion_stages, or recomb_ratecoeffs. They are the three
    rules of SpencerFanoSolver.add_element(), which this passes on:

    - ion_fractions: the fraction of the element in each ion stage, keyed by ion stage
    - saha_ion_stages: contiguous ion stages whose populations come from the Saha equation at the
      temperature of the solution
    - recomb_ratecoeffs: recombination rate coefficients in cm^3 s^-1, keyed by the ion stage that
      recombines, balanced against the non-thermal ionisation rates of the solution

    Z:
        the atomic number
    n_elem:
        the number density of the element in cm^-3, summed over its ion stages
    partfuncs:
        partition functions keyed by ion stage, for saha_ion_stages only
    excitation:
        add the bound-bound excitations of every ion stage that has level data, with LTE level
        populations at the temperature of the solution
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
    ion_fractions: Mapping[int, float] | None = None
    saha_ion_stages: Sequence[int] | None = None
    recomb_ratecoeffs: Mapping[int, float] | None = None
    partfuncs: Mapping[int, float] | None = None
    excitation: bool = False
    builtin_channels: bool = True
    ionisation_channels: Sequence[CustomChannel] = ()
    excitations: Sequence[CustomExcitation] = ()


class SpencerFanoResult:
    """The solved Spencer-Fano equation, as solve_spencerfano() returns it.

    The result is read only. Solve again to get the solution for another energy grid or another
    deposition rate density.

    Every energy fraction is the share of the deposited energy that one channel takes (Kozma &
    Fransson 1992 equations 8 to 10), and every rate coefficient is per ion in s^-1 and scales with
    the deposition rate density.
    """

    __slots__ = ("_solver",)

    def __init__(self, solver: SpencerFanoSolver) -> None:
        """Wrap a solved SpencerFanoSolver. Call solve_spencerfano() instead of this."""
        self._solver = solver

    # the solution

    @property
    def engrid(self) -> npt.NDArray[np.float64]:
        """The energy grid of the solution in eV."""
        return self._solver.engrid

    @property
    def yvec(self) -> npt.NDArray[np.float64]:
        """The degradation spectrum y(E) [electrons cm^-2 s^-1 eV^-1] at every energy of engrid."""
        yvec = self._solver.yvec.view()
        yvec.flags.writeable = False
        return yvec

    @property
    def deposition_ev_per_s_per_cm3(self) -> float:
        """The deposition rate density of the solution in eV s^-1 cm^-3."""
        return self._solver.deposition_ev_per_s_per_cm3

    @property
    def temperature(self) -> float | None:
        """The temperature in K of the LTE level populations and the Saha equation, if one was given."""
        return self._solver.temperature

    @property
    def balance_iterations(self) -> int:
        """The number of iterations that the ionisation balance took, or zero without a balanced element."""
        return self._solver.balance_iterations

    # densities

    @property
    def n_e(self) -> float:
        """The free (thermal) electron density in cm^-3."""
        return self._solver.get_n_e()

    @property
    def n_e_nt(self) -> float:
        """The non-thermal electron density in cm^-3."""
        return self._solver.get_n_e_nt()

    @property
    def n_ion_tot(self) -> float:
        """The total number density of all nuclei in cm^-3."""
        return self._solver.get_n_ion_tot()

    def ion_populations(self, Z: int) -> dict[int, float]:
        """Get the number density in cm^-3 of each ion stage of element Z, keyed by ion stage."""
        return {ion_stage: n_ion for (Z_ion, ion_stage), n_ion in self._solver.ionpopdict.items() if Z_ion == Z}

    def ion_fractions(self, Z: int) -> dict[int, float]:
        """Get the fraction of element Z in each ion stage, keyed by ion stage."""
        return self._solver.get_ion_fractions(Z)

    # energy fractions

    @property
    def frac_heating(self) -> float:
        """The fraction of the deposited energy that heats the free thermal electrons."""
        return self._solver.get_frac_heating()

    @property
    def frac_ionisation(self) -> float:
        """The fraction of the deposited energy that goes to ionisation, over all ions."""
        return self._solver.get_frac_ionisation_tot()

    @property
    def frac_excitation(self) -> float:
        """The fraction of the deposited energy that goes to excitation, over all ions."""
        return self._solver.get_frac_excitation_tot()

    @property
    def frac_sum(self) -> float:
        """The sum of the three energy fractions. It is one when the energy grid resolves every channel."""
        return self._solver.get_frac_sum()

    def frac_ionisation_ion(self, Z: int, ion_stage: int) -> float:
        """Get one ion's share of the ionisation fraction."""
        return self._solver.get_frac_ionisation_ion(Z, ion_stage)

    def frac_excitation_ion(self, Z: int, ion_stage: int) -> float:
        """Get one ion's share of the excitation fraction."""
        return self._solver.get_frac_excitation_ion(Z, ion_stage)

    # rates

    def ionisation_ratecoeff(self, Z: int, ion_stage: int) -> float:
        """Get the non-thermal ionisation rate coefficient of one ion in s^-1.

        Multiply it by the ion number density for ionisations per second per cm^3.
        """
        return self._solver.get_ionisation_ratecoeff(Z, ion_stage)

    def excitation_ratecoeff(self, Z: int, ion_stage: int, transitionkey: t.Any) -> float:
        """Get the non-thermal excitation rate coefficient of one transition in s^-1.

        Multiply it by the population of the lower level for excitations per second per cm^3. For
        the excitations of Element(excitation=True) the key is (lower level index, upper level
        index); for a CustomExcitation it is the key that the element gave.
        """
        return self._solver.get_excitation_ratecoeff(Z, ion_stage, transitionkey)

    def transitionkeys(self, Z: int, ion_stage: int) -> list[t.Any]:
        """Get the keys of the transitions of one ion, in the order that they were added."""
        return list(self._solver.excitationlists.get((Z, ion_stage), {}))

    def eff_ionpot(self, Z: int, ion_stage: int) -> float:
        """Get the effective ionisation potential of one ion in eV (Kozma & Fransson 1992 equation 12)."""
        return self._solver.get_eff_ionpot(Z, ion_stage)

    # plots

    def plot_yspectrum(
        self,
        en_y_on_d_en: bool = False,
        xscalelog: bool = False,
        outputfilename: Path | str | None = None,
        axis: mplax.Axes | None = None,
    ) -> None:
        """Plot the degradation spectrum y(E) against electron energy (see SpencerFanoSolver)."""
        self._solver.plot_yspectrum(
            en_y_on_d_en=en_y_on_d_en, xscalelog=xscalelog, outputfilename=outputfilename, axis=axis
        )

    def plot_channels(
        self, outputfilename: Path | str | None = None, axis: mplax.Axes | None = None, xscalelog: bool = False
    ) -> None:
        """Plot each electron energy's contribution to ionisation, excitation, and heating."""
        self._solver.plot_channels(outputfilename=outputfilename, axis=axis, xscalelog=xscalelog)

    def plot_spec_channels(self, outputfilename: Path | str | None = None, xscalelog: bool = False) -> None:
        """Plot the degradation spectrum and the deposition channels as two stacked panels."""
        self._solver.plot_spec_channels(outputfilename=outputfilename, xscalelog=xscalelog)

    def print_analysis(self) -> None:
        """Print the per-ion and per-shell breakdown of the deposited energy."""
        verbose = self._solver.verbose
        self._solver.verbose = True
        try:
            self._solver.analyse_ntspectrum()
        finally:
            self._solver.verbose = verbose


def solve_spencerfano(
    elements: Sequence[Element],
    deposition_ev_per_s_per_cm3: float,
    emin_ev: float = 0.1,
    emax_ev: float = 3000.0,
    npts: int = 4096,
    *,
    temperature: float | None = None,
    free_electron_density: float | None = None,
    adata_polars: pl.DataFrame | None = None,
    use_collstrengths: bool = True,
    maxnlevelslower: int | None = 5,
    maxnlevelsupper: int | None = 250,
    balance_tol: float = 1e-4,
    use_ar1985: bool = False,
    heating_only_approximation: bool = False,
    verbose: bool = False,
) -> SpencerFanoResult:
    """Solve the Spencer-Fano equation for a list of elements and return the solution.

    elements:
        the elements of the gas, one entry per atomic number (see Element)
    deposition_ev_per_s_per_cm3:
        the rate of energy deposition per volume in eV s^-1 cm^-3. The energy fractions of fixed
        populations do not depend on it, but the rate coefficients scale with it, and the
        populations of an element with recomb_ratecoeffs depend on it.
    emin_ev, emax_ev:
        the bounds of the uniform energy grid in eV. An electron that degrades below emin_ev is
        taken to have thermalised, so its remaining energy counts as heating. Every ionisation
        potential must lie above emin_ev.
    npts:
        the number of energy grid points. More points cost memory and time; check frac_sum of the
        result. The defaults emin_ev=0.1 and npts=4096 are the ARTIS values.
    temperature:
        the temperature in K of the LTE level populations and of the Saha equation. It is required
        if any element gives saha_ion_stages or excitation=True.
    free_electron_density:
        the free electron density in cm^-3, in place of the one that the ion charges give. It
        cannot be combined with saha_ion_stages or recomb_ratecoeffs, which set the free electron
        density through charge neutrality.
    adata_polars:
        a levels/transitions table to use instead of the internal database (the CMFGEN-derived
        ARTIS atomic data), in the format that artistools.atomic.get_levels() returns with
        get_transitions=True
    use_collstrengths:
        compute the excitation cross sections from tabulated collision strengths where available
        (Li et al. 2012 equation 11), rather than from the oscillator strength alone
    maxnlevelslower, maxnlevelsupper:
        include only transitions whose lower level index is below maxnlevelslower and whose upper
        level index is below maxnlevelsupper; None disables that cutoff. The defaults match ARTIS.
    balance_tol:
        the relative tolerance of the population ratios of an element with recomb_ratecoeffs
    use_ar1985:
        use the original Arnaud & Rothenflug (1985) ionisation cross sections
    heating_only_approximation:
        leave the excitation and ionisation loss terms out of the matrix and solve with the heating
        loss alone. The rates still follow from that approximate solution, so the energy fractions
        do not sum to one.
    verbose:
        print the setup, every added channel, and the per-ion breakdown of the solution
    """
    if not elements:
        msg = "solve_spencerfano() needs at least one element"
        raise ValueError(msg)
    for element in elements:
        if not isinstance(element, Element):
            msg = f"every entry of elements must be an Element but one is a {type(element).__name__}"
            raise TypeError(msg)
    atomic_numbers = [element.Z for element in elements]
    duplicates = sorted({Z for Z in atomic_numbers if atomic_numbers.count(Z) > 1})
    if duplicates:
        msg = f"every element needs its own atomic number, but Z={duplicates} appear more than once"
        raise ValueError(msg)
    # the chained comparison also rejects nan
    if temperature is not None and not 0.0 < temperature < math.inf:
        msg = f"temperature must be greater than zero and finite but is {temperature}"
        raise ValueError(msg)

    solver = SpencerFanoSolver(
        emin_ev=emin_ev,
        emax_ev=emax_ev,
        npts=npts,
        verbose=verbose,
        use_ar1985=use_ar1985,
        heating_only_approximation=heating_only_approximation,
    )
    if temperature is not None:
        solver.set_temperature(temperature)
    solver.set_atomic_data(
        adata_polars=adata_polars,
        use_collstrengths=use_collstrengths,
        maxnlevelslower=maxnlevelslower,
        maxnlevelsupper=maxnlevelsupper,
    )

    for element in elements:
        solver.add_element(
            element.Z,
            element.n_elem,
            ion_fractions=element.ion_fractions,
            saha_ion_stages=element.saha_ion_stages,
            recomb_ratecoeffs=element.recomb_ratecoeffs,
            partfuncs=element.partfuncs,
            excitation=element.excitation,
            builtin_channels=element.builtin_channels,
        )
        for channel in element.ionisation_channels:
            solver.add_ionisation_channel(
                element.Z, channel.ion_stage, None, channel.ionpot_ev, channel.xs, channel.key
            )
        for transition in element.excitations:
            solver.add_excitation(
                element.Z,
                transition.ion_stage,
                None,
                transition.xs,
                transition.epsilon_trans_ev,
                transition.key,
                levelpopfrac=transition.levelpopfrac,
            )

    solver.solve(deposition_ev_per_s_per_cm3, override_n_e=free_electron_density, balance_tol=balance_tol)

    return SpencerFanoResult(solver)
