"""solve_spencerfano(): the Spencer-Fano equation for a Plasma, and the result it returns."""

import typing as t
from pathlib import Path

import matplotlib.axes as mplax
import numpy as np
import numpy.typing as npt

from pynonthermal.plasma import Plasma
from pynonthermal.spencerfano import SpencerFanoSolver


class SpencerFanoResult:
    """The solved Spencer-Fano equation for one plasma, as solve_spencerfano() returns it.

    The result is read only. Solve the plasma again to get the solution for another energy grid or
    another deposition rate density.

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
        """The temperature in K of the LTE level populations and the Saha equation, if the plasma set one."""
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
        index); for a CustomExcitation it is the key that the plasma gave.
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
    plasma: Plasma,
    deposition_ev_per_s_per_cm3: float,
    emin_ev: float = 1.0,
    emax_ev: float = 3000.0,
    npts: int = 4096,
    *,
    balance_tol: float = 1e-4,
    use_ar1985: bool = False,
    heating_only_approximation: bool = False,
    verbose: bool = False,
) -> SpencerFanoResult:
    """Solve the Spencer-Fano equation for a plasma and return the solution.

    plasma:
        what the gas contains (see pynonthermal.Plasma)
    deposition_ev_per_s_per_cm3:
        the rate of energy deposition per volume in eV s^-1 cm^-3. The energy fractions of a plasma
        with fixed populations do not depend on it, but the rate coefficients scale with it, and
        the populations of an IonBalance element depend on it.
    emin_ev, emax_ev:
        the bounds of the uniform energy grid in eV. An electron that degrades below emin_ev is
        taken to have thermalised, so its remaining energy counts as heating. Every ionisation
        potential of the plasma must lie above emin_ev.
    npts:
        the number of energy grid points. More points cost memory and time; check frac_sum of the
        result. The ARTIS defaults are emin_ev=0.1 and npts=4096.
    balance_tol:
        the relative tolerance of the population ratios of an IonBalance element. It has no effect
        on a plasma without one.
    use_ar1985:
        use the original Arnaud & Rothenflug (1985) ionisation cross sections
    heating_only_approximation:
        leave the excitation and ionisation loss terms out of the matrix and solve with the heating
        loss alone. The rates still follow from that approximate solution, so the energy fractions
        do not sum to one.
    verbose:
        print the setup, every added channel, and the per-ion breakdown of the solution
    """
    solver = SpencerFanoSolver(
        emin_ev=emin_ev,
        emax_ev=emax_ev,
        npts=npts,
        verbose=verbose,
        use_ar1985=use_ar1985,
        heating_only_approximation=heating_only_approximation,
    )
    if plasma.temperature is not None:
        solver.set_temperature(plasma.temperature)
    solver.set_atomic_data(
        adata_polars=plasma.adata_polars,
        use_collstrengths=plasma.use_collstrengths,
        maxnlevelslower=plasma.maxnlevelslower,
        maxnlevelsupper=plasma.maxnlevelsupper,
    )

    for element in plasma.elements:
        solver.add_element(
            element.Z,
            element.n_elem,
            element.populations,
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

    solver.solve(deposition_ev_per_s_per_cm3, override_n_e=plasma.free_electron_density, balance_tol=balance_tol)

    return SpencerFanoResult(solver)
