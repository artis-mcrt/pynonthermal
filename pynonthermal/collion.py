"""Collisional ionisation: the shell table, the secondary-electron distribution, and the channels.

The comments and the docstrings of this module cite these publications by their labels:

- KF92: Kozma & Fransson 1992, ApJ, 390, 602-621, doi:10.1086/171311
- OPB71: Opal, Peterson & Beaty 1971, J. Chem. Phys., 55, 4100-4106, doi:10.1063/1.1676707
- AR85: Arnaud & Rothenflug 1985, A&AS, 60, 425-457, ADS 1985A&AS...60..425A (the journal gave no DOI)
- Y81: Younger 1981, J. Quant. Spectrosc. Radiat. Transfer, 26, 329-337, doi:10.1016/0022-4073(81)90127-8
- KM93: Kaastra & Mewe 1993, A&AS, 97, 443-482, ADS 1993A&AS...97..443K (the journal gave no DOI)
- S20: Shingles et al. 2020, MNRAS, 492, 2029-2043, doi:10.1093/mnras/stz3412
"""

import dataclasses
import math
import os
import typing as t
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import numpy as np
import numpy.typing as npt
import polars as pl

import pynonthermal
from pynonthermal.axelrod import get_binding_energies
from pynonthermal.axelrod import get_lotz_xs_ionisation_vec
from pynonthermal.axelrod import get_shell_configs
from pynonthermal.axelrod import LOTZ_A_CM2_EV2
from pynonthermal.base import _check_bool
from pynonthermal.base import _check_ion
from pynonthermal.base import _is_bool
from pynonthermal.base import _is_integer
from pynonthermal.base import _warn
from pynonthermal.base import CrossSectionFunc
from pynonthermal.base import get_xs_on_grid
from pynonthermal.constants import EV

SUBSHELLNAMES = [
    "K ",
    "L1",
    "L2",
    "L3",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "N1",
    "N2",
    "N3",
    "N4",
    "N5",
    "N6",
    "N7",
    "O1",
    "O2",
    "O3",
    "O4",
    "O5",
    "O6",
    "O7",
    "P1",
    "P2",
    "P3",
    "P4",
    "Q1",
]


def get_nist_ionisation_energies_ev() -> dict[tuple[int, int], float]:
    """Get a dictionary of ionisation_energy [eV] keyed by (atomic_number, ion_stage).

    Each call gives a new copy of the cached table. A write to the copy therefore does not change
    the checks of the channels, or the default energy of their Auger electrons.
    """
    return dict(_get_nist_ionisation_energies_ev_cached())


@lru_cache
def _get_nist_ionisation_energies_ev_cached() -> Mapping[tuple[int, int], float]:
    # the table that the package reads. It is read-only, because every caller shares it.
    dfnist = (
        pl.read_csv(
            Path(pynonthermal.DATADIR / "nist_ionization.txt.zst"),
            separator="\t",
            infer_schema_length=50,
            columns=["At. num", "Ion Charge", "Ionization Energy (a) (eV)"],
            ignore_errors=True,
        )
        .rename({"At. num": "Z", "Ionization Energy (a) (eV)": "ionisation_energy_ev"})
        .with_columns(ion_stage=pl.col("Ion Charge").cast(pl.Int64) + 1)
        .drop("Ion Charge")
        .drop_nulls()
    )

    return MappingProxyType(
        {
            (atomic_number, ion_stage): ionisation_energy_ev
            for atomic_number, ion_stage, ionisation_energy_ev in dfnist.select(
                [
                    "Z",
                    "ion_stage",
                    "ionisation_energy_ev",
                ]
            ).iter_rows(named=False)
        }
    )


def read_colliondata(collionfilename: str | Path = "collion.txt") -> pl.DataFrame:
    """Get the table of ionisation shells: the fits of collionfilename, and Lotz rows for the other ions.

    The result is a copy of a cached table, so a write to it does not change the table of
    any other caller or solver.
    """
    return _read_colliondata_cached(os.fspath(collionfilename)).clone()


@lru_cache
def _read_colliondata_cached(collionfilename: str) -> pl.DataFrame:
    dfcollion = pl.read_csv(
        Path(pynonthermal.DATADIR, collionfilename),
        separator=" ",
        has_header=False,
        skip_lines=1,
        truncate_ragged_lines=True,
        schema={
            "Z": pl.Int64,
            "nelec": pl.Int64,
            "n": pl.Int64,
            "l": pl.Int64,
            "ionpot_ev": pl.Float64,
            "A": pl.Float64,
            "B": pl.Float64,
            "C": pl.Float64,
            "D": pl.Float64,
        },
    )

    nist_ionisation_energies_ev = _get_nist_ionisation_energies_ev_cached()
    elements_electron_binding = get_binding_energies()
    all_shells_q = get_shell_configs()
    covered_z_nelec = set(dfcollion.select(["Z", "nelec"]).iter_rows())
    new_rows: list[dict[str, int | float]] = []
    for Z in range(1, len(elements_electron_binding) + 1):
        for ionstage in range(1, Z + 1):
            any_data_matched = (Z, Z - ionstage + 1) in covered_z_nelec

            if not any_data_matched:
                ioncharge = ionstage - 1
                nbound = Z - ioncharge  # number of bound electrons

                try:
                    ion_shells_q = all_shells_q[Z - 1]
                    ionpot = nist_ionisation_energies_ev[(Z, ionstage)] * EV
                except (KeyError, IndexError):
                    continue

                electron_count = 0
                for shellindex in range(len(ion_shells_q)):
                    electronsinshell = ion_shells_q[shellindex]

                    electron_count += electronsinshell

                    if electronsinshell <= 0:
                        continue
                    enbinding = elements_electron_binding[Z - 1][shellindex]
                    if enbinding <= 0:
                        # if we don't have the shell's binding energy, use the previous one.
                        # shellindex 0 has no previous shell, and a negative index would silently
                        # wrap around to the outermost shell.
                        if shellindex == 0:
                            msg = f"No binding energy for the innermost shell of Z={Z}"
                            raise ValueError(msg)
                        enbinding = elements_electron_binding[Z - 1][shellindex - 1]
                        assert enbinding > 0

                    p = max(ionpot, enbinding)
                    collionrow: dict[str, int | float] = {
                        "Z": Z,
                        "nelec": Z - ionstage + 1,
                        "n": -1,
                        "l": -shellindex,
                        "ionpot_ev": p / EV,
                        "A": -1.0,
                        "B": -1.0,
                        "C": -1.0,
                        "D": -1.0,
                    }

                    new_rows.append(collionrow)
                    if electron_count >= nbound:
                        break

    # Append Lotz approximate cross sections to the Arnaud/Rothenflug data
    return (
        pl.concat([dfcollion, pl.DataFrame(new_rows)])
        .with_columns(ion_stage=pl.col("Z") - pl.col("nelec") + 1)
        .sort(by=["Z", "ion_stage", "ionpot_ev", "n", "l"])
    )


# The X-ray shells K, L1, L2, L3, M1, M2, and M3 of KM93 table 2, as (n, l, g). The statistical
# weight g = 2j + 1 gives the weights of the mean of two X-ray shells with the same n and l.
_KM93_SHELLS: tuple[tuple[int, int, int], ...] = (
    (1, 0, 2),
    (2, 0, 2),
    (2, 1, 2),
    (2, 1, 4),
    (3, 0, 2),
    (3, 1, 2),
    (3, 1, 4),
)

# the largest number of Auger electrons after one inner-shell ionisation in KM93 table 2
KM93_MAX_AUGER_ELECTRONS: int = 9


def read_auger_data() -> dict[tuple[int, int, int, int], tuple[float, tuple[float, ...]]]:
    """Get the Auger data of KM93 table 2 for each subshell, as ARTIS (S20) reads it.

    The key is (Z, ion_stage, n, l), with the ion stage before the ionisation. The value is
    (auger_energy_ev, probabilities):

    - auger_energy_ev is the mean total kinetic energy [eV] of the Auger electrons after one
      ionisation of the subshell. It is the E_A of the table with the correction of KM93 equation 7.
      The mean includes the ionisations that emit no Auger electron. A negative value becomes zero.
    - probabilities[a] is the probability that the ionisation emits a Auger electrons, for a from 0
      to KM93_MAX_AUGER_ELECTRONS.

    The table gives the X-ray shells L2 and L3, and M2 and M3, in different rows. Each pair is one
    subshell here, with the mean of the two rows, and the statistical weights as the weights.

    Each call gives a new copy of the cached table.
    """
    return dict(_read_auger_data_cached())


@lru_cache
def _read_auger_data_cached() -> Mapping[tuple[int, int, int, int], tuple[float, tuple[float, ...]]]:
    # the table that the package reads. It is read-only, because every caller shares it.
    # g_sum holds the statistical weight that each subshell has accumulated
    g_sum: dict[tuple[int, int, int, int], int] = {}
    energies_ev: dict[tuple[int, int, int, int], float] = {}
    probabilities: dict[tuple[int, int, int, int], npt.NDArray[np.float64]] = {}
    with Path(pynonthermal.DATADIR, "auger-km1993-table2.txt").open(encoding="utf-8") as augerfile:
        for line in augerfile:
            if not line.strip() or line.startswith("#"):
                continue
            # The columns have a fixed width. An ionisation potential of five digits touches the shell
            # number, and two probabilities can touch.
            Z = int(line[0:2])
            ion_stage = int(line[2:5])
            n, l, g = _KM93_SHELLS[int(line[5:7]) - 1]  # noqa: E741
            ionpot_ev = float(line[7:14])
            en_auger_nocorrection_ev = float(line[14:21])
            epsilon_e3 = int(line[21:25])
            # The table gives the probability of 1 to 10 ejected electrons. That count includes the
            # electron of the ionisation itself, so column a is the probability of a Auger electrons.
            row_probabilities = np.array(
                [int(line[26 + 5 * a : 31 + 5 * a]) / 1.0e4 for a in range(KM93_MAX_AUGER_ELECTRONS + 1)]
            )
            # KM93 equation 7, with the ionisation potential of the table
            en_auger_ev = max(0.0, en_auger_nocorrection_ev - epsilon_e3 / 1000.0 * ionpot_ev)

            key = (Z, ion_stage, n, l)
            g_old = g_sum.get(key, 0)
            g_sum[key] = g_old + g
            energies_ev[key] = (g_old * energies_ev.get(key, 0.0) + g * en_auger_ev) / (g_old + g)
            probabilities[key] = (g_old * probabilities.get(key, 0.0) + g * row_probabilities) / (g_old + g)

    return MappingProxyType(
        {key: (energies_ev[key], tuple(float(p) for p in probabilities[key])) for key in energies_ev}
    )


def Psecondary_vec(
    e_p: npt.NDArray[np.float64] | float, e_s: npt.NDArray[np.float64] | float, ionpot_ev: float, J: float
) -> npt.NDArray[np.float64]:
    """Evaluate the secondary-electron energy distribution over arrays of e_p and/or e_s.

    This is the Lorentzian distribution of KF92 equation 4, their analytically
    integrable adaptation of the shape OPB71 fitted to their measurements
    (whose published exponent is 2.1 rather than 2), with the width parameter J from get_J().

    e_p is the primary electron energy [eV] and e_s the secondary electron energy [eV]. The two
    arguments broadcast against each other, so either may be a scalar. Psecondary() is the scalar form.

    The distribution is normalised over its physical support, e_s in [0, (e_p - ionpot_ev) / 2], but is
    not truncated to it here, so the caller must supply limits of integration that stay inside it. Note
    also that the normalisation diverges as e_p approaches ionpot_ev from above, so a limit of
    integration that is snapped onto a coarse energy grid near the threshold can be badly wrong.
    """
    arr_e_p = np.asarray(e_p, dtype=np.float64)
    arr_e_s = np.asarray(e_s, dtype=np.float64)

    # below the ionisation threshold there are no secondaries (and the normalisation
    # atan((e_p - ionpot_ev) / 2 / J) would be zero or negative)
    norm = np.arctan((arr_e_p - ionpot_ev) / 2.0 / J)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(arr_e_p > ionpot_ev, 1.0 / J / norm / (1 + (arr_e_s / J) ** 2), 0.0)


def Psecondary(e_p: float, ionpot_ev: float, J: float, e_s: float = -1, epsilon: float = -1) -> float:
    """Evaluate the secondary-electron energy distribution [eV^-1] of KF92 equation 4 at one energy.

    e_p is the energy [eV] of the primary electron of an impact ionisation. Give the energy e_s [eV]
    of the secondary electron, or the energy loss epsilon [eV] of the primary electron. Then e_s is
    epsilon - ionpot_ev. J [eV] is the width parameter of get_J(). Psecondary_vec() is the form for
    arrays, and its docstring gives the support that the limits of integration must stay inside.
    """
    if e_s < 0 and epsilon < 0:
        msg = "either the secondary energy e_s or the primary energy loss epsilon must be given"
        raise ValueError(msg)

    if e_s < 0:
        e_s = epsilon - ionpot_ev

    return float(Psecondary_vec(e_p=e_p, e_s=e_s, ionpot_ev=ionpot_ev, J=J))


def get_J(Z: int, ion_stage: int, ionpot_ev: float) -> float:
    """Get the width parameter J [eV] of the secondary-electron distribution of one ion.

    He I, Ne I, and Ar I have the values of OPB71, as KF92 applied them. The solver uses that value for
    every shell of these three ions, as ARTIS does. Every other ion has 0.6 times the ionpot_ev [eV] of
    each channel, so its J is different for each shell.
    """
    # a charge in place of the ion stage would give the J of the wrong stage
    Z, ion_stage = _check_ion(Z, ion_stage)
    # values from OPB71 as applied by KF92. They are whole-atom measurements
    # dominated by valence-shell ionisation, but are used for every shell of the ion to match ARTIS.
    if ion_stage == 1:
        if Z == 2:  # He I
            return 15.8
        if Z == 10:  # Ne I
            return 24.2
        if Z == 18:  # Ar I
            return 10.0

    return 0.6 * ionpot_ev


def get_arxs_array_shell(
    arr_enev: npt.NDArray[np.float64], shell: Mapping[str, int | float], lotz_a_cm2_ev2: float = LOTZ_A_CM2_EV2
) -> npt.NDArray[np.float64]:
    """Get the impact ionisation cross section [cm^2] of one shell at each energy of arr_enev [eV].

    This is the sigma_ic of KF92 (equations 5, 10, and 11, and the ionisation term of equation 7).
    shell is a row of read_colliondata(). A shell with fit data uses the formula of Y81 with the A to
    D coefficients of AR85. A shell with n < 0 has no fit data, so it uses the Lotz formula with the
    constant lotz_a_cm2_ev2 [cm^2 eV^2].
    """
    if shell["n"] < 0:
        return get_lotz_xs_ionisation_vec(shell, arr_en_ev=arr_enev, lotz_a_cm2_ev2=lotz_a_cm2_ev2)

    ionpot_ev = float(shell["ionpot_ev"])
    # an explicit float dtype, because zeros_like of an integer energy array would truncate every
    # cross section to zero
    xs = np.zeros(arr_enev.shape, dtype=np.float64)
    abovethreshold = arr_enev > ionpot_ev
    u = arr_enev[abovethreshold] / ionpot_ev
    # a few of the fits (Ne I n=2 l=0, Na II n=2 l=1) go slightly negative just above threshold,
    # so clamp to zero to keep the cross section physical
    xs[abovethreshold] = np.maximum(
        0.0,
        1e-14
        * (
            shell["A"] * (1 - 1 / u)
            + shell["B"] * (1 - 1 / u) ** 2
            + shell["C"] * np.log(u)
            + shell["D"] * np.log(u) / u
        )
        / (u * ionpot_ev**2),
    )
    return xs


# The relative tolerance of the threshold of a channel below the energy that the ion must keep. That
# energy is the sum of the NIST ground-state potentials that the channel crosses, less the energy of
# the initial level. The check applies to a multiple ionisation and to an autoionisation. A calculated
# threshold can be a little below that energy. Inside the tolerance the Auger electrons get zero
# energy. Outside the tolerance, IonisationChannel.from_xs() gives a UserWarning. The channel then
# uses the threshold and the Auger electron energy as given, and the default Auger electron energy is
# zero. Thresholds from other atomic data can differ from NIST by more than the tolerance, and a
# single direct ionisation has no such check.
MULTIPLE_IONPOT_REL_TOL: float = 0.01


@dataclass(frozen=True, slots=True, eq=False)
class IonisationChannel:
    """One collisional ionisation channel of an ion, with the cross section that the solver uses.

    The channel carries neither the atomic number nor the ion stage nor the ion number density.
    The solver holds a list of channels for each ion, and it holds the number densities in
    SpencerFanoSolver.ionpopdict.

    Build a channel with from_xs(), which evaluates and checks the cross section.
    SpencerFanoSolver makes the built-in channels through get_ion_ionisation_channels(), and a
    custom channel through SpencerFanoSolver.add_ionisation_channel().
    """

    ionpot_ev: float
    """The threshold of the channel [eV].

    It is the ionisation potential, or the excitation energy of the autoionising level of an
    autoionisation channel.
    """

    xs: CrossSectionFunc
    """The cross section sigma(E) [cm^2] at any array of energies [eV].

    calculate_N_e() evaluates this between the grid points, just above the ionisation potential.
    The domain of that integral is narrower than one bin of the grid. A function is therefore necessary
    here, and an array on the grid is not enough. An autoionisation channel uses only xs_grid.
    """

    xs_grid: npt.NDArray[np.float64]
    """The cross section [cm^2] at every energy of SpencerFanoSolver.engrid."""

    J_ev: float
    """The width of the secondary-electron distribution [eV], from get_J().

    An autoionisation channel does not use it.
    """

    key: t.Any
    """A key that identifies the channel in its ion, and names it in the verbose output."""

    lotz: bool = False
    """True if the cross section is the Lotz formula, which the shells without a fit use."""

    n_ejected: int = 1
    """The number of electrons that one ionisation removes from the ion.

    The ionisation takes the ion from ion_stage to ion_stage + n_ejected. In a direct ionisation,
    the first ejected electron has the Lorentzian distribution of Psecondary_vec(). The other
    electrons are Auger electrons (n_auger_electrons), which share auger_electron_energy_ev.
    """

    auger_electron_energy_ev: float = 0.0
    """The total kinetic energy [eV] of the Auger electrons of one ionisation.

    An Auger electron appears at one energy, auger_electron_energy_ev / n_auger_electrons, and
    not with the Lorentzian distribution. The energy is zero for a direct single ionisation. The
    ion keeps ionpot_ev - auger_electron_energy_ev as ionisation energy, and the ionisation
    fraction counts only that energy. With mean_auger_electron, one electron holds all of this
    energy, for every value of n_ejected.
    """

    autoionisation: bool = False
    """True for an excitation-autoionisation channel.

    The primary electron loses exactly ionpot_ev, the excitation energy of the autoionising
    level. No electron has the Lorentzian distribution. All n_ejected electrons appear at the
    energy auger_electron_energy_ev / n_ejected.
    """

    mean_auger_electron: bool = False
    """True for a built-in channel with the Auger data of KM93, in the method of ARTIS (S20).

    The source term then holds one electron for each ionisation, with the energy
    auger_electron_energy_ev. That energy is the mean total energy of the Auger electrons of the
    subshell, for every value of n_ejected. SpencerFanoSolver(use_km1993_auger=True) makes these
    channels.
    """

    @property
    def n_auger_electrons(self) -> int:
        """The number of Auger electrons in the source term of one ionisation.

        An autoionisation channel has no Lorentzian electron, so all of its n_ejected electrons
        are Auger electrons. A direct ionisation has n_ejected - 1 of them. A channel with
        mean_auger_electron has one electron in the source term, which holds the mean total energy.
        """
        if self.mean_auger_electron:
            return 1
        return self.n_ejected if self.autoionisation else self.n_ejected - 1

    def evaluate_xs(self, energies_ev: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate and check the cross section at each supplied energy."""
        values = np.asarray(self.xs(energies_ev), dtype=np.float64)
        name = f"The cross section of ionisation channel {self.key}"
        if values.shape != energies_ev.shape:
            msg = f"{name} must have shape {energies_ev.shape}, but has shape {values.shape}"
            raise ValueError(msg)
        invalid = ~np.isfinite(values) | (values < 0.0)
        if invalid.any():
            index = int(np.flatnonzero(invalid)[0])
            msg = (
                f"{name} must be finite and non-negative. Its value at {energies_ev[index]} eV is {values[index]} cm^2."
            )
            raise ValueError(msg)
        return values

    @property
    def extra_electron_energy_ev(self) -> float:
        """The former name of auger_electron_energy_ev. It is deprecated."""
        _warn(
            "IonisationChannel.extra_electron_energy_ev is deprecated. Its name is now auger_electron_energy_ev.",
            DeprecationWarning,
        )
        return self.auger_electron_energy_ev

    @classmethod
    def from_xs_grid(
        cls,
        arr_enev: npt.NDArray[np.float64],
        Z: int,
        ion_stage: int,
        ionpot_ev: float,
        xs_vec: npt.NDArray[np.float64],
        key: t.Any,
        *,
        n_ejected: int = 1,
        level_energy_ev: float = 0.0,
        auger_electron_energy_ev: float | None = None,
        autoionisation: bool | np.bool_ = False,
        extra_electron_energy_ev: float | None = None,
    ) -> t.Self:
        """Make a channel from cross sections [cm^2] at every energy of the grid arr_enev [eV].

        The keywords are those of from_xs().
        """
        auger_electron_energy_ev = _resolve_auger_electron_energy_ev(auger_electron_energy_ev, extra_electron_energy_ev)
        name = f"The cross section of ionisation channel {key}"
        xs_grid = get_xs_on_grid(xs_vec, arr_enev, name)
        # check the array of the caller here. The interpolation below holds the cross section at
        # zero up to the threshold, so from_xs() would find nothing left to reject. An
        # autoionisation channel keeps its value at the threshold, as an excitation does.
        if not autoionisation:
            _check_zero_below_ionpot(arr_enev, xs_grid, float(ionpot_ev), name)
        # a read-only copy of the grid, as of the cross section. With the array of the caller, a later
        # write to it changed xs() between the grid points, and xs_grid stayed as checked.
        enev_grid = np.array(arr_enev, dtype=np.float64)
        enev_grid.flags.writeable = False

        return cls.from_xs(
            arr_enev=enev_grid,
            Z=Z,
            ion_stage=ion_stage,
            ionpot_ev=ionpot_ev,
            xs=_GridXs(enev_grid, xs_grid, float(ionpot_ev), keep_threshold=bool(autoionisation)),
            key=key,
            n_ejected=n_ejected,
            level_energy_ev=level_energy_ev,
            auger_electron_energy_ev=auger_electron_energy_ev,
            autoionisation=autoionisation,
        )

    @classmethod
    def from_xs(
        cls,
        arr_enev: npt.NDArray[np.float64],
        Z: int,
        ion_stage: int,
        ionpot_ev: float,
        xs: CrossSectionFunc,
        key: t.Any,
        lotz: bool = False,
        *,
        n_ejected: int = 1,
        level_energy_ev: float = 0.0,
        auger_electron_energy_ev: float | None = None,
        autoionisation: bool | np.bool_ = False,
        extra_electron_energy_ev: float | None = None,
    ) -> t.Self:
        """Make a channel, and check the cross section that xs gives on the energy grid arr_enev [eV].

        get_J() gives the width of the secondary-electron distribution from Z, ion_stage, and
        ionpot_ev, so the channel is always consistent with its ionisation potential.

        n_ejected:
            the number of electrons that one ionisation removes. ion_stage + n_ejected must not
            be more than Z + 1, the bare nucleus.
        level_energy_ev:
            the energy [eV] of the initial level above the ground state. It must be less than the
            NIST ionisation potential of the ion. For a multiple ionisation or an autoionisation,
            it lowers the energy that the ion must keep, and it increases the default
            auger_electron_energy_ev. The channel does not keep it.
        auger_electron_energy_ev:
            the total kinetic energy [eV] of the Auger electrons. None (the
            default) gives zero for a channel with n_ejected=1 and autoionisation=False. In the
            other cases, None gives the value from energy conservation (see
            _get_auger_electron_energy_ev()). A value must be less than ionpot_ev. If the NIST
            data holds the potentials that the ionisation crosses, the ion keeps at least their
            sum, less level_energy_ev and less MULTIPLE_IONPOT_REL_TOL. A threshold or a value
            that leaves the ion less than that gives a UserWarning, and the channel uses it.
        autoionisation:
            True for an excitation-autoionisation channel. Then ionpot_ev is the excitation
            energy of the autoionising level, and all n_ejected electrons are Auger electrons.
            The cross section below that energy is set to zero, as in
            SpencerFanoSolver.add_excitation(). The cross section is the excitation cross
            section times the probability of autoionisation, because the level can also emit
            a photon.
        extra_electron_energy_ev:
            the former name of auger_electron_energy_ev. It is deprecated.
        """
        auger_electron_energy_ev = _resolve_auger_electron_energy_ev(auger_electron_energy_ev, extra_electron_energy_ev)
        name = f"The cross section of ionisation channel {key}"
        # the same check as every method of the solver. Without it, a charge in place of the ion
        # stage gives a channel with the wrong J from get_J().
        Z, ion_stage = _check_ion(Z, ion_stage, needs_electron=True)

        # the chained comparison also rejects nan, for which every comparison is False
        if _is_bool(ionpot_ev) or not 0.0 < ionpot_ev < math.inf:
            msg = f"ionpot_ev must be greater than zero and finite but is {ionpot_ev}"
            raise ValueError(msg)
        # a Python float, so that every comparison below uses the same value. A numpy float32
        # makes a comparison in float32.
        ionpot_ev = float(ionpot_ev)

        if not _is_integer(n_ejected) or n_ejected < 1:
            msg = f"n_ejected must be an integer of at least 1 but is {n_ejected!r}"
            raise ValueError(msg)
        # a Python integer, so that the sum below cannot wrap around in a fixed-width numpy type
        n_ejected = int(n_ejected)
        # a numpy comparison gives a numpy bool, which is not a subclass of bool
        autoionisation = _check_bool(autoionisation, "autoionisation")
        # the chained comparison also rejects nan
        if _is_bool(level_energy_ev) or not 0.0 <= level_energy_ev < math.inf:
            msg = f"level_energy_ev must be at least zero and finite but is {level_energy_ev}"
            raise ValueError(msg)
        level_energy_ev = float(level_energy_ev)
        _check_level_energy_ev(Z, ion_stage, level_energy_ev)
        if ion_stage + n_ejected > Z + 1:
            msg = (
                f"Z={Z} ion_stage {ion_stage} has {Z + 1 - ion_stage} electrons, so an ionisation cannot remove"
                f" n_ejected={n_ejected} of them"
            )
            raise ValueError(msg)
        if n_ejected == 1 and not autoionisation:
            # the comparison with zero also rejects nan
            if auger_electron_energy_ev is not None and auger_electron_energy_ev != 0.0:
                msg = (
                    "a channel with n_ejected=1 and autoionisation=False has no Auger electrons, so"
                    f" auger_electron_energy_ev must be zero but is {auger_electron_energy_ev}. Set"
                    " autoionisation=True for a channel that emits its electron as an Auger electron."
                )
                raise ValueError(msg)
            auger_ev = 0.0
            auger_warning = None
        else:
            auger_ev, auger_warning = _get_auger_electron_energy_ev(
                Z, ion_stage, ionpot_ev, n_ejected, auger_electron_energy_ev, level_energy_ev
            )
        # the Auger electrons get their energy from the ionisation potential, so the ion keeps
        # a positive energy. The chained comparison also rejects nan.
        if not 0.0 <= auger_ev < ionpot_ev:
            msg = (
                f"auger_electron_energy_ev must be at least zero and less than ionpot_ev ({ionpot_ev} eV)"
                f" but is {auger_ev}"
            )
            raise ValueError(msg)

        xs_grid = get_xs_on_grid(xs, arr_enev, name)

        if autoionisation:
            # no electron below the threshold can excite the level, so the cross section there is
            # zero, and the value at the threshold stays. This is the rule of
            # SpencerFanoSolver.add_excitation(). The array is a fresh copy, so the write is safe.
            xs_grid.flags.writeable = True
            xs_grid[arr_enev < ionpot_ev] = 0.0
            xs_grid.flags.writeable = False
        else:
            _check_zero_below_ionpot(arr_enev, xs_grid, ionpot_ev, name)

        # calculate_N_e() evaluates the cross section between the points of the energy grid, so a
        # function that ignores its argument fails there. The probe finds that at the call site,
        # where the message can name the channel.
        probe_enev = np.linspace(float(arr_enev[0]), float(arr_enev[-1]), num=3 if len(arr_enev) != 3 else 4)
        probe_xs = np.asarray(xs(probe_enev), dtype=np.float64)
        if probe_xs.shape != probe_enev.shape:
            msg = (
                f"{name} must give one value for each energy that it receives, but it gave"
                f" {probe_xs.shape} values for {len(probe_enev)} energies."
                " The solver also evaluates it between the points of the energy grid."
            )
            raise ValueError(msg)

        # the warning comes after the last check, so that a rejected call gives only its error
        if auger_warning is not None:
            _warn(auger_warning)

        return cls(
            ionpot_ev=ionpot_ev,
            xs=xs,
            xs_grid=xs_grid,
            J_ev=get_J(Z, ion_stage, ionpot_ev),
            key=key,
            lotz=lotz,
            n_ejected=n_ejected,
            auger_electron_energy_ev=float(auger_ev),
            autoionisation=autoionisation,
        )


def _resolve_auger_electron_energy_ev(
    auger_electron_energy_ev: float | None, extra_electron_energy_ev: float | None
) -> float | None:
    # the value of the Auger electron energy from either keyword. extra_electron_energy_ev
    # is the former name, which v2026.9.23 released.
    if extra_electron_energy_ev is None:
        return auger_electron_energy_ev
    _warn(
        "the extra_electron_energy_ev argument is deprecated. Its name is now auger_electron_energy_ev.",
        DeprecationWarning,
    )
    if auger_electron_energy_ev is not None:
        msg = "give the energy of the Auger electrons once, as auger_electron_energy_ev"
        raise ValueError(msg)
    return extra_electron_energy_ev


def _check_level_energy_ev(Z: int, ion_stage: int, level_energy_ev: float) -> None:
    # a bound level of the ion lies below its ionisation potential. The check needs the NIST data,
    # so an ion that the data does not hold gets no check.
    ionpot_ground_ev = _get_nist_ionisation_energies_ev_cached().get((Z, ion_stage))
    if ionpot_ground_ev is not None and level_energy_ev >= ionpot_ground_ev:
        msg = (
            f"level_energy_ev ({level_energy_ev} eV) must be less than the ionisation potential of Z={Z}"
            f" ion_stage {ion_stage} ({ionpot_ground_ev:.3f} eV)"
        )
        raise ValueError(msg)


def _get_auger_electron_energy_ev(
    Z: int,
    ion_stage: int,
    ionpot_ev: float,
    n_ejected: int,
    auger_electron_energy_ev: float | None,
    level_energy_ev: float,
) -> tuple[float, str | None]:
    # the total energy [eV] of the Auger electrons of one ionisation of a channel, and the message of a
    # warning or None. The ion must keep at least the sum of the NIST ground-state potentials that the
    # ionisation crosses, less the energy of the initial level above the ground state. Else the channel
    # makes energy. Without a value from the caller, energy conservation gives the energy: the
    # threshold of the channel minus that sum. The caller gives the warning after its last check, so
    # that a rejected call gives only its error.
    ionpots_ev = _get_nist_ionisation_energies_ev_cached()
    stages = range(ion_stage, ion_stage + n_ejected)
    missing = [stage for stage in stages if (Z, stage) not in ionpots_ev]
    if missing:
        if auger_electron_energy_ev is not None:
            # the value of the caller is the only source for an ion that the NIST data does not hold
            return auger_electron_energy_ev, None
        msg = (
            f"the NIST data has no ionisation potential for Z={Z} ion_stages {missing}, so energy"
            " conservation cannot give the energy of the Auger electrons. Give auger_electron_energy_ev."
        )
        raise ValueError(msg)
    nist_sum_ev = sum(ionpots_ev[(Z, stage)] for stage in stages)
    # _check_level_energy_ev() keeps level_energy_ev below the first potential, so this is positive
    retained_ev = nist_sum_ev - level_energy_ev
    retained_min_ev = retained_ev * (1.0 - MULTIPLE_IONPOT_REL_TOL)
    auger_ev = max(0.0, ionpot_ev - retained_ev) if auger_electron_energy_ev is None else auger_electron_energy_ev
    if ionpot_ev - auger_ev >= retained_min_ev:
        return auger_ev, None
    channel_str = f"a channel that takes Z={Z} ion_stage {ion_stage} to ion_stage {ion_stage + n_ejected}"
    keep_str = (
        f"The ion must keep at least {retained_min_ev:.3f} eV. That is the sum of the ground-state ionisation"
        f" potentials ({nist_sum_ev:.3f} eV), less level_energy_ev ({level_energy_ev} eV), less"
        " MULTIPLE_IONPOT_REL_TOL."
    )
    if ionpot_ev < retained_min_ev:
        advice = f"Set ionpot_ev to at least {retained_ev:.3f} eV."
    else:
        # round the limit down to three decimals. The product can round up, so the printed value is
        # checked again until it passes. Each step is at least two units in the last place. A step of
        # 0.001 eV alone does not change a value above 2^44 eV, and then the loop did not stop.
        auger_max_ev = math.floor((ionpot_ev - retained_min_ev) * 1000.0) / 1000.0
        step_ev = max(0.001, 2.0 * math.ulp(auger_max_ev))
        while ionpot_ev - float(f"{auger_max_ev:.3f}") < retained_min_ev:
            auger_max_ev -= step_ev
        advice = f"Set auger_electron_energy_ev to at most {auger_max_ev:.3f} eV."
    # The NIST data is a plausibility check, because thresholds from other atomic data can differ from
    # it, and a single direct ionisation has no such check. The channel conserves energy with any
    # value: the ion keeps ionpot_ev - auger_ev as ionisation energy.
    if auger_electron_energy_ev is None:
        msg = (
            f"ionpot_ev ({ionpot_ev} eV) of {channel_str} is less than the energy that the ion must keep."
            f" {keep_str} The Auger electrons get zero energy, and the ion keeps all of ionpot_ev. {advice}"
        )
    else:
        msg = (
            f"with auger_electron_energy_ev={auger_electron_energy_ev} eV, the ion of {channel_str} keeps"
            f" {ionpot_ev - auger_ev:.3f} eV, which is less than the NIST data give. {keep_str} The channel"
            f" uses the value as given. {advice}"
        )
    return auger_ev, msg


def _check_zero_below_ionpot(
    arr_enev: npt.NDArray[np.float64], xs_grid: npt.NDArray[np.float64], ionpot_ev: float, name: str
) -> None:
    # the matrix fill starts at the ionisation potential, but the analysis integrates the whole
    # grid. A cross section below the ionisation potential therefore adds to the ionisation
    # fraction without a matching loss term. The energy fractions would then not sum to one, and
    # nothing would say why.
    subthreshold = np.flatnonzero((arr_enev <= ionpot_ev) & (xs_grid > 0.0))
    if len(subthreshold) > 0:
        i = int(subthreshold[-1])
        msg = (
            f"{name} must be zero at and below its ionisation potential of {ionpot_ev} eV,"
            f" but it is {xs_grid[i]} cm^2 at {arr_enev[i]} eV."
            " Give the energy at which the cross section starts."
        )
        raise ValueError(msg)


@dataclass(frozen=True, slots=True, eq=False)
class _GridXs:
    # calculate_N_e() evaluates the cross section between the points of the energy grid, just above
    # the ionisation potential, so a channel given as an array on that grid interpolates it there.
    # The np.where keeps the function at zero below its own ionisation potential, where a straight
    # interpolation from the last grid point below it would give a small positive value. An
    # autoionisation channel keeps its value at the threshold (keep_threshold).
    # A class at module level and not a closure, so that pickle can serialise a solver that holds the channel.
    arr_enev: npt.NDArray[np.float64]
    xs_grid: npt.NDArray[np.float64]
    ionpot_ev: float
    keep_threshold: bool = False

    def __call__(self, en_ev: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        above = en_ev >= self.ionpot_ev if self.keep_threshold else en_ev > self.ionpot_ev
        return np.where(above, np.interp(en_ev, self.arr_enev, self.xs_grid, left=0.0, right=0.0), 0.0)


@dataclass(frozen=True, slots=True, eq=False)
class _ShellXs:
    # the cross section of one shell row of the table. A class at module level and not a closure,
    # so that pickle can serialise a solver that holds the channel. The class keeps a read-only view of
    # a copy of the row. Else a write to the row changes xs and not xs_grid of the channel.
    shell: Mapping[str, t.Any]
    lotz_a_cm2_ev2: float
    # the fraction of the ionisations of the shell that the channel holds
    scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "shell", MappingProxyType(dict(self.shell)))

    @t.override
    def __reduce__(self) -> tuple[type[t.Self], tuple[dict[str, t.Any], float, float]]:
        # pickle cannot serialise a read-only view, so it gets a copy of the row as a dict
        return (type(self), (dict(self.shell), self.lotz_a_cm2_ev2, self.scale))

    def __call__(self, arr_enev: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        xs = get_arxs_array_shell(arr_enev, self.shell, lotz_a_cm2_ev2=self.lotz_a_cm2_ev2)
        # a scale of one gives the array of the shell itself, so the default results do not change
        return xs if self.scale == 1.0 else self.scale * xs


def get_ion_ionisation_channels(
    dfcollion: pl.DataFrame,
    Z: int,
    ion_stage: int,
    arr_enev: npt.NDArray[np.float64],
    lotz_a_cm2_ev2: float = LOTZ_A_CM2_EV2,
    *,
    use_km1993_auger: bool = False,
    max_auger_electrons: int = KM93_MAX_AUGER_ELECTRONS,
) -> list[IonisationChannel]:
    """Get the ionisation channels of every subshell of an ion in the fit table dfcollion.

    arr_enev:
        the energy grid [eV] on which to evaluate and check each cross section
    lotz_a_cm2_ev2:
        the constant A [cm^2 eV^2] of the Lotz formula for the shells without fit data
    use_km1993_auger:
        False (the default) gives one channel with n_ejected=1 for each subshell. True uses the
        Auger data of KM93 (read_auger_data()) in the method of ARTIS (S20). A subshell with
        Auger data then gives one channel for each possible number of Auger electrons a. That
        channel has n_ejected = 1 + a, and its cross section is the cross section of the subshell
        times the probability of a. Each of these channels has mean_auger_electron=True and the
        mean Auger energy of the subshell. A Lotz shell has no Auger data.
    max_auger_electrons:
        the largest number of Auger electrons of one ionisation. The probabilities of the higher
        numbers go to this number. The number of electrons of the ion is also a limit.
    """
    shells = dfcollion.filter((pl.col("Z") == Z) & (pl.col("ion_stage") == ion_stage)).to_dicts()
    if not shells:
        msg = f"No ionisation cross-section data for Z={Z} ion_stage {ion_stage}"
        raise ValueError(msg)

    auger_data = _read_auger_data_cached() if use_km1993_auger else {}
    # an ion cannot lose more electrons than it has
    max_auger_electrons = min(max_auger_electrons, Z - ion_stage)
    channels: list[IonisationChannel] = []
    for shell in shells:
        ionpot_ev = float(shell["ionpot_ev"])
        lotz = int(shell["n"]) < 0
        # the key names the subshell, so the verbose output needs no separate label
        shellkey = (
            f"Lotz shell {SUBSHELLNAMES[-int(shell['l'])]}" if lotz else f"n {int(shell['n'])} l {int(shell['l'])}"
        )
        auger = None if lotz else auger_data.get((Z, ion_stage, int(shell["n"]), int(shell["l"])))
        if auger is None:
            channels.append(
                IonisationChannel.from_xs(
                    arr_enev=arr_enev,
                    Z=Z,
                    ion_stage=ion_stage,
                    ionpot_ev=ionpot_ev,
                    xs=_ShellXs(shell, lotz_a_cm2_ev2),
                    key=shellkey,
                    lotz=lotz,
                )
            )
            continue

        en_auger_ev, probabilities = auger
        if not en_auger_ev < ionpot_ev:
            msg = (
                f"the mean Auger energy ({en_auger_ev} eV) of Z={Z} ion_stage {ion_stage} {shellkey} must be"
                f" less than the ionisation potential of the subshell ({ionpot_ev} eV)"
            )
            raise ValueError(msg)
        # the probabilities of the numbers above the limit go to the limit, as in ARTIS. The table
        # gives four decimals, so the sum can differ from one by 1e-4. The division keeps the total
        # cross section of the subshell.
        folded = [*probabilities[:max_auger_electrons], sum(probabilities[max_auger_electrons:])]
        for n_auger, probability in enumerate(folded):
            if probability <= 0.0:
                continue
            # from_xs() makes a checked channel of a single ionisation, which has no Auger electrons.
            # The Auger data then replaces three of its fields. The checks of from_xs() for a multiple
            # ionisation compare the threshold with the NIST potentials. Some subshells of KM93 fail
            # that check, and ARTIS uses their data as given.
            channels.append(
                dataclasses.replace(
                    IonisationChannel.from_xs(
                        arr_enev=arr_enev,
                        Z=Z,
                        ion_stage=ion_stage,
                        ionpot_ev=ionpot_ev,
                        xs=_ShellXs(shell, lotz_a_cm2_ev2, scale=probability / sum(folded)),
                        key=f"{shellkey}, n_ejected {1 + n_auger}",
                    ),
                    n_ejected=1 + n_auger,
                    auger_electron_energy_ev=en_auger_ev,
                    mean_auger_electron=True,
                )
            )
    return channels
