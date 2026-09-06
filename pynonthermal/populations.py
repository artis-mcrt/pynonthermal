"""Population models for SpencerFanoSolver.add_element().

Each model says how the solver gets the ion populations of an element from its number density:
Fixed takes the ion fractions from the caller, Saha uses the Saha equation at the solver
temperature, and IonBalance balances the non-thermal ionisation rates of the solution against
given recombination rate coefficients. A later model can give the populations from a
collisional-radiative calculation without a change to add_element().
"""

import dataclasses
from collections.abc import Mapping
from collections.abc import Sequence


@dataclasses.dataclass(frozen=True, slots=True)
class Fixed:
    """Ion fractions that the caller gives.

    ion_fractions:
        the fraction of the element in each ion stage, keyed by ion stage. The fractions must be
        non-negative and sum to one. A stage that is not in the mapping has no ions.
    """

    ion_fractions: Mapping[int, float]

    def get_ion_stages(self) -> tuple[int, ...]:
        """Get the ion stages of the model, or an empty tuple if the input is malformed.

        add_element() reads the stages before it validates the model, to build the excitations
        without changing the solver. An empty tuple leaves the report of the fault to add_element().
        """
        try:
            return tuple(sorted(self.ion_fractions))
        except TypeError:
            return ()


@dataclasses.dataclass(frozen=True, slots=True)
class Saha:
    """Ion fractions from the Saha equation at the solver temperature (see add_element()).

    ion_stages:
        at least two contiguous ion stages between 1 and Z + 1
    partfuncs:
        partition functions keyed by ion stage, for stages without level data
    """

    ion_stages: Sequence[int]
    partfuncs: Mapping[int, float] | None = None

    def get_ion_stages(self) -> tuple[int, ...]:
        """Get the ion stages of the model, or an empty tuple if the input is malformed (see Fixed)."""
        try:
            return tuple(int(ion_stage) for ion_stage in self.ion_stages)
        except (TypeError, ValueError):
            return ()


@dataclasses.dataclass(frozen=True, slots=True)
class IonBalance:
    """Ion fractions from the balance of non-thermal ionisation against recombination.

    See add_element(). solve() iterates until the populations converge.

    recomb_ratecoeffs:
        the recombination rate coefficients in cm^3 s^-1, keyed by the ion stage that recombines.
        The chain of ion stages runs from one below the lowest key to the highest key.
    """

    recomb_ratecoeffs: Mapping[int, float]

    def get_ion_stages(self) -> tuple[int, ...]:
        """Get the ion stages of the model, or an empty tuple if the input is malformed (see Fixed)."""
        try:
            upper_stages = sorted(self.recomb_ratecoeffs)
            return tuple(range(upper_stages[0] - 1, upper_stages[-1] + 1))
        except (TypeError, IndexError):
            return ()


PopulationModel = Fixed | Saha | IonBalance
