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


@dataclasses.dataclass(frozen=True, slots=True)
class Saha:
    """Ion fractions from the Saha equation at the solver temperature (see add_element_saha()).

    ion_stages:
        at least two contiguous ion stages between 1 and Z + 1
    partfuncs:
        partition functions keyed by ion stage, for stages without level data
    """

    ion_stages: Sequence[int]
    partfuncs: Mapping[int, float] | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class IonBalance:
    """Ion fractions from the balance of non-thermal ionisation against recombination.

    See add_element_ionbalance(). solve() iterates until the populations converge.

    recomb_ratecoeffs:
        the recombination rate coefficients in cm^3 s^-1, keyed by the ion stage that recombines.
        The chain of ion stages runs from one below the lowest key to the highest key.
    """

    recomb_ratecoeffs: Mapping[int, float]


PopulationModel = Fixed | Saha | IonBalance
