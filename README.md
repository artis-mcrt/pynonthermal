# pynonthermal
[![DOI](https://zenodo.org/badge/359805556.svg)](https://zenodo.org/badge/latestdoi/359805556)
[![PyPI - Version](https://img.shields.io/pypi/v/pynonthermal)](https://pypi.org/project/pynonthermal)
[![License](https://img.shields.io/github/license/lukeshingles/pynonthermal)](https://github.com/lukeshingles/pynonthermal/blob/main/LICENSE)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/pynonthermal)](https://pypi.org/project/pynonthermal/)
[![Build and test](https://github.com/lukeshingles/pynonthermal/actions/workflows/pytest.yml/badge.svg)](https://github.com/lukeshingles/pynonthermal/actions/workflows/pytest.yml)

pynonthermal is a Python solver for the Spencer-Fano equation, which describes the energy distribution of non-thermal (fast) electrons slowing down in a plasma. When high-energy leptons — such as the Compton, photoelectric, and pair-production electrons and positrons produced by radioactive decay in supernova ejecta — are injected into a partially ionised gas, they lose energy through three competing channels: Coulomb heating of the free thermal electrons, collisional ionisation, and collisional excitation of bound states.

Given a set of ions (with number densities) and an energy deposition rate, pynonthermal computes:

- the **degradation spectrum** y(E) of the non-thermal electron population,
- the **fraction of deposited energy** going to heating, ionisation, and excitation (per channel and per ion),
- **non-thermal ionisation rate coefficients** for each ion and **excitation rate coefficients** for individual bound-bound transitions, ready to be used in non-LTE plasma modelling.

These quantities are important, for example, in modelling the late-time spectra and light curves of Type Ia and core-collapse supernovae, where non-thermal ionisation can dominate over photoionisation. The solver follows the method of [Kozma & Fransson (1992)](https://ui.adsabs.harvard.edu/abs/1992ApJ...390..602K/abstract) (see [Method background](#method-background) for details and further references) and ships with the atomic data needed to run out of the box: ionisation cross sections for a wide range of ions, and level/transition data for bound-bound excitation.

## Contents
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage guide](#usage-guide)
- [Where the ion populations come from](#where-the-ion-populations-come-from)
- [Complete example: pure-oxygen plasma](#complete-example-pure-oxygen-plasma)
- [Units and conventions](#units-and-conventions)
- [Method background](#method-background)
- [Cross-section datasets](#cross-section-datasets)
- [Advanced usage: custom cross sections](#advanced-usage-custom-cross-sections)
- [Citing pynonthermal](#citing-pynonthermal)
- [License](#license)

## Installation

Released package (recommended for most users):

```sh
pip install pynonthermal
```

Development install with [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/lukeshingles/pynonthermal.git
cd pynonthermal
uv sync --frozen
source ./.venv/bin/activate
uv pip install --editable .
prek install
```

Run the test suite with:

```sh
uv run -- python3 -m pytest
```

## Quick start

Make a solver, add the elements of the gas, solve, then read the results:

```python
import pynonthermal

with pynonthermal.SpencerFanoSolver() as sf:
    # O II (ion_stage=2, i.e. charge +1) at a number density of 1e8 cm^-3
    sf.add_element(8, ion_densities={2: 1.0e8})

    sf.solve(deposition_ev_per_s_per_cm3=1.0e8)  # the rate of energy deposition per volume

    print("heating fraction:", sf.get_frac_heating())
    print("ionisation fraction:", sf.get_frac_ionisation_tot())
    print("excitation fraction:", sf.get_frac_excitation_tot())
    print("sum of fractions:", sf.get_frac_sum())
    print("ionisation rate coeff [s^-1]:", sf.get_ionisation_ratecoeff(Z=8, ion_stage=2))
```

The solver is a builder: each call adds ions, channels, or transitions to the matrix, and `solve()`
solves it. The context manager is optional; it closes the figures that the plot methods open.

The [quickstart notebook](https://github.com/lukeshingles/pynonthermal/blob/main/quickstart.ipynb) contains a fuller worked example, and can be launched on Binder:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lukeshingles/pynonthermal/HEAD?filepath=quickstart.ipynb)

## Usage guide

### 1. Make the solver

```python
sf = pynonthermal.SpencerFanoSolver(emin_ev=0.1, emax_ev=16000.0, npts=4096)
```

- `emin_ev`, `emax_ev`: the bounds of the uniform energy grid in eV (defaults `0.1` and `16000.0`). An
  electron that degrades below `emin_ev` is taken to have thermalised, so its energy counts as heating.
  Every ionisation potential must lie above `emin_ev`, and a `ValueError` says which lower `emin_ev` to
  use. The default `emax_ev` covers the K-shell ionisation of iron at about 7 keV; lower it to 3000 eV
  to match Kozma & Fransson (1992).
- `npts`: the number of energy grid points (default `4096`). More points cost memory and time; check
  `get_frac_sum()`.
- `verbose`: print the setup, each added channel, and a per-ion, per-shell breakdown.
- `use_ar1985`: use the original Arnaud & Rothenflug (1985) ionisation cross sections
  (see [Cross-section datasets](#cross-section-datasets)).
- `heating_only_approximation`: leave the excitation and ionisation loss terms out of the matrix and solve
  with the heating loss alone. The rates still follow from that approximate solution, so the fractions do
  not sum to one.

### 2. Set the temperature and the atomic data

```python
sf.set_temperature(6000)  # K, for the LTE level populations and the Saha equation
sf.set_atomic_data(use_collstrengths=True, maxnlevelslower=5, maxnlevelsupper=250)
```

The solver has one temperature. Set it before any excitation with LTE level populations and before
`saha_ion_stages`; it is not needed otherwise.

`set_atomic_data()` chooses the level data for the excitations and how to build their cross sections.
`adata_polars` takes your own level/transition table in the format of `artistools.atomic.get_levels()`;
the other options default to the ARTIS values (collision strengths where available, and transitions from
the lowest 5 levels up to the lowest 250). Without this call the solver uses the internal database with
those defaults.

### 3. Add the elements

```python
sf.add_element(8, 1.0e10, ion_fractions={1: 0.99, 2: 0.01}, excitation=True)
sf.add_element(26, 1.0e6, saha_ion_stages=[1, 2, 3])
```

`add_element()` takes:

- `Z`: the atomic number, and `n_elem`: the number density of the element in cm^-3, summed over its ion
  stages. Every rule needs `n_elem` except `ion_densities`, which gives the densities themselves.
- the population rule: exactly one of `ion_densities`, `ion_fractions`, `saha_ion_stages`, or
  `recomb_ratecoeffs` — see [the next section](#where-the-ion-populations-come-from).
- `excitation`: also add the bound-bound excitations of every ion stage that has level data, with LTE
  level populations at the temperature of the solver. Every stage gets the built-in ionisation cross
  sections either way.
- `builtin_channels`: set it to `False` to leave the built-in ionisation cross sections out and give
  every channel yourself (see [custom cross sections](#advanced-usage-custom-cross-sections)).

To add one ion at a time instead, call `add_ionisation(Z, ion_stage, n_ion)` for the built-in shells of
that ion, and `add_ion_excitation(Z, ion_stage, n_ion)` for its bound-bound excitations with LTE level
populations. Both take `n_ion=None` for an ion whose population the solver already holds.

### 4. Solve

```python
sf.solve(deposition_ev_per_s_per_cm3=1.0e8)
```

- `deposition_ev_per_s_per_cm3`: the rate of energy deposition per volume in eV s^-1 cm^-3 (positive and
  finite). With fixed populations the energy *fractions* do not depend on it and the *rate coefficients*
  scale linearly with it; with `recomb_ratecoeffs` the populations depend on it too.
- `override_n_e`: a free electron density in cm^-3, in place of the one that the ion charges give. It
  cannot be combined with `saha_ion_stages` or `recomb_ratecoeffs`, which set it through charge
  neutrality.
- `balance_tol`: the relative tolerance of the population ratios of an element with
  `recomb_ratecoeffs` (default `1e-4`).

### 5. Read the results

Energy fractions, as shares of the deposited energy:

```python
sf.get_frac_heating()  # to heating of the thermal electrons
sf.get_frac_ionisation_tot()  # to ionisation, over all ions
sf.get_frac_excitation_tot()  # to excitation, over all ions
sf.get_frac_sum()  # the sum; ~1.0 when the grid resolves every channel
sf.get_frac_ionisation_ion(Z, ion_stage)  # one ion's share
sf.get_frac_excitation_ion(Z, ion_stage)
```

Populations, densities, and rates:

```python
sf.get_ion_fractions(Z)  # {ion_stage: fraction of the element}
sf.ionpopdict  # {(Z, ion_stage): number density [cm^-3]}
sf.get_n_e()  # free (thermal) electron density [cm^-3]
sf.get_n_e_nt()  # non-thermal electron density [cm^-3]
sf.get_n_ion_tot()  # total nuclei [cm^-3]

sf.get_ionisation_ratecoeff(Z, ion_stage)  # [s^-1]
sf.get_excitation_ratecoeff(Z, ion_stage, transitionkey)  # [s^-1]
sf.excitationlists[(Z, ion_stage)]  # the transitions of that ion, keyed by transition key
sf.get_eff_ionpot(Z, ion_stage)  # effective ionisation potential [eV]
```

Multiply `get_ionisation_ratecoeff()` by the ion number density for ionisations per second per cm^3, and
`get_excitation_ratecoeff()` by the lower level's population density for excitations per second per cm^3.
For the excitations of `add_element(excitation=True)` and of `add_ion_excitation()` the key is
`(lower_level_index, upper_level_index)`, for example `(0, 8)`.

The solution itself is `sf.yvec` over `sf.engrid`, both read-only arrays. With `verbose=True` the
solver prints the per-ion and per-shell breakdown as it analyses the solution.

### 6. Plot the solution

```python
sf.plot_yspectrum()  # degradation spectrum y(E)
sf.plot_channels(xscalelog=True)  # energy going to each channel vs electron energy
sf.plot_spec_channels("channels.pdf")  # both panels in one figure, saved to file
```

Each method shows the figure interactively, or saves it when `outputfilename` is given;
`plot_yspectrum()` and `plot_channels()` also accept a Matplotlib `axis` to draw into an existing figure.

## Where the ion populations come from

Every `add_element()` call gives exactly one of four rules. A future non-LTE rule will be a fifth
keyword.

### ion_densities

```python
sf.add_element(26, ion_densities={2: 3.0e5, 3: 7.0e5})
```

The number density in cm^-3 of each ion stage, keyed by ion stage. `n_elem` is their sum, so do not
give it as well. This is usually what a plasma code already holds.

### ion_fractions

```python
sf.add_element(26, 1.0e6, ion_fractions={2: 0.3, 3: 0.7})
```

The same populations as a share of `n_elem`. The fractions must lie between 0 and 1 and sum to one.

### saha_ion_stages

```python
sf.add_element(8, 1.0e10, saha_ion_stages=[1, 2, 3])
```

At least two contiguous ion stages, whose populations come from the Saha equation. For each pair of
adjacent stages,
`n_{i+1} n_e / n_i = 2 (U_{i+1} / U_i) (2 pi m_e k_B T / h^2)^(3/2) exp(-chi_i / (k_B T))`, with the
temperature `T` of `set_temperature()` and the ionisation potentials `chi_i` from the NIST table. The
partition functions `U_i` come from the LTE level populations of the level data; the built-in data
covers He, O, and Fe. For other elements give them as `add_element(..., partfuncs={ion_stage: U, ...})`,
or supply a level table as `set_atomic_data(adata_polars=...)`. The bare nucleus (`ion_stage = Z + 1`)
has a partition function of 1. The free electron density follows from charge neutrality in one pass.

### recomb_ratecoeffs

```python
sf.add_element(8, 1.0e10, recomb_ratecoeffs={2: 3.0e-13, 3: 3.0e-12, 4: 1.0e-11})
```

The recombination rate coefficients in cm^3 s^-1, keyed by the ion stage that recombines. For each
pair of adjacent stages `i` and `i+1` the balance is `n_i Gamma_i = n_{i+1} n_e alpha_{i+1}`, where
`Gamma_i` is the non-thermal ionisation rate coefficient of stage `i` from the Spencer-Fano solution
and `alpha_{i+1}` is the coefficient you give. The chain runs from one below the lowest key to the
highest key, so the example is O I to O IV.

The solution depends on the ion densities, so `solve()` iterates: it solves the equation, updates the
densities from the balance and the free electron density from charge neutrality, and repeats until the
population ratios agree to `balance_tol`. Typical cases converge in about 5 to 10 iterations; a
`RuntimeError` reports a balance that did not converge within 100. `sf.balance_iterations` says how many
it took.

Points to note:

- The balance includes only non-thermal ionisation and the recombination that you give. It does not
  include thermal collisional ionisation, photoionisation, or charge exchange. The ion fractions
  therefore depend on the deposition rate density, unlike the fixed-population case.
- The top stage of the chain is a sink: its ionisation is an energy loss in the matrix, but the ions
  it makes have no stage to go to. A warning is raised if the ionisation rate out of the top stage
  exceeds 1 % of the total ionisation rate of the element, because about that fraction of the element
  then belongs in a higher stage. Extend the chain with a rate coefficient for the next stage.

The functions behind the two balance rules are in `pynonthermal.ionbalance`: `get_saha_factor()`,
`get_ion_fractions()`, `solve_charge_neutral_n_e_ratios()`, and the general root find
`solve_charge_neutral_n_e()`, which takes any charge density function that does not increase with the
free electron density.

The [iron ionisation balance notebook](https://github.com/lukeshingles/pynonthermal/blob/main/fe_ionbalance_sn1a.ipynb) is a worked example: the ion fractions of iron in the core of a Type Ia supernova at 250 days, with the deposition rate from the 56Co decay, a comparison with the Saha equation, and the evolution from 150 to 400 days.

## Complete example: pure-oxygen plasma

This reproduces Figure 2 of Kozma & Fransson (1992): a pure-oxygen plasma with electron fraction
x_e = 0.01, including both ionisation and excitation channels. With `verbose=True` the solver prints its
setup and a per-ion, per-shell breakdown as it runs.

```python
import pynonthermal

n_e = 1e8  # free electron density [cm^-3]
x_e = 1e-2  # ionisation fraction n_OII / (n_OI + n_OII)
n_oxygen = n_e / x_e

# the grid is theirs rather than the default: emin_ev=1 is their low-energy cutoff E_0,
# and emax_ev=3000 is the top of their energy range.
with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=4096, verbose=True) as sf:
    sf.set_temperature(6000)  # K, for the LTE level populations
    sf.add_element(8, n_oxygen, ion_fractions={1: 1 - x_e, 2: x_e}, excitation=True)

    # with fixed ion densities, any positive deposition rate works here: the energy fractions
    # are independent of it (with recomb_ratecoeffs they would not be).
    sf.solve(deposition_ev_per_s_per_cm3=2950.49 * n_oxygen)

    sf.plot_channels(xscalelog=True)
```

The resulting plot shows the energy distribution of contributions to ionisation, excitation, and heating; the area under each curve gives the fraction of deposited energy in that channel:

![Energy deposition channels for a pure oxygen plasma](https://raw.githubusercontent.com/lukeshingles/pynonthermal/main/docs/oxygen_channels.svg)

## Units and conventions

- Energies are in eV.
- Number densities are in cm^-3.
- Cross sections are in cm^2.
- `ion_stage = charge + 1` (for example, Fe I has `ion_stage=1`, Fe II has `ion_stage=2`).
- `deposition_ev_per_s_per_cm3` is in eV s^-1 cm^-3.
- `get_ionisation_ratecoeff()` and `get_excitation_ratecoeff()` both return rates in s^-1.
- The `recomb_ratecoeffs` of `add_element()` are in cm^3 s^-1, keyed by the ion stage that recombines.

## Method background

The numerical solver is similar to the Spencer-Fano implementation in the [ARTIS](https://github.com/artis-mcrt/artis) radiative transfer code ([Shingles et al. 2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.492.2029S/abstract)), itself an independent implementation of [Kozma and Fransson (1992, ApJ, 390, 602)](https://ui.adsabs.harvard.edu/abs/1992ApJ...390..602K/abstract), based on the electron slowing-down equation of [Spencer and Fano (1954, Phys. Rev., 93, 1172)](https://ui.adsabs.harvard.edu/abs/1954PhRv...93.1172S/abstract). A similar approach is used in [CMFGEN](https://kookaburra.phyast.pitt.edu/hillier/web/CMFGEN.htm).

The integral form of the Kozma and Fransson degradation equation (their equation 7) is discretised on a uniform energy grid as an upper-triangular matrix equation and solved by back-substitution from the highest energy downward. The `SpencerFanoSolver` class docstring maps each term of the equation to the method that implements it, and the code comments cite the specific Kozma and Fransson equations at each site. The secondary-electron energy distribution follows [Opal, Peterson and Beaty (1971)](https://ui.adsabs.harvard.edu/abs/1971JChPh..55.4100O/abstract) as applied by Kozma and Fransson, and the energy loss rate to thermal electrons uses their Coulomb-logarithm prescription (after [Schunk and Hays 1971](https://ui.adsabs.harvard.edu/abs/1971P%26SS...19..113S/abstract)).

If internal level/transition data are used (for example, via `add_ion_excitation()`), they are imported from the CMFGEN atomic data compilation (see the source data files for references), with excitation cross sections computed from the tabulated collision strengths ([Li, Dessart and Hillier 2012, equation 11](https://doi.org/10.1111/j.1365-2966.2012.21198.x)) or, for permitted transitions without one, from the oscillator strength via the van Regemorter (1962) approximation with the g-bar factor of [Mewe (1972)](https://ui.adsabs.harvard.edu/abs/1972A%26A....20..215M/abstract), as described in [Shingles et al. (2020, section 2.5)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.492.2029S/abstract).

## Cross-section datasets

Ionization cross sections from H (Z=1) to Ni (Z=28) use the shell-resolved analytical fits compiled by [Arnaud and Rothenflug (1985, A&AS, 60, 425)](https://ui.adsabs.harvard.edu/abs/1985A%26AS...60..425A/abstract), with updates to Fe from [Arnaud and Raymond (1992, ApJ, 398, 394)](https://ui.adsabs.harvard.edu/abs/1992ApJ...398..394A/abstract). For heavier elements (Z>28) and any other ions missing from the fit data, the approximation of [Axelrod (1980, PhD thesis, Eq. 3.38)](https://ui.adsabs.harvard.edu/abs/1980PhDT.........1A/abstract) is used — the high-energy limit of the [Lotz (1967, Z. Phys., 206, 205)](https://doi.org/10.1007/BF01325928) formula with relativistic corrections — with subshell binding energies from [Lotz (1970, J. Opt. Soc. Am., 60, 206)](https://doi.org/10.1364/JOSA.60.000206).

Passing `use_ar1985=True` to the solver selects the original Arnaud and Rothenflug (1985) compilation without the Fe updates, which can be useful for comparison with older published results.

## Advanced usage: custom cross sections

Give a cross section as a function of an array of energies in eV that returns cross sections in cm^2.
The solver calls it on its own grid, and between the grid points where it needs to, so the plasma does
not depend on the energy grid of the solution. An array at every energy of `sf.engrid` is accepted too,
but then the plasma is tied to that grid and the solver can only interpolate between the points.

A custom cross section follows the same path through the solver as a built-in one, so the matrix, the
energy fractions, and the rate coefficients stay consistent.

```python
import numpy as np
import pynonthermal


def my_ionisation_xs(en_ev):
    return np.interp(en_ev, my_en_ev, my_xs_cm2, left=0.0, right=0.0)


with pynonthermal.SpencerFanoSolver() as sf:
    sf.add_element(8, ion_densities={2: 1.0e8})
    # this keeps the built-in shells and adds one channel; builtin_channels=False replaces them
    sf.add_ionisation_channel(8, 2, None, ionpot_ev=35.0, xs_vec=my_ionisation_xs, channelkey="mine")
    sf.add_excitation(8, 2, None, my_excitation_xs, epsilon_trans_ev=20.0, transitionkey=(0, 3), levelpopfrac=0.9)
```

`add_ionisation_channel()` takes:

- `Z` and `ion_stage`: the ion that the channel ionises, and `n_ion`: its number density in cm^-3, or
  `None` for an ion whose population the solver already holds.
- `ionpot_ev`: the ionisation potential in eV. It must lie between `emin_ev` and `emax_ev`, and the cross
  section must be zero at and below it. Any value in that range is allowed, so a channel need not be a
  subshell of the built-in table; a total ionisation cross section for the ion works too.
- `xs_vec`: the cross section, non-negative and finite.
- `channelkey`: any key that is unique within the ion, for the verbose output.

`add_excitation()` takes:

- `Z`, `ion_stage`, and `levelnumberdensity`: the population of the lower level in cm^-3. Give `None`
  instead, with `levelpopfrac` between 0 and 1, for an ion whose population the solver already holds. The
  level population then follows the ion population, whether it is fixed or comes from a balance.
- `epsilon_trans_ev`: the transition energy in eV. It must be positive and no greater than `emax_ev`,
  since no electron the solver represents could otherwise drive the transition. Transitions below
  `emin_ev` are allowed here, but `add_element(excitation=True)` drops them: Kozma and Fransson (1992)
  take every electron below `emin_ev` to have thermalised, so that energy is accounted for as heating
  instead.
- `xs_vec`, and `transitionkey`: the key to pass to `get_excitation_ratecoeff()`.

`sf.calculate_N_e()` integrates over a domain just above the ionisation potential that is narrower than
one grid cell. A cross section given as a function is called there; an array can only be interpolated, so
resolve that region with `npts` if it matters for your ion. The term it feeds is the energy that
thermalises below `emin_ev`, which is a small part of the heating fraction.

The solver keeps the Lorentzian secondary-electron distribution of Kozma and Fransson (1992, equation 4),
whose width comes from `pynonthermal.collion.get_J()`. The matrix fill integrates that distribution
analytically, so its shape is not adjustable.

## Citing pynonthermal

If you use pynonthermal, please cite it via the [Zenodo record](https://zenodo.org/badge/latestdoi/359805556). Please also consider citing the papers describing the method: [Kozma and Fransson (1992)](https://ui.adsabs.harvard.edu/abs/1992ApJ...390..602K/abstract) and [Shingles et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.492.2029S/abstract).

## License

Distributed under the MIT license. See [LICENSE](LICENSE) for details.
