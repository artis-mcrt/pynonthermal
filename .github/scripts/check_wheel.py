"""Check the data and results of an installed wheel outside the source directory."""  # noqa: INP001

import math

import pynonthermal

sf = pynonthermal.SpencerFanoSolver(npts=2048, emax_ev=3000.0)
sf.set_temperature(6000.0)
sf.add_element(Z=8, ion_densities={2: 1e8}, excitation=True)
sf.solve(deposition_ev_per_s_per_cm3=1e8)

fractions = (
    sf.get_frac_heating(),
    sf.get_frac_ionisation_tot(),
    sf.get_frac_excitation_tot(),
)
assert all(math.isfinite(value) and value > 0.0 for value in fractions)
# This grid gives an energy sum within 0.5% of one for this model.
assert math.isclose(sf.get_frac_sum(), 1.0, abs_tol=0.005)

rate = sf.get_ionisation_ratecoeff(8, 2)
assert math.isfinite(rate)
assert rate > 0.0
transitions = sf.excitationlists[(8, 2)]
assert transitions
rates = [sf.get_excitation_ratecoeff(8, 2, key) for key in transitions]
assert all(math.isfinite(value) and value >= 0.0 for value in rates)
assert any(value > 0.0 for value in rates)
