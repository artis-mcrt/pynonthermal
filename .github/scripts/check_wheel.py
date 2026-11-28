"""Check the data and results of an installed wheel outside the source directory."""  # noqa: INP001

import math

import pynonthermal

sf = pynonthermal.SpencerFanoSolver()
sf.set_temperature(6000.0)
sf.add_element(Z=8, ion_densities={2: 1e8}, excitation=True)
sf.solve(deposition_ev_per_s_per_cm3=1e8)

# A positive excitation fraction shows that the wheel contains the excitation data.
fractions = (sf.get_frac_heating(), sf.get_frac_ionisation_tot(), sf.get_frac_excitation_tot())
assert all(fraction > 0.0 for fraction in fractions)
# The energy sum is one to rounding on each grid, also with an error in one channel. Thus the sum does not
# show an error in the data of the wheel.
assert math.isclose(sf.get_frac_sum(), 1.0, abs_tol=1e-9)
# The reference excitation fraction is for the default grid of 4096 points. A grid of 200 points gives 0.09545,
# so the tolerance is much smaller than the grid error of a coarse grid.
assert math.isclose(sf.get_frac_excitation_tot(), 0.093245, rel_tol=1e-3)

# The chained comparisons also reject nan.
assert 0.0 < sf.get_ionisation_ratecoeff(8, 2) < math.inf
assert all(0.0 <= sf.get_excitation_ratecoeff(8, 2, key) < math.inf for key in sf.excitationlists[(8, 2)])

# The wheel contains the Auger data of the built-in channels.
sf = pynonthermal.SpencerFanoSolver(use_km1993_auger=True)
sf.add_element(Z=26, ion_densities={2: 1e5})
sf.solve(deposition_ev_per_s_per_cm3=1e3)
assert sf.get_ionisation_ratecoeff(26, 2, n_ejected=2) > 0.0
assert math.isclose(sf.get_frac_sum(), 1.0, abs_tol=1e-3)
