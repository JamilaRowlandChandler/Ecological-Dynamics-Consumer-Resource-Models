# -*- coding: utf-8 -*-
"""
Smoke test for the pool_sizes / initial-conditions / model_specific_rates
refactor ported from the "Student tutorials" copy of these modules.

Covers: SL_CRM, SL_SI_CRM, ES_CRM, Hybrid_CRM, SL_TL_CRM (incl. the fixed
model_specific_rates dims-ordering bug), SL_CRPM (now also pool_sizes-based),
and the "user-supplied" initial condition path.

Run with:
    python test_pool_sizes_refactor.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

np.random.seed(0)

from models import Consumer_Resource_Model

M, S = 6, 8

# %% ---- SL_CRM (pool_sizes as a tuple) ----

sl_crm = Consumer_Resource_Model('Self-limiting resource supply', pool_sizes=(M, S))
assert sl_crm.no_resources == M and sl_crm.no_species == S
assert sl_crm.pool_sizes == [M, S]

sl_crm.growth_consumption_rates('coupled by rho', mu_c=1, sigma_c=0.1,
                                mu_g=1, sigma_g=0.1, rho=0.5)
sl_crm.model_specific_rates(death_args={'d': 0.5}, resource_growth_args={'b': 1})
sl_crm.simulate_community(t_end=50, no_init_cond=2)
for sol in sl_crm.ODE_sols:
    assert sol.y.shape[0] == M + S
    assert not np.any(np.isnan(sol.y))
sl_crm.calculate_community_properties()
assert len(sl_crm.resource_survival_fraction) == 2
assert len(sl_crm.species_survival_fraction) == 2
print("SL_CRM: OK", [sol.y.shape for sol in sl_crm.ODE_sols])

# %% ---- SL_SI_CRM (pool_sizes as a dict) ----

sl_si_crm = Consumer_Resource_Model('Self-limiting resource supply, self-inhibition',
                                    pool_sizes={'no_resources': M, 'no_species': S})
assert sl_si_crm.no_resources == M and sl_si_crm.no_species == S

sl_si_crm.growth_consumption_rates('coupled by rho', mu_c=1, sigma_c=0.1,
                                   mu_g=1, sigma_g=0.1, rho=0.5)
sl_si_crm.model_specific_rates(death_args={'d': 0.5}, resource_growth_args={'b': 1},
                               si_args={'si': 0.1})
sl_si_crm.simulate_community(t_end=50, no_init_cond=1)
assert not np.any(np.isnan(sl_si_crm.ODE_sols[0].y))
print("SL_SI_CRM (dict pool_sizes): OK", sl_si_crm.ODE_sols[0].y.shape)

# %% ---- ES_CRM ----

es_crm = Consumer_Resource_Model('Externally-supplied resources', pool_sizes=(M, S))
es_crm.growth_consumption_rates('coupled by rho', mu_c=1, sigma_c=0.1,
                                mu_g=1, sigma_g=0.1, rho=0.5)
es_crm.model_specific_rates(death_args={'d': 0.5}, influx_args={'b': 1},
                            outflux_args={'o': 0.5})
es_crm.simulate_community(t_end=50, no_init_cond=1)
assert not np.any(np.isnan(es_crm.ODE_sols[0].y))
print("ES_CRM: OK", es_crm.ODE_sols[0].y.shape)

# %% ---- Hybrid_CRM ----

hybrid_crm = Consumer_Resource_Model('Hybrid resource supply', pool_sizes=(M, S))
hybrid_crm.growth_consumption_rates('coupled by rho', mu_c=1, sigma_c=0.1,
                                    mu_g=1, sigma_g=0.1, rho=0.5)
hybrid_crm.model_specific_rates(death_args={'d': 0.5}, influx_args={'b': 1},
                                outflux_args={'o': 0.1}, resource_inhibition_args={'a': 0.05})
hybrid_crm.simulate_community(t_end=50, no_init_cond=1)
assert not np.any(np.isnan(hybrid_crm.ODE_sols[0].y))
print("Hybrid_CRM: OK", hybrid_crm.ODE_sols[0].y.shape)

# %% ---- SL_TL_CRM (multi-trophic - pool_sizes = [resources, level2, level3]) ----
# deliberately DIFFERENT pool sizes per level, to exercise the fixed
# dims_list/p_labels ordering bug in model_specific_rates (previously this
# would silently misassign death-rate/resource-growth dims across levels
# whenever the trophic levels had different pool sizes).
pool_sizes_tl = [M, S, 4]
sl_tl_crm = Consumer_Resource_Model('Self-limiting resource supply, multi-trophic level',
                                    pool_sizes=pool_sizes_tl)
assert sl_tl_crm.trophic_levels == 3

for tl in [2, 3]:
    sl_tl_crm.growth_consumption_rates('coupled by rho', mu_c=1, sigma_c=0.1,
                                       mu_g=1, sigma_g=0.1, rho=0.5, trophic_level=tl)

sl_tl_crm.model_specific_rates(death_methods=['constant', 'constant'],
                               death_args=[{'d_2': 0.5}, {'d_3': 0.5}],
                               resource_growth_args={'b': 1},
                               resource_interaction_args={'Aij': 0})

# verify each parameter now has the dims matching its OWN pool, not a
# neighbouring level's pool (the bug this test is guarding against)
assert sl_tl_crm.d_2.shape == (S,)
assert sl_tl_crm.d_3.shape == (4,)
assert sl_tl_crm.b.shape == (M,)
assert sl_tl_crm.Aij.shape == (M, M)

sl_tl_crm.simulate_community(t_end=50, no_init_cond=1)
assert not np.any(np.isnan(sl_tl_crm.ODE_sols[0].y))
print("SL_TL_CRM (unequal pool sizes): OK", sl_tl_crm.ODE_sols[0].y.shape)

# %% ---- SL_CRPM ("leached", now also pool_sizes-based, internal species-first
# ODE layout unchanged) ----

sl_crpm = Consumer_Resource_Model('Self-limiting resource supply, leached',
                                  pool_sizes=(M, S))
assert sl_crpm.no_resources == M and sl_crpm.no_species == S
assert sl_crpm.pool_sizes == [S, M, M]

sl_crpm.growth_consumption_rates('coupled by rho', mu_c=1, sigma_c=0.1,
                                 mu_g=1, sigma_g=0.1, rho=0.5)
sl_crpm.model_specific_rates(death_args={'d': 0.5}, resource_growth_args={'b': 1},
                             resource_interaction_args={'Aij': 0})
sl_crpm.simulate_community(t_end=50, no_init_cond=1)
assert not np.any(np.isnan(sl_crpm.ODE_sols[0].y))
sl_crpm.calculate_community_properties()
assert len(sl_crpm.species_survival_fraction) == 1
assert len(sl_crpm.resource_survival_fraction) == 1
print("SL_CRPM (untouched): OK", sl_crpm.ODE_sols[0].y.shape)

# %% ---- user-supplied initial conditions (single wrapped full-length array) ----

flat_ic = np.random.uniform(1e-3, 1, M + S)
user_supplied_sol = sl_crm.simulate_community(t_end=10, no_init_cond=1,
                                              init_cond_func='user-supplied',
                                              assign=False,
                                              initial_conditions=[flat_ic])
assert user_supplied_sol[0].y.shape[0] == M + S
assert np.allclose(user_supplied_sol[0].y[:, 0], flat_ic, atol=1e-2)
print("user-supplied initial conditions: OK", user_supplied_sol[0].y.shape)

print("\nAll pool_sizes refactor smoke tests passed.")
