# -*- coding: utf-8 -*-
"""
Smoke test for the new Metabolic Pathway consumer-resource model (MP_CRM).

Run with:
    python test_mp_crm.py
(run from within the consumer_resource_modules folder, or anywhere - the
 sys.path insert below makes sure the local modules are importable either way)
"""

import sys
import os

# make sure this folder is importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

np.random.seed(0)

from models import Consumer_Resource_Model

# %%

M, S = 25, 25

# --- build the model via the wrapper ---
community = Consumer_Resource_Model('Metabolic pathways', pool_sizes=(M, S))

assert community.no_resources == M
assert community.no_species == S

# --- growth/consumption rates (shared machinery from ParametersInterface) ---
community.growth_consumption_rates('coupled by rho',
                                    mu_c=1, sigma_c=0.1,
                                    mu_g=1, sigma_g=0.1,
                                    rho=1)

assert community.growth.shape == (S, M)
assert community.consumption.shape == (M, S)

# --- model-specific rates (death, outflux, resource decay, self-inhibition) ---
community.model_specific_rates(death_args={'d': 0.5},
                                outflux_args={'o': 1},
                                resource_growth_args={'b': 0.1},
                                resource_inhibition_args={'A': 0.05})

assert community.d.shape == (S,)
assert community.o.shape == (M,)
assert community.b.shape == (M,)
assert community.A.shape == (M,)

# --- metabolic network + production rates ---
community.metabolic_network(resource_conversions={'mean': 1, 'variance': 1},
                             production_args={'p': 0.3})

assert community.w.shape == (M,)
assert community.q.shape == (S, M, M)
assert set(np.unique(community.q)).issubset({0, 1})
assert community.p.shape == (S, M)

# --- simulate ---
community.simulate_community(t_end=50, no_init_cond=2)

print("ODE solutions:", [sol.y.shape for sol in community.ODE_sols])
print("q shape:", community.q.shape, " w shape:", community.w.shape, " p shape:", community.p.shape)

for sol in community.ODE_sols:
    assert not np.any(np.isnan(sol.y)), "NaNs found in simulation output"

# --- community properties ---
community.calculate_community_properties()

print("species survival fraction:", community.species_survival_fraction)
print("resource survival fraction:", community.resource_survival_fraction)

# --- also test the user-supplied energies / normal production / user-supplied network options ---
community2 = Consumer_Resource_Model('Metabolic pathways', pool_sizes=(M, S))
community2.growth_consumption_rates('coupled by rho',
                                     mu_c=1, sigma_c=0.1,
                                     mu_g=1, sigma_g=0.1,
                                     rho=0.5)
community2.model_specific_rates()
community2.metabolic_network(energies=np.linspace(0, 1, M),
                              resource_conversions={'mean': 2, 'variance': 0.5},
                              production_method='normal',
                              production_args={'mu': 0.5, 'sigma': 0.1})

assert np.allclose(community2.w, np.linspace(0, 1, M))
assert community2.p.shape == (S, M)

community2.simulate_community(t_end=20, no_init_cond=1)
print("Second community ODE solution shape:", community2.ODE_sols[0].y.shape)

print("\nAll MP_CRM smoke tests passed.")
