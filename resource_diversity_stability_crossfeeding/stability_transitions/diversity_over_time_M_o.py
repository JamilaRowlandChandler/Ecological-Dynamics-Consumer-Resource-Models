# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 2026

@author: jamil

Analyse how M and influx o affect consumer diversity OVER TIME, for
single-resource vs all-resources-supplied injection, using the
growth_saturation=True (R_alpha**2/(R_alpha+R_beta) saturating flux) MP_CRM
variant. p_s = min(5/M, 1) throughout, so the dominant resource's out-degree
stays roughly constant across M. d=0.1, mu_C=40 (mu_c=mu_C/M), S=50,
sigma_C=1.6 - all matching the earlier coexistence investigations.

Run as `python diversity_over_time_M_o.py <M>` to run just one M value's
worth of simulations (80 runs) and save to diversity_over_time_M_o_M<M>.pkl.
This is deliberately split into one process per M - running the full
4*80=320-simulation sweep in a single long-lived Python process was found to
hang after ~240 accumulated solve_ivp() calls (a case that runs in ~3s in
isolation stalls for minutes+ when it's the 242nd call in the same process -
likely memory fragmentation or internal LSODA/NumPy state buildup, not a
model bug). Splitting by M keeps each process to 80 calls, safely under that
threshold. merge_diversity_over_time_M_o.py combines the per-M pickles
afterwards.
"""

import numpy as np
import sys
import os
import pickle

# %%

abspath = os.path.abspath(__file__)
file_directory_name = os.path.dirname(abspath)
os.chdir(file_directory_name)

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/consumer_resource_modules')
from models import Consumer_Resource_Model

# %%

mu_C = 40
sigma_C = 1.6
d = 0.1
S = 50
t_end = 7000
no_communities = 8

o_values = [1, 0.3, 0.1, 0.03, 0.01]

M = int(sys.argv[1])

mu_c = mu_C / M
sigma_c = sigma_C / np.sqrt(M)
p_s = min(5 / M, 1)

results = {}

for condition in ['single', 'all']:

    for o_val in o_values:

        key = (M, condition, o_val)
        results[key] = []

        for c in range(no_communities):

            np.random.seed(10000 * M + 1000 * int(o_val * 1000) + c)

            w = np.random.uniform(0, 1, M)

            if condition == 'single':
                o = np.zeros(M)
                o[np.argmax(w)] = o_val
            else:
                o = np.full(M, o_val)

            community = Consumer_Resource_Model('Metabolic pathways', pool_sizes=(M, S))
            community.growth_consumption_rates('growth function of consumption',
                                               mu_c=mu_c, sigma_c=sigma_c, mu_g=1, sigma_g=0)
            community.model_specific_rates(death_args={'d': d},
                                           influx_method='user-supplied', influx_args={'o': o},
                                           resource_growth_args={'b': -1},
                                           resource_inhibition_args={'A': 0})
            community.metabolic_network(energies=w, network_method='step',
                                        resource_conversions={'p_s': p_s},
                                        growth_saturation=True,
                                        production_method='constant', production_args={'p': 1})

            community.simulate_community(t_end=t_end, no_init_cond=1)
            sol = community.ODE_sols[0]

            species_traj = sol.y[M:, :]
            survival_frac_t = np.mean(species_traj > 1e-4, axis=0)

            results[key].append({'t': sol.t, 'survival_fraction': survival_frac_t})

        final_fracs = [r['survival_fraction'][-1] for r in results[key]]
        print(f"M={M}, {condition}, o={o_val}: mean final survival fraction = "
              f"{np.mean(final_fracs):.3f} (n_species = {np.mean(final_fracs)*S:.1f})", flush=True)

with open(f'diversity_over_time_M_o_M{M}.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"Saved results to diversity_over_time_M_o_M{M}.pkl")
