# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 2026

@author: jamil

MP_CRM (growth_saturation=True) with a single resource-injection point:
w sampled Uniform(0,1,M) as usual, but influx o is zero everywhere except
the highest-energy resource, which receives o_val. Resources otherwise
decay linearly (b=-1, A=0) rather than growing logistically. Tracks how
the consumer (species) survival fraction evolves over the course of the
simulation, for 5 communities per o_val, across
o_val in [1, 0.5, 0.3, 0.1, 0.01].

Assumptions (not explicitly specified by the request, chosen to match this
session's established conventions - flagged here for visibility):
    - S = M (gamma = 1)
    - mu_c = 100 is read as the unscaled "mu_C" convention used throughout
      this session (mu_c actual = mu_C/M), giving mu_c = 100/50 = 2
    - sigma_c uses the same sigma_C = 1.6 convention as recent sweeps
      (sigma_c = 1.6/sqrt(M))
    - mu_y = 1 (only sigma_y = 0 was specified)
    - d = 1, p = 1, network_method='step' with p_s=1, gated=True (default)
"""

import numpy as np
import sys
import os
import pickle

# %%

abspath = os.path.abspath(__file__)
file_directory_name = os.path.dirname(abspath)
os.chdir(file_directory_name)

DATA_DIR = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability_crossfeeding/influx_species_diversity"

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/consumer_resource_modules')
from models import Consumer_Resource_Model

# %%

M = S = 50
mu_C = 100
mu_c = mu_C / M
sigma_C = 1.6
sigma_c = sigma_C / np.sqrt(M)
mu_y = 1
sigma_y = 0
d = 1
A = 0
b = -1
p = 1

o_values = [1, 0.5, 0.3, 0.1, 0.01]
no_communities = 5
t_end = 7000

results = {}

for o_val in o_values:

    results[o_val] = []

    for c in range(no_communities):

        np.random.seed(2000 * int(o_val * 1000) + c)

        # w sampled the same way as metabolic_network()'s built-in default
        w = np.random.uniform(0, 1, M)

        # influx only into the highest-energy resource
        o = np.zeros(M)
        o[np.argmax(w)] = o_val

        community = Consumer_Resource_Model('Metabolic pathways', pool_sizes=(M, S))
        community.growth_consumption_rates('growth function of consumption',
                                           mu_c=mu_c, sigma_c=sigma_c,
                                           mu_g=mu_y, sigma_g=sigma_y)
        community.model_specific_rates(death_args={'d': d},
                                       influx_method='user-supplied', influx_args={'o': o},
                                       resource_growth_args={'b': b},
                                       resource_inhibition_args={'A': A})
        community.metabolic_network(energies=w, network_method='step',
                                    resource_conversions={'p_s': 1},
                                    growth_saturation=True,
                                    production_method='constant', production_args={'p': p})

        community.simulate_community(t_end=t_end, no_init_cond=1)
        sol = community.ODE_sols[0]

        species_traj = sol.y[M:, :]
        survival_frac_t = np.mean(species_traj > 1e-4, axis=0)

        results[o_val].append({'t': sol.t, 'survival_fraction': survival_frac_t,
                               'status': sol.status, 't_final': sol.t[-1]})

        print(f"o={o_val} community={c}: t_final={sol.t[-1]:.1f}, "
              f"final survival fraction={survival_frac_t[-1]:.3f}")

out_path = os.path.join(DATA_DIR, 'single_source_influx_results.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(results, f)

print(f"Saved results to {out_path}")
