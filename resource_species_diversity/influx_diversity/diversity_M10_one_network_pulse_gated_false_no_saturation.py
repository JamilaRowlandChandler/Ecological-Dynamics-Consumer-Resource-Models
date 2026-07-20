# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 2026

@author: jamil

Repeat of diversity_M10_one_network_pulse_gated_false.py (M=10, one
non-gated metabolic network, pulse the highest-energy resource to R_star,
o=0 otherwise), but using growth_saturation=False - the original MP_CRM
model, with the bare R_alpha factor (unsaturated) in growth/consumption/
production, rather than any of the saturating-flux variants. Smoke-tested
stable and fast across the full R_star range at this M=10/S=50/non-gated
scale before this full run.

Uses the SAME network_seed (900000, gated=False) and rate_seed convention as
diversity_M10_one_network_pulse_gated_false.py for direct comparability.

M=10, p_s=min(5/M,1)=0.5, d=0.1, mu_C=40 (mu_c=mu_C/M), S=50, sigma_C=1.6,
b=-0.001, p=1, R_star=[10, 50, 100, 300, 500, 1000], t_end=300. K_m is
unused (only relevant when growth_saturation=True).
"""

import numpy as np
import sys
import os
import pickle

abspath = os.path.abspath(__file__)
file_directory_name = os.path.dirname(abspath)
os.chdir(file_directory_name)

DATA_DIR = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability_crossfeeding/influx_species_diversity"

sys.path.insert(0, file_directory_name)
from timeout_utils import simulate_with_timeout, sample_shared_network

# %%

mu_C = 40
sigma_C = 1.6
d = 0.1
S = 50
t_end = 300
timeout = 60

b = -0.001
p = 1

no_communities = 20

M = 10
mu_c = mu_C / M
sigma_c = sigma_C / np.sqrt(M)
p_s = min(5 / M, 1)

R_star_values = [10, 50, 100, 300, 500, 1000]

if __name__ == '__main__':

    results = {}

    # SAME network_seed/gated=False as diversity_M10_one_network_pulse_gated_false.py
    network_seed = 900000
    w, adjacency = sample_shared_network(M, p_s, network_seed, gated=False)

    for R_idx, R_star in enumerate(R_star_values):

        n_timed_out = 0

        for community_idx in range(no_communities):

            rate_seed = 950000 + 0 * 10000 + community_idx

            key = (R_star, community_idx)

            result = simulate_with_timeout(M=M, S=S, mu_c=mu_c, sigma_c=sigma_c,
                                           d=d, p_s=p_s, K_m=0, condition='single',
                                           o_val=0, seed=rate_seed, w=w,
                                           adjacency=adjacency, t_end=t_end,
                                           timeout=timeout, b=b, p=p, R_star=R_star,
                                           growth_saturation=False)

            if result['timed_out']:
                n_timed_out += 1

            results[key] = result

        if n_timed_out > 0:
            print(f"TIMEOUT_ALERT: R*={R_star}: {n_timed_out}/{no_communities} "
                  f"communities timed out", flush=True)

        print(f"R*={R_star}: {no_communities} communities done", flush=True)

    out_path = os.path.join(DATA_DIR, 'diversity_M10_one_network_pulse_gated_false_no_saturation_results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Saved results to {out_path}")
