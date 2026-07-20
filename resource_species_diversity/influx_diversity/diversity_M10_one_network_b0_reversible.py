# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 2026

@author: jamil

Repeat of diversity_M10_one_network_b0.py (M=10, ONE gated metabolic
network, continuous influx o swept, b=-0.001 near-zero dilution, d=0.1),
but using saturation_kinetics='reversible' instead of 'flux' - testing
whether reversible kinetics support genuine steady-state coexistence (more
than 1-2 surviving species) under continuous supply, given the user's
niche-partitioning hypothesis: resources further down the network's energy
cascade can reach HIGHER steady-state levels than the directly-supplied
resource (confirmed in a prior check - even gated, with the supplied
resource having zero in-degree), so different species could in principle
specialise on different resources rather than all competing for the same
one.

Uses the SAME network_seed (900000, gated=True) and rate_seed convention as
diversity_M10_one_network_b0.py for direct comparability. K_m=1e-2, V_max=1
(scalar, not sampled - kept simple for a clean apples-to-apples comparison
against the existing 'flux' result; K_m/V_max sampling can be layered on
top separately if this shows anything interesting).

M=10, p_s=min(5/M,1)=0.5, K_m=1e-2, d=0.1, mu_C=40 (mu_c=mu_C/M), S=50,
sigma_C=1.6, b=-0.001, p=1, o=[0.1,0.3,0.5,0.7,0.9,1.1], condition='single'
only (matching the niche-partitioning discussion's focus on the
single-resource-supplied case).
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
t_end = 7000
K_m = 1e-2
timeout = 60

b = -0.001
p = 1

no_communities = 20

M = 10
mu_c = mu_C / M
sigma_c = sigma_C / np.sqrt(M)
p_s = min(5 / M, 1)

o_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]

if __name__ == '__main__':

    results = {}

    # SAME network as diversity_M10_one_network_b0.py (gated=True)
    network_seed = 900000
    w, adjacency = sample_shared_network(M, p_s, network_seed, gated=True)

    condition = 'single'

    for community_idx in range(no_communities):

        rate_seed = 950000 + community_idx

        n_timed_out = 0
        max_abs_y_seen = 0.0

        for o_val in o_values:

            key = (condition, community_idx, o_val)

            result = simulate_with_timeout(M=M, S=S, mu_c=mu_c, sigma_c=sigma_c,
                                           d=d, p_s=p_s, K_m=K_m, condition=condition,
                                           o_val=o_val, seed=rate_seed, w=w,
                                           adjacency=adjacency, t_end=t_end,
                                           timeout=timeout, b=b, p=p,
                                           growth_saturation=True,
                                           saturation_kinetics='reversible')

            if result['timed_out']:
                n_timed_out += 1
            elif result['max_abs_y'] is not None:
                max_abs_y_seen = max(max_abs_y_seen, result['max_abs_y'])

            results[key] = result

        if n_timed_out > 0:
            print(f"TIMEOUT_ALERT: community={community_idx}: "
                  f"{n_timed_out}/{len(o_values)} o-values timed out", flush=True)

        if max_abs_y_seen > 1e5:
            print(f"UNBOUNDED_GROWTH_ALERT: community={community_idx}: "
                  f"max_abs_y={max_abs_y_seen:.2e}", flush=True)

    print(f"{condition}: {no_communities} communities done", flush=True)

    out_path = os.path.join(DATA_DIR, 'diversity_M10_one_network_b0_reversible_results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Saved results to {out_path}")
