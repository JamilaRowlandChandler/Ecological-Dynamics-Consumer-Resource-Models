# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 2026

@author: jamil

Repeat of diversity_low_M_investigation_o_sweep.py's design (one fixed
metabolic network + fixed community rates, o swept per community), but with
M=10 fixed (not swept), only ONE metabolic network sampled (not 3 resamples),
and the resource self-decay rate b set to (near) 0 instead of -1 - checking
whether removing the explicit decay term causes unbounded growth given the
metabolic network's production efficiency p=1 ("perfect recycling").

Smoke-testing found that b=0 exactly, and even b=-1e-5 or p=1-1e-5 (both
tried first, per the brief), are NOT enough - resource/species abundances
grow roughly linearly without bound out to t_end=7000 (confirmed via
max_abs_y still climbing steadily at t_end, not plateauing) because a
constant external supply o has essentially no loss channel to balance it
once b~0 and p~1 (species death is the only other sink, and isn't enough on
its own at this o range). b=-0.001 was the smallest decay rate found to
robustly reach a genuine steady state (max_abs_y plateaus well before t_end)
across the full o range and both conditions - still two orders of magnitude
smaller than the b=-1 used elsewhere in this investigation, so much closer
to "no dilution" while avoiding runaway growth.

For M=10 and each condition:
  - the metabolic network (w, q, shared across all consumers) is sampled ONCE
  - no_communities=20 independent sets of growth/consumption rates are
    sampled (network held fixed)
  - for each of those 20 communities, every o in o_values is simulated with
    that same community, varying only the influx vector

M=10, p_s=min(5/M,1)=0.5, K_m=1e-2, d=0.1, mu_C=40 (mu_c=mu_C/M), S=50,
sigma_C=1.6, b=-0.001, p=1, o=[0.1,0.3,0.5,0.7,0.9,1.1].
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

    # ONE metabolic network, shared across both conditions and every o/community
    network_seed = 900000
    w, adjacency = sample_shared_network(M, p_s, network_seed)

    for cond_idx, condition in enumerate(['single', 'all']):

        for community_idx in range(no_communities):

            # one seed per community, shared across every o value in the
            # sweep below, so growth/consumption rates (c_ia, y_ia) stay
            # fixed while only o varies
            rate_seed = 950000 + cond_idx * 10000 + community_idx

            n_timed_out = 0
            max_abs_y_seen = 0.0

            for o_val in o_values:

                key = (condition, community_idx, o_val)

                result = simulate_with_timeout(M=M, S=S, mu_c=mu_c, sigma_c=sigma_c,
                                               d=d, p_s=p_s, K_m=K_m, condition=condition,
                                               o_val=o_val, seed=rate_seed, w=w,
                                               adjacency=adjacency, t_end=t_end,
                                               timeout=timeout, b=b, p=p)

                if result['timed_out']:
                    n_timed_out += 1
                elif result['max_abs_y'] is not None:
                    max_abs_y_seen = max(max_abs_y_seen, result['max_abs_y'])

                results[key] = result

            if n_timed_out > 0:
                print(f"TIMEOUT_ALERT: {condition}, community={community_idx}: "
                      f"{n_timed_out}/{len(o_values)} o-values timed out", flush=True)

            if max_abs_y_seen > 1e5:
                print(f"UNBOUNDED_GROWTH_ALERT: {condition}, community={community_idx}: "
                      f"max_abs_y={max_abs_y_seen:.2e}", flush=True)

        print(f"{condition}: {no_communities} communities done", flush=True)

    out_path = os.path.join(DATA_DIR, 'diversity_M10_one_network_b0_results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Saved results to {out_path}")
