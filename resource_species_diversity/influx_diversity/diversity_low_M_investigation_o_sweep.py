# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 2026

@author: jamil

Third variant of the low-M diversity investigation. Previous variants either
resampled a fresh metabolic network AND fresh growth/consumption rates for
every (M, condition, o) combination (diversity_low_M_investigation.py), or
fixed one network per (M, condition, o) combination but still varied
growth/consumption rates with o (diversity_low_M_investigation_fixed_network.py) -
in both cases, o and the sampled rates were confounded within a given
"community"'s trajectory across the o sweep.

Here, o is swept for a genuinely fixed community: one metabolic network
(w, q, shared across all consumers) and one set of growth/consumption rates
(c_ia, y_ia) are sampled once, then reused identically across every o value
in o_values - isolating the pure effect of increasing influx on a single
fixed community, rather than comparing across independently-sampled
communities at each o.

For each M and condition:
  - the metabolic network is resampled n_network_resamples=3 times
  - for each network, no_communities=20 independent sets of
    growth/consumption rates are sampled (network held fixed)
  - for each of those 20 communities, every o in o_values is simulated with
    that same community (same w, q, c_ia, y_ia), varying only the influx
    vector

M=[5,10,15,20], p_s=min(5/M,1) (scaled with M), K_m=1e-2, d=0.1, mu_C=40
(mu_c=mu_C/M), S=50, sigma_C=1.6, o=[0.1,0.3,0.5,0.7,0.9,1.1].
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

n_network_resamples = 3
no_communities = 20

M_values = [5, 10, 15, 20]
o_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]

if __name__ == '__main__':

    results = {}

    for M_idx, M in enumerate(M_values):

        mu_c = mu_C / M
        sigma_c = sigma_C / np.sqrt(M)
        p_s = min(5 / M, 1)

        for cond_idx, condition in enumerate(['single', 'all']):

            for net_idx in range(n_network_resamples):

                network_seed = 800000 + M_idx * 10000 + cond_idx * 1000 + net_idx
                w, adjacency = sample_shared_network(M, p_s, network_seed)

                for community_idx in range(no_communities):

                    # one seed per community, shared across every o value in
                    # the sweep below, so growth/consumption rates (c_ia,
                    # y_ia) stay fixed while only o varies
                    rate_seed = 850000 + M_idx * 100000 + cond_idx * 10000 \
                        + net_idx * 1000 + community_idx

                    n_timed_out = 0

                    for o_val in o_values:

                        key = (M, condition, net_idx, community_idx, o_val)

                        result = simulate_with_timeout(M=M, S=S, mu_c=mu_c, sigma_c=sigma_c,
                                                       d=d, p_s=p_s, K_m=K_m, condition=condition,
                                                       o_val=o_val, seed=rate_seed, w=w,
                                                       adjacency=adjacency, t_end=t_end,
                                                       timeout=timeout)

                        if result['timed_out']:
                            n_timed_out += 1

                        results[key] = result

                    if n_timed_out > 0:
                        print(f"TIMEOUT_ALERT: M={M}, {condition}, net={net_idx}, "
                              f"community={community_idx}: {n_timed_out}/{len(o_values)} "
                              f"o-values timed out", flush=True)

                print(f"M={M}, {condition}, net={net_idx}: {no_communities} communities done",
                      flush=True)

    out_path = os.path.join(DATA_DIR, 'diversity_low_M_investigation_o_sweep_results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Saved results to {out_path}")
