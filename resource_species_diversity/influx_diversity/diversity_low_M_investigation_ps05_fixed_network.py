# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 2026

@author: jamil

Repeat of diversity_low_M_investigation_ps05.py (p_s FIXED at 0.5 for every
M, instead of scaling as min(5/M,1)), except the metabolic network is no
longer resampled for every community. For each (M, condition, o) parameter
set, a single set of resource energies w and a single metabolic matrix
q_{alpha, beta} (shared across all consumers, via metabolic_network's
adjacency argument) are sampled once with sample_shared_network(), then
reused identically across all no_communities=20 communities in that
parameter set. Only the growth/consumption rates (c_ia, y_ia) are freshly
sampled per community.

M=[5,10,15,20], o=[0.1,0.3,0.5,0.7,0.9,1.1], no_communities=20, p_s=0.5
(fixed), K_m=1e-4, d=0.1, mu_C=40 (mu_c=mu_C/M), S=50, sigma_C=1.6 -
matching diversity_low_M_investigation_ps05.py's settings exactly except for
the network-sharing scheme.
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
no_communities = 20
timeout = 60

M_values = [5, 10, 15, 20]
o_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]

if __name__ == '__main__':

    results = {}

    for M_idx, M in enumerate(M_values):

        mu_c = mu_C / M
        sigma_c = sigma_C / np.sqrt(M)
        p_s = 0.5

        for cond_idx, condition in enumerate(['single', 'all']):

            for o_idx, o_val in enumerate(o_values):

                key = (M, condition, o_val)

                # one shared (w, adjacency) network per parameter set,
                # reused identically across all no_communities communities
                network_seed = 700000 + M_idx * 10000 + cond_idx * 1000 + o_idx
                w, adjacency = sample_shared_network(M, p_s, network_seed)

                results[key] = []
                n_timed_out = 0

                for c in range(no_communities):

                    seed = 10000 * M + 1000 * int(round(o_val * 1000)) + c

                    result = simulate_with_timeout(M=M, S=S, mu_c=mu_c, sigma_c=sigma_c, d=d,
                                                   p_s=p_s, K_m=1e-4, condition=condition,
                                                   o_val=o_val, seed=seed, w=w, adjacency=adjacency,
                                                   t_end=t_end, timeout=timeout)

                    if result['timed_out']:
                        n_timed_out += 1

                    results[key].append(result)

                final_fracs = [r['survival_fraction'][-1] for r in results[key]
                              if not r['timed_out']]

                if len(final_fracs) > 0:
                    mean_sf = np.mean(final_fracs)
                    print(f"M={M}, {condition}, o={o_val}: mean final survival fraction = "
                          f"{mean_sf:.3f} (n_species = {mean_sf*S:.1f}), "
                          f"timed_out={n_timed_out}/{no_communities}", flush=True)
                else:
                    print(f"M={M}, {condition}, o={o_val}: ALL {no_communities} RUNS TIMED OUT",
                          flush=True)

    out_path = os.path.join(DATA_DIR, 'diversity_low_M_investigation_ps05_fixed_network_results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Saved results to {out_path}")
