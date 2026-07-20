# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 2026

@author: jamil

Repeat of diversity_M10_one_network_b0.py's design (M=10, one fixed
metabolic network, b=-0.001, p=1), but instead of a continuous influx o
swept across communities, each community gets a ONE-TIME pulse at t=0:
o=0 everywhere, and the highest-energy resource (argmax(w)) starts at R_star
instead of the usual small Mallmin-range baseline. Every other
resource/species still starts at the usual Mallmin baseline. Reuses the
SAME metabolic network as diversity_M10_one_network_b0.py (network_seed
900000) for direct comparability between the two supply regimes.

Since there's no continued supply, this is a decay experiment - initial
smoke-testing found that with the Mallmin initial condition range
(Uniform(1e-8, 2/S)), MOST of the 50-species pool starts above the 1e-4
survival threshold trivially (an artefact of the initial sampling, not real
diversity), and the real dynamics - resource depletion, then species dying
off at rate d once resources can no longer sustain them - play out and fully
resolve within about 100-200 time units regardless of R_star (species decay
at rate d=0.1 once starved, crossing the 1e-4 threshold from a typical
~0.02 initial abundance in ~46 time units - matching the observed crash
window). t_end=300 was chosen (instead of the 7000 used for the continuous-
supply task) so the fixed 200-point t_eval grid actually resolves this
crash, rather than spending nearly all its points on a trajectory that's
already flatlined at zero.

For each R_star in R_star_values:
  - no_communities=20 independent sets of growth/consumption rates are
    sampled (network held fixed, o=0, only the pulse size R_star and the
    community's own rates vary)

M=10, p_s=min(5/M,1)=0.5, K_m=1e-2, d=0.1, mu_C=40 (mu_c=mu_C/M), S=50,
sigma_C=1.6, b=-0.001, p=1, R_star=[10, 50, 100, 300, 500, 1000].
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
K_m = 1e-2
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

    # SAME metabolic network as diversity_M10_one_network_b0.py
    network_seed = 900000
    w, adjacency = sample_shared_network(M, p_s, network_seed)

    for R_idx, R_star in enumerate(R_star_values):

        n_timed_out = 0

        for community_idx in range(no_communities):

            # same rate-seed convention as diversity_M10_one_network_b0.py's
            # 'single' condition, so growth/consumption rates are drawn
            # identically for the matching community_idx across both scripts
            rate_seed = 950000 + 0 * 10000 + community_idx

            key = (R_star, community_idx)

            result = simulate_with_timeout(M=M, S=S, mu_c=mu_c, sigma_c=sigma_c,
                                           d=d, p_s=p_s, K_m=K_m, condition='single',
                                           o_val=0, seed=rate_seed, w=w,
                                           adjacency=adjacency, t_end=t_end,
                                           timeout=timeout, b=b, p=p, R_star=R_star)

            if result['timed_out']:
                n_timed_out += 1

            results[key] = result

        if n_timed_out > 0:
            print(f"TIMEOUT_ALERT: R*={R_star}: {n_timed_out}/{no_communities} "
                  f"communities timed out", flush=True)

        print(f"R*={R_star}: {no_communities} communities done", flush=True)

    out_path = os.path.join(DATA_DIR, 'diversity_M10_one_network_pulse_results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Saved results to {out_path}")
