# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 2026

@author: jamil

Repeat of diversity_M10_one_network_pulse_gated_false_reversible.py (M=10,
one non-gated metabolic network, pulse the highest-energy resource to
R_star, o=0 otherwise, saturation_kinetics='reversible'), but with K_m and
V_max sampled independently per (species, resource-pair) from Uniform
distributions (K_m ~ Uniform(0.001, 0.1), V_max ~ Uniform(0.5, 1.5)) instead
of shared scalars - see metabolic_network()'s K_m_method/v_max_method
docstring. Uniform (rather than normal) guarantees strictly positive draws.

Full R_star sweep this time (rather than just R*=10/1000), to see how the
mean-surviving-species-over-time curve's dependence on pulse size holds up
once K_m/V_max carry real per-consumer structure, rather than being
identical for every species/edge.

Uses the SAME network_seed (900000, gated=False) and rate_seed convention as
the other pulse scripts for comparability. NOTE: K_m/V_max are sampled once
per COMMUNITY (i.e. tied to the same np.random.seed(rate_seed) draw that
also generates that community's growth/consumption rates) rather than once
globally, so - like c_ia/y_ia - they vary community-to-community too, not
just species-to-species within one community.

M=10, p_s=min(5/M,1)=0.5, d=0.1, mu_C=40 (mu_c=mu_C/M), S=50, sigma_C=1.6,
b=-0.001, p=1, R_star=[2, 10, 50, 100, 500, 1000], t_end=300.
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

K_m_method = 'uniform'
K_m_args = {'low': 0.001, 'high': 0.1}
v_max_method = 'uniform'
v_max_args = {'low': 0.5, 'high': 1.5}

no_communities = 20

M = 10
mu_c = mu_C / M
sigma_c = sigma_C / np.sqrt(M)
p_s = min(5 / M, 1)

R_star_values = [2, 10, 50, 100, 500, 1000]

if __name__ == '__main__':

    results = {}

    # SAME network as diversity_M10_one_network_pulse_gated_false.py
    network_seed = 900000
    w, adjacency = sample_shared_network(M, p_s, network_seed, gated=False)

    for R_idx, R_star in enumerate(R_star_values):

        n_timed_out = 0

        for community_idx in range(no_communities):

            rate_seed = 950000 + community_idx

            key = (R_star, community_idx)

            result = simulate_with_timeout(M=M, S=S, mu_c=mu_c, sigma_c=sigma_c,
                                           d=d, p_s=p_s, K_m=1e-2, condition='single',
                                           o_val=0, seed=rate_seed, w=w,
                                           adjacency=adjacency, t_end=t_end,
                                           timeout=timeout, b=b, p=p, R_star=R_star,
                                           growth_saturation=True,
                                           saturation_kinetics='reversible',
                                           K_m_method=K_m_method, K_m_args=K_m_args,
                                           v_max_method=v_max_method, v_max_args=v_max_args)

            if result['timed_out']:
                n_timed_out += 1

            results[key] = result

        if n_timed_out > 0:
            print(f"TIMEOUT_ALERT: R*={R_star}: {n_timed_out}/{no_communities} "
                  f"communities timed out", flush=True)

        print(f"R*={R_star}: {no_communities} communities done", flush=True)

    out_path = os.path.join(DATA_DIR,
                            'diversity_M10_one_network_pulse_gated_false_reversible_sampled_Km_Vmax_results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Saved results to {out_path}")
