# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 2026

@author: jamil

Companion to M25_sparse_continuous_supply.py / M25_sparse_pulse.py - those
scripts only kept each simulation's FINAL survival_fraction (to keep the
960-run batch's pickle size down), not the full survival_fraction(t)
trajectory. This script reruns a smaller subset - continuous supply at
o=1.1 only, pulse at R_star in {10, 100, 1000} only - saving the full (t,
survival_fraction(t)) trajectory from each run, for plotting mean surviving
species over time (with across-community/network variability) comparing
'flux' vs 'reversible'.

Same networks/parameters as the two parent scripts (RETROFITTED to
sample_connected_gamma_network(), mean=0.04/variance=0.0014, seeds
900201/202/204/206/207 - see M25_sparse_continuous_supply.py's docstring
for why: the old sampler left the dominant resource disconnected from most
of the network in 10/10 tested seeds; d=0.05 continuous supply, d=0.04
pulse - both re-tuned for the connected network, see
M25_sparse_continuous_supply.py/M25_sparse_pulse.py's docstrings;
K_m~Uniform(0.001,0.1) tensor for 'reversible' only).
"""

import numpy as np
import sys
import os
import pickle
import multiprocessing as mp

abspath = os.path.abspath(__file__)
file_directory_name = os.path.dirname(abspath)
os.chdir(file_directory_name)

DATA_DIR = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability_crossfeeding/influx_species_diversity"

sys.path.insert(0, file_directory_name)
sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/consumer_resource_modules')

from network_diagnostics import sample_connected_gamma_network, check_connectivity
from models import Consumer_Resource_Model

# %%

M, S = 25, 50
mu_C, sigma_C = 40, 1.6
mu_c, sigma_c = mu_C / M, sigma_C / np.sqrt(M)
K_m = 1e-2
b, p = -0.001, 1
no_communities = 8

GAMMA_SEEDS = [900201, 900202, 900204, 900206, 900207]
GAMMA_MEAN, GAMMA_VARIANCE = 0.04, 0.0014
K_m_method, K_m_args = 'uniform', {'low': 0.001, 'high': 0.1}

d_continuous = 0.05
d_pulse = 0.04
t_end_continuous = 7000
t_end_pulse = 300
o_val = 1.1
R_star_values = [10, 100, 1000]
kinetics_values = ['flux', 'reversible']


def _run_continuous(combo):

    w, adjacency, dominant = combo['w'], combo['adjacency'], combo['dominant']
    o = np.zeros(M)
    o[dominant] = o_val
    np.random.seed(combo['rate_seed'])

    community = Consumer_Resource_Model('Metabolic pathways', pool_sizes=(M, S))
    community.growth_consumption_rates('growth function of consumption',
                                       mu_c=mu_c, sigma_c=sigma_c, mu_g=1, sigma_g=0)
    community.model_specific_rates(death_args={'d': d_continuous},
                                   influx_method='user-supplied', influx_args={'o': o},
                                   resource_growth_args={'b': b}, resource_inhibition_args={'A': 0})
    kwargs = dict(energies=w, adjacency=adjacency, network_method='step',
                 resource_conversions={'p_s': 1}, growth_saturation=True,
                 saturation_kinetics=combo['kinetics'], K_m=K_m,
                 production_method='constant', production_args={'p': p})
    if combo['kinetics'] == 'reversible':
        kwargs['K_m_method'] = K_m_method
        kwargs['K_m_args'] = K_m_args
    community.metabolic_network(**kwargs)
    community.simulate_community(t_end=t_end_continuous, no_init_cond=1)

    sol = community.ODE_sols[0]
    survival_t = np.mean(sol.y[M:, :] > 1e-4, axis=0)

    return {'net_idx': combo['net_idx'], 'kinetics': combo['kinetics'],
           'community_idx': combo['community_idx'], 't': sol.t, 'survival_fraction': survival_t}


def _run_pulse(combo):

    w, adjacency, dominant = combo['w'], combo['adjacency'], combo['dominant']
    R_star = combo['R_star']
    np.random.seed(combo['rate_seed'])

    community = Consumer_Resource_Model('Metabolic pathways', pool_sizes=(M, S))
    community.growth_consumption_rates('growth function of consumption',
                                       mu_c=mu_c, sigma_c=sigma_c, mu_g=1, sigma_g=0)
    community.model_specific_rates(death_args={'d': d_pulse},
                                   influx_method='user-supplied', influx_args={'o': np.zeros(M)},
                                   resource_growth_args={'b': b}, resource_inhibition_args={'A': 0})
    kwargs = dict(energies=w, adjacency=adjacency, network_method='step',
                 resource_conversions={'p_s': 1}, growth_saturation=True,
                 saturation_kinetics=combo['kinetics'], K_m=K_m,
                 production_method='constant', production_args={'p': p})
    if combo['kinetics'] == 'reversible':
        kwargs['K_m_method'] = K_m_method
        kwargs['K_m_args'] = K_m_args
    community.metabolic_network(**kwargs)

    resources_ic = np.random.uniform(1e-8, 2 / M, M)
    resources_ic[dominant] = R_star
    species_ic = np.random.uniform(1e-8, 2 / S, S)
    flat_ic = np.concatenate((resources_ic, species_ic))

    community.simulate_community(t_end=t_end_pulse, no_init_cond=1,
                                 init_cond_func='user-supplied', initial_conditions=[flat_ic])

    sol = community.ODE_sols[0]
    survival_t = np.mean(sol.y[M:, :] > 1e-4, axis=0)

    return {'net_idx': combo['net_idx'], 'kinetics': combo['kinetics'], 'R_star': R_star,
           'community_idx': combo['community_idx'], 't': sol.t, 'survival_fraction': survival_t}


if __name__ == '__main__':

    networks = [sample_connected_gamma_network(M, GAMMA_MEAN, GAMMA_VARIANCE, s) for s in GAMMA_SEEDS]
    for w, adjacency in networks:
        check_connectivity(w, adjacency, verbose=True)

    cont_combos = []
    pulse_combos = []

    for net_idx, (w, adjacency) in enumerate(networks):
        dominant = int(np.argmax(w))
        for kinetics in kinetics_values:
            for community_idx in range(no_communities):
                rate_seed = 950000 + community_idx
                cont_combos.append(dict(w=w, adjacency=adjacency, dominant=dominant,
                                        net_idx=net_idx, kinetics=kinetics,
                                        community_idx=community_idx, rate_seed=rate_seed))
                for R_star in R_star_values:
                    pulse_combos.append(dict(w=w, adjacency=adjacency, dominant=dominant,
                                             net_idx=net_idx, kinetics=kinetics, R_star=R_star,
                                             community_idx=community_idx, rate_seed=rate_seed))

    print(f"Continuous-supply trajectories to run: {len(cont_combos)}", flush=True)

    with mp.Pool(processes=14) as pool:
        cont_results = list(pool.imap_unordered(_run_continuous, cont_combos, chunksize=1))

    print(f"Continuous-supply done: {len(cont_results)}", flush=True)

    print(f"Pulse trajectories to run: {len(pulse_combos)}", flush=True)

    with mp.Pool(processes=14) as pool:
        pulse_results = list(pool.imap_unordered(_run_pulse, pulse_combos, chunksize=1))

    print(f"Pulse done: {len(pulse_results)}", flush=True)

    out_path = os.path.join(DATA_DIR, 'M25_sparse_trajectories_connected.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({'continuous': cont_results, 'pulse': pulse_results,
                    'o_val': o_val, 'R_star_values': R_star_values,
                    'no_communities': no_communities, 'M': M, 'S': S}, f)

    print(f"Saved to {out_path}")
