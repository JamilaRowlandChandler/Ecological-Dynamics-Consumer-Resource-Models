# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 2026

@author: jamil

Repeat the R*-pulse diversity-vs-R* analysis (one-time pulse of the
dominant/highest-energy resource to R_star, o=0 everywhere/no continuous
supply, comparing 'flux' vs 'reversible' saturation kinetics) on the M=25
sparse (gamma network_method, chain/branching) topology, with K_m ~
Uniform(0.001, 0.1) sampled per (species, resource-pair) for 'reversible'
(see M25_sparse_continuous_supply.py for the same K_m-sampling rationale).

Same 5 network seeds as M25_sparse_continuous_supply.py, RETROFITTED to use
sample_connected_gamma_network() (network_diagnostics.py) instead of
sample_shared_network_gamma() - see that script's docstring for why: the
old sampler left the dominant resource disconnected from most of the
network in 10/10 tested seeds, regardless of species dynamics.
check_connectivity() is called right after sampling to confirm full
connectivity on every run.

Death rate: d=0.04 - re-tuned for the connected network (NOT the old
disconnected-network d=0.03, and still not the continuous-supply script's
d=0.05 - a pulse's death-driven extinction needs several 1/d timescales to
play out within the short t_end=300, unlike continuous supply's t_end=7000,
so the two scenarios need separately-tuned d regardless of network).
Pilot-tested d in {0.03, 0.035, 0.04, 0.05, 0.08} against the R*-dependent
survivor trend on a connected network - d=0.03 saturates too early (34-50
survivors already by R*=10), d=0.05 barely responds until R*=1000 (0
survivors from R*=2 to R*=500). d=0.04 gives a smooth monotonic transition:
0 at R*=2/10, ~5 at R*=50, ~21 at R*=100, ~37 at R*=300, ~41 at R*=500, ~45
at R*=1000 (3-community pilot, net_idx=0, kinetics='reversible').

M=25, S=50, mu_C=40 (mu_c=mu_C/M), sigma_C=1.6, b=-0.001, p=1, d=0.04,
R_star=[2,10,50,100,300,500,1000], t_end=300, 5 networks x 8 communities x
2 kinetics.
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

mu_C = 40
sigma_C = 1.6
d = 0.04
S = 50
t_end = 300
K_m = 1e-2

b = -0.001
p = 1

no_communities = 8

M = 25
mu_c = mu_C / M
sigma_c = sigma_C / np.sqrt(M)

R_star_values = [2, 10, 50, 100, 300, 500, 1000]
kinetics_values = ['flux', 'reversible']

GAMMA_SEEDS = [900201, 900202, 900204, 900206, 900207]
GAMMA_MEAN, GAMMA_VARIANCE = 0.04, 0.0014

K_m_method, K_m_args = 'uniform', {'low': 0.001, 'high': 0.1}


def _run_one(combo):

    M_, S_ = combo['M'], combo['S']
    w, adjacency = combo['w'], combo['adjacency']
    dominant = combo['dominant']
    R_star = combo['R_star']

    np.random.seed(combo['rate_seed'])

    try:

        community = Consumer_Resource_Model('Metabolic pathways', pool_sizes=(M_, S_))
        community.growth_consumption_rates('growth function of consumption',
                                           mu_c=combo['mu_c'], sigma_c=combo['sigma_c'],
                                           mu_g=1, sigma_g=0)
        community.model_specific_rates(death_args={'d': combo['d']},
                                       influx_method='user-supplied', influx_args={'o': np.zeros(M_)},
                                       resource_growth_args={'b': combo['b']},
                                       resource_inhibition_args={'A': 0})

        metabolic_network_kwargs = dict(
            energies=w, adjacency=adjacency, network_method='step',
            resource_conversions={'p_s': 1}, growth_saturation=True,
            saturation_kinetics=combo['kinetics'], K_m=combo['K_m'],
            production_method='constant', production_args={'p': combo['p']})

        if combo['kinetics'] == 'reversible':
            metabolic_network_kwargs['K_m_method'] = K_m_method
            metabolic_network_kwargs['K_m_args'] = K_m_args

        community.metabolic_network(**metabolic_network_kwargs)

        resources_ic = np.random.uniform(1e-8, 2 / M_, M_)
        resources_ic[dominant] = R_star
        species_ic = np.random.uniform(1e-8, 2 / S_, S_)
        flat_ic = np.concatenate((resources_ic, species_ic))

        community.simulate_community(t_end=combo['t_end'], no_init_cond=1,
                                     init_cond_func='user-supplied',
                                     initial_conditions=[flat_ic])

        sol = community.ODE_sols[0]
        R_final = sol.y[:M_, -1]
        N_final = sol.y[M_:, -1]

        result = {'net_idx': combo['net_idx'], 'kinetics': combo['kinetics'],
                  'R_star': R_star, 'community_idx': combo['community_idx'],
                  'survival_fraction': float(np.mean(N_final > 1e-4)),
                  'max_abs_y': float(np.max(np.abs(sol.y))), 'status': int(sol.status),
                  'R_final': R_final, 'N_final': N_final, 'failed': False}

    except Exception as e:

        result = {'net_idx': combo['net_idx'], 'kinetics': combo['kinetics'],
                  'R_star': R_star, 'community_idx': combo['community_idx'],
                  'failed': True, 'error': str(e)}

    return result


if __name__ == '__main__':

    networks = []
    for seed in GAMMA_SEEDS:
        w, adjacency = sample_connected_gamma_network(M, GAMMA_MEAN, GAMMA_VARIANCE, seed)
        check_connectivity(w, adjacency, verbose=True)
        networks.append((w, adjacency))

    combos = []

    for net_idx, (w, adjacency) in enumerate(networks):
        dominant = np.argmax(w)
        for kinetics in kinetics_values:
            for R_star in R_star_values:
                for community_idx in range(no_communities):
                    rate_seed = 950000 + community_idx
                    combos.append(dict(
                        M=M, S=S, mu_c=mu_c, sigma_c=sigma_c, d=d, b=b, p=p, K_m=K_m,
                        t_end=t_end, w=w, adjacency=adjacency, dominant=dominant,
                        net_idx=net_idx, kinetics=kinetics, R_star=R_star,
                        community_idx=community_idx, rate_seed=rate_seed))

    print(f"Total simulations to run: {len(combos)}", flush=True)

    results = {}
    n_done, n_failed = 0, 0

    with mp.Pool(processes=14) as pool:

        for res in pool.imap_unordered(_run_one, combos, chunksize=1):

            key = (res['net_idx'], res['kinetics'], res['R_star'], res['community_idx'])
            results[key] = res
            n_done += 1

            if res['failed']:
                n_failed += 1
                print(f"FAILED: {key}: {res['error']}", flush=True)

            if n_done % 100 == 0:
                print(f"{n_done}/{len(combos)} done ({n_failed} failed)", flush=True)

    print(f"All done: {n_done}/{len(combos)} ({n_failed} failed)", flush=True)

    out_path = os.path.join(DATA_DIR, 'M25_sparse_pulse_connected_results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({'results': results, 'networks': networks,
                    'R_star_values': R_star_values, 'kinetics_values': kinetics_values,
                    'no_communities': no_communities, 'd': d, 'M': M}, f)

    print(f"Saved results to {out_path}")
