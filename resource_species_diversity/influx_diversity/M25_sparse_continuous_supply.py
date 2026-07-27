# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 2026

@author: jamil

Repeat network_topology_diversity_sweep.py's 'gamma_linear' (sparse,
chain/branching) continuous-supply diversity comparison (flux vs reversible,
mean surviving species vs o), but at M=25 instead of M=10, with K_m ~
Uniform(0.001, 0.1) sampled per (species, resource-pair) for the 'reversible'
variant (K_m_method='uniform', K_m_args) instead of a shared scalar - giving
'reversible' the same kind of fixed structural heterogeneity across
consumers that 'flux' already has "for free" via its energy weighting.
'flux' keeps the original scalar K_m=1e-2 (unchanged - it has no tensor K_m
support), so the comparison stays apples-to-apples on everything except the
deliberate K_m-sampling difference under test.

Network: sample_connected_gamma_network() (network_diagnostics.py) with
mean=0.04, variance=0.0014 - RETROFITTED from the original
sample_shared_network_gamma() (see network_diagnostics.py's docstring/the
gamma-method bug fix): that independent-Bernoulli-per-edge method left
resources disconnected from the dominant (externally-supplied) resource far
more often than expected - a verification sweep found 10/10 tested seeds
had the dominant resource isolated in a small component, cut off from most
of the other M-1 resources, REGARDLESS of species dynamics. This affected
every M25_sparse_*.py script and the parameter sweep - all of this
investigation's earlier "most byproducts never accumulate" findings were
compounded by (not purely explained by) that structural bug. The new
sampler builds a spanning tree rooted at the dominant resource first
(guaranteeing every resource is reachable from it), then layers a few extra
edges on top (extra_edge_scale=0.3 default) - same mean/variance semantics,
same target sparsity (~25-40 edges for M=25), but now ALWAYS fully
connected. check_connectivity() (also network_diagnostics.py) is called
right after sampling to confirm this on every run.

Death rate: d=0.05, up from the (disconnected-network) d=0.01. With every
resource actually reachable, communities support far more species than
before at the same d (a d=0.01 spot check on a connected network gave
44-50/50 survivors - saturated, uninformative). Re-tuned by pilot-testing
d in {0.01, 0.03, 0.05, 0.08, 0.1} against mean survivor count across the
full o range - d=0.05 gives a meaningful spread (mean ~10-12/50, range
4-15) with a visible (if modest) increasing trend across o, without
saturating.

M=25, S=50, mu_C=40 (mu_c=mu_C/M), sigma_C=1.6, b=-0.001, p=1, d=0.05,
o=[0.1,0.3,0.5,0.7,0.9,1.1], t_end=7000, condition='single', 5 networks x 8
communities x 2 kinetics.
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
d = 0.05
S = 50
t_end = 7000
K_m = 1e-2

b = -0.001
p = 1

no_communities = 8

M = 25
mu_c = mu_C / M
sigma_c = sigma_C / np.sqrt(M)

o_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
kinetics_values = ['flux', 'reversible']

GAMMA_SEEDS = [900201, 900202, 900204, 900206, 900207]
GAMMA_MEAN, GAMMA_VARIANCE = 0.04, 0.0014

K_m_method, K_m_args = 'uniform', {'low': 0.001, 'high': 0.1}


def _run_one(combo):

    M_, S_ = combo['M'], combo['S']
    w, adjacency = combo['w'], combo['adjacency']
    dominant = combo['dominant']

    o = np.zeros(M_)
    o[dominant] = combo['o_val']

    np.random.seed(combo['rate_seed'])

    try:

        community = Consumer_Resource_Model('Metabolic pathways', pool_sizes=(M_, S_))
        community.growth_consumption_rates('growth function of consumption',
                                           mu_c=combo['mu_c'], sigma_c=combo['sigma_c'],
                                           mu_g=1, sigma_g=0)
        community.model_specific_rates(death_args={'d': combo['d']},
                                       influx_method='user-supplied', influx_args={'o': o},
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
        community.simulate_community(t_end=combo['t_end'], no_init_cond=1)

        sol = community.ODE_sols[0]
        R_final = sol.y[:M_, -1]
        N_final = sol.y[M_:, -1]

        result = {'net_idx': combo['net_idx'], 'kinetics': combo['kinetics'],
                  'o_val': combo['o_val'], 'community_idx': combo['community_idx'],
                  'survival_fraction': float(np.mean(N_final > 1e-4)),
                  'max_abs_y': float(np.max(np.abs(sol.y))), 'status': int(sol.status),
                  'R_final': R_final, 'N_final': N_final, 'failed': False}

    except Exception as e:

        result = {'net_idx': combo['net_idx'], 'kinetics': combo['kinetics'],
                  'o_val': combo['o_val'], 'community_idx': combo['community_idx'],
                  'failed': True, 'error': str(e)}

    return result


if __name__ == '__main__':

    networks = []
    for seed in GAMMA_SEEDS:
        w, adjacency = sample_connected_gamma_network(M, GAMMA_MEAN, GAMMA_VARIANCE, seed)
        check_connectivity(w, adjacency, verbose=True)
        networks.append((w, adjacency))

    for net_idx, (w, adjacency) in enumerate(networks):
        dominant = np.argmax(w)
        print(f"net {net_idx}: dominant out-degree={int(adjacency[dominant].sum())}, "
              f"total edges={int(adjacency.sum())}", flush=True)

    combos = []

    for net_idx, (w, adjacency) in enumerate(networks):
        dominant = np.argmax(w)
        for kinetics in kinetics_values:
            for o_val in o_values:
                for community_idx in range(no_communities):
                    rate_seed = 950000 + community_idx
                    combos.append(dict(
                        M=M, S=S, mu_c=mu_c, sigma_c=sigma_c, d=d, b=b, p=p, K_m=K_m,
                        t_end=t_end, w=w, adjacency=adjacency, dominant=dominant,
                        net_idx=net_idx, kinetics=kinetics, o_val=o_val,
                        community_idx=community_idx, rate_seed=rate_seed))

    print(f"Total simulations to run: {len(combos)}", flush=True)

    results = {}
    n_done, n_failed = 0, 0

    with mp.Pool(processes=14) as pool:

        for res in pool.imap_unordered(_run_one, combos, chunksize=1):

            key = (res['net_idx'], res['kinetics'], res['o_val'], res['community_idx'])
            results[key] = res
            n_done += 1

            if res['failed']:
                n_failed += 1
                print(f"FAILED: {key}: {res['error']}", flush=True)

            if n_done % 100 == 0:
                print(f"{n_done}/{len(combos)} done ({n_failed} failed)", flush=True)

    print(f"All done: {n_done}/{len(combos)} ({n_failed} failed)", flush=True)

    out_path = os.path.join(DATA_DIR, 'M25_sparse_continuous_supply_connected_results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({'results': results, 'networks': networks,
                    'o_values': o_values, 'kinetics_values': kinetics_values,
                    'no_communities': no_communities, 'd': d, 'M': M}, f)

    print(f"Saved results to {out_path}")
