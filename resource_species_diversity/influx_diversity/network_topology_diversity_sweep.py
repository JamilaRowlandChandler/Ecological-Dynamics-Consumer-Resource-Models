# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 2026

@author: jamil

Repeat the gated, continuous-supply 'reversible' vs 'flux' diversity
comparison (diversity_M10_one_network_b0.py / _reversible.py) AND the
best-growth-source specialisation check (does any surviving species
specialise on a resource other than the directly-supplied/dominant one?)
across two new metabolic network topologies, to test whether the earlier
p_s=0.5 result - reversible kinetics give LOWER diversity than flux, and
every surviving species' best growth source is the dominant resource,
regardless of topology - was an artefact of that one network's structure
(the dominant resource happened to have out-degree 1).

Topology 1 ('dense_ps1'): p_s=1, gated - every energy-descending resource
pair is linked (a total order DAG). The dominant resource now has the
MAXIMUM possible out-degree (M-1=9), spreading its outflow across many
edges instead of concentrating it on one - the opposite extreme from the
p_s=0.5 case.

Topology 2 ('gamma_linear'): network_method='gamma' (see
metabolic_network()'s docstring) with mean=0.1, variance=0.009 - chosen by
grid search (mean total edges ~9-10 for M=10, i.e. close to a spanning
tree/chain) after confirming shape=mean**2/variance must be >=1 (shape<1
makes gamma.pdf(mode=0,...) diverge to inf, silently zeroing every link
probability - a degenerate case, not usable). This gives a sparse,
near-neighbour-in-energy-only, chain/branching structure (median out/in
degree 0-1, occasional branch points) rather than the dense-ish p_s=0.5
random graph. Network seeds were filtered to require the dominant
resource's out-degree >= 1 (else continuous supply to it does nothing,
since consumption/production only happen along existing edges).

5 independently-sampled networks per topology (10 total). For each network:
  - no_communities=8 sets of growth/consumption rates (network held fixed)
  - o swept over o_values for both saturation_kinetics in ('flux','reversible')
  - each simulation's FINAL resource/species state (R_final, N_final) is
    kept (not just summary survival_fraction(t)) so the o=1.1 'reversible'
    runs can double as the specialisation-check data (best growth source per
    surviving species) without a separate simulation pass.

Simulations are run directly (no per-run subprocess timeout kill, unlike
simulate_with_timeout) inside a multiprocessing.Pool, since this parameter
regime (K_m=1e-2, d=0.1, b=-0.001, gated) has not shown stiffness/timeout
issues in prior runs - traded for the ability to parallelise across the
much larger combo count (10 networks x 6 o-values x 8 communities x 2
kinetics = 960 runs) on this machine's 16 cores.

M=10, K_m=1e-2, d=0.1, mu_C=40 (mu_c=mu_C/M), S=50, sigma_C=1.6, b=-0.001,
p=1, o=[0.1,0.3,0.5,0.7,0.9,1.1], t_end=7000, condition='single' only.
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

from timeout_utils import sample_shared_network, sample_shared_network_gamma
from models import Consumer_Resource_Model

# %%

mu_C = 40
sigma_C = 1.6
d = 0.1
S = 50
t_end = 7000
K_m = 1e-2

b = -0.001
p = 1

no_communities = 8

M = 10
mu_c = mu_C / M
sigma_c = sigma_C / np.sqrt(M)

o_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
kinetics_values = ['flux', 'reversible']

DENSE_SEEDS = [900001, 900002, 900003, 900004, 900005]
GAMMA_SEEDS = [900102, 900103, 900104, 900105, 900108]
GAMMA_MEAN, GAMMA_VARIANCE = 0.1, 0.009


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
        community.metabolic_network(energies=w, adjacency=adjacency, network_method='step',
                                    resource_conversions={'p_s': 1}, growth_saturation=True,
                                    saturation_kinetics=combo['kinetics'], K_m=combo['K_m'],
                                    production_method='constant', production_args={'p': combo['p']})
        community.simulate_community(t_end=combo['t_end'], no_init_cond=1)

        sol = community.ODE_sols[0]
        R_final = sol.y[:M_, -1]
        N_final = sol.y[M_:, -1]

        result = {'topology': combo['topology'], 'net_idx': combo['net_idx'],
                  'kinetics': combo['kinetics'], 'o_val': combo['o_val'],
                  'community_idx': combo['community_idx'],
                  'survival_fraction': float(np.mean(N_final > 1e-4)),
                  'max_abs_y': float(np.max(np.abs(sol.y))), 'status': int(sol.status),
                  'R_final': R_final, 'N_final': N_final, 'failed': False}

    except Exception as e:

        result = {'topology': combo['topology'], 'net_idx': combo['net_idx'],
                  'kinetics': combo['kinetics'], 'o_val': combo['o_val'],
                  'community_idx': combo['community_idx'], 'failed': True,
                  'error': str(e)}

    return result


if __name__ == '__main__':

    networks = {'dense_ps1': [], 'gamma_linear': []}

    for seed in DENSE_SEEDS:
        w, adjacency = sample_shared_network(M, 1.0, seed, gated=True)
        networks['dense_ps1'].append((w, adjacency))

    for seed in GAMMA_SEEDS:
        w, adjacency = sample_shared_network_gamma(M, GAMMA_MEAN, GAMMA_VARIANCE, seed)
        networks['gamma_linear'].append((w, adjacency))

    # sanity-print network structure before committing to the full sweep
    for topology, net_list in networks.items():
        for net_idx, (w, adjacency) in enumerate(net_list):
            dominant = np.argmax(w)
            print(f"{topology} net {net_idx}: dominant out-degree="
                  f"{int(adjacency[dominant].sum())}, total edges={int(adjacency.sum())}",
                  flush=True)

    combos = []

    for topology, net_list in networks.items():
        for net_idx, (w, adjacency) in enumerate(net_list):
            dominant = np.argmax(w)
            for kinetics in kinetics_values:
                for o_val in o_values:
                    for community_idx in range(no_communities):
                        rate_seed = 950000 + community_idx
                        combos.append(dict(
                            M=M, S=S, mu_c=mu_c, sigma_c=sigma_c, d=d, b=b, p=p, K_m=K_m,
                            t_end=t_end, w=w, adjacency=adjacency, dominant=dominant,
                            topology=topology, net_idx=net_idx, kinetics=kinetics,
                            o_val=o_val, community_idx=community_idx, rate_seed=rate_seed))

    print(f"Total simulations to run: {len(combos)}", flush=True)

    results = {}
    n_done = 0
    n_failed = 0

    with mp.Pool(processes=14) as pool:

        for res in pool.imap_unordered(_run_one, combos, chunksize=1):

            key = (res['topology'], res['net_idx'], res['kinetics'], res['o_val'], res['community_idx'])
            results[key] = res
            n_done += 1

            if res['failed']:
                n_failed += 1
                print(f"FAILED: {key}: {res['error']}", flush=True)

            if n_done % 100 == 0:
                print(f"{n_done}/{len(combos)} done ({n_failed} failed)", flush=True)

    print(f"All done: {n_done}/{len(combos)} ({n_failed} failed)", flush=True)

    out_path = os.path.join(DATA_DIR, 'network_topology_diversity_sweep_results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({'results': results, 'networks': networks,
                    'o_values': o_values, 'kinetics_values': kinetics_values,
                    'no_communities': no_communities}, f)

    print(f"Saved results to {out_path}")
