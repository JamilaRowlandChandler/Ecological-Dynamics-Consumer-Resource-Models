# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 2026

@author: jamil

Repeat of M25_sparse_trajectories.py's continuous-supply and resource-pulse
investigation on the M=25 sparse (gamma network_method, gated) topology, but
using saturation_kinetics='boltzmann' instead of 'reversible'/'flux'. Saves
the full survival_fraction(t) trajectory directly from this run (unlike the
'reversible' investigation, which needed a separate follow-up rerun to get
trajectories after the initial batch only kept final values).

Same 5 network seeds as the earlier M=25 scripts, RETROFITTED to
sample_connected_gamma_network() (mean=0.04, variance=0.0014, seeds
900201/202/204/206/207) - see M25_sparse_continuous_supply.py's docstring
for why: the old sampler left the dominant resource disconnected from most
of the network in 10/10 tested seeds, regardless of species dynamics.
check_connectivity() confirms full connectivity right after sampling.

K_m=1.0 (NOT the 1e-2 used for 'flux'/'reversible' - 'boltzmann' uses K_m as
a Boltzmann-factor thermal scale inside exp(w/K_m), not a Michaelis-Menten
half-saturation constant, so a small K_m causes exp(w/K_m) to overflow; see
metabolic_network()'s K_m docstring).

Death rates RE-TUNED for the connected network (the old d=0.01 continuous /
d=0.03 pulse, both carried over unchanged from 'reversible', now badly
under-select - a spot check on the connected network gave 44-50/50
survivors at d=0.01):
  - continuous supply: d=0.05 gives mean ~10-12/50 survivors, matching what
    was re-tuned for 'reversible' on the same connected network (see
    M25_sparse_continuous_supply.py's docstring) - the two kinetics still
    behave similarly under d re-tuning, as they did before the retrofit.
  - pulse: d=0.04 gives a comparable R*-dependent transition to
    'reversible's (see M25_sparse_pulse.py's docstring) - spot-checked
    directly: 0 at R*=10, ~15 at R*=100, ~41 at R*=1000.

o_values = [0.1, 0.3, 1.0] (user-specified, narrower than the 6-value sweep
used for 'reversible'/'flux'). R_star_values reuses the established 7-value
sweep. 5 networks x 8 communities x (3 o-values + 7 R*-values) = 400 runs.
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
K_m = 1.0
b, p = -0.001, 1
no_communities = 8

GAMMA_SEEDS = [900201, 900202, 900204, 900206, 900207]
GAMMA_MEAN, GAMMA_VARIANCE = 0.04, 0.0014

d_continuous = 0.05
d_pulse = 0.04
t_end_continuous = 7000
t_end_pulse = 300
o_values = [0.1, 0.3, 1.0]
R_star_values = [2, 10, 50, 100, 300, 500, 1000]

SATURATION_KINETICS = 'boltzmann'


def _run_continuous(combo):

    w, adjacency, dominant = combo['w'], combo['adjacency'], combo['dominant']
    o = np.zeros(M)
    o[dominant] = combo['o_val']
    np.random.seed(combo['rate_seed'])

    community = Consumer_Resource_Model('Metabolic pathways', pool_sizes=(M, S))
    community.growth_consumption_rates('growth function of consumption',
                                       mu_c=mu_c, sigma_c=sigma_c, mu_g=1, sigma_g=0)
    community.model_specific_rates(death_args={'d': d_continuous},
                                   influx_method='user-supplied', influx_args={'o': o},
                                   resource_growth_args={'b': b}, resource_inhibition_args={'A': 0})
    community.metabolic_network(energies=w, adjacency=adjacency, network_method='step',
                                resource_conversions={'p_s': 1}, growth_saturation=True,
                                saturation_kinetics=SATURATION_KINETICS, K_m=K_m,
                                production_method='constant', production_args={'p': p})
    community.simulate_community(t_end=t_end_continuous, no_init_cond=1)

    sol = community.ODE_sols[0]
    survival_t = np.mean(sol.y[M:, :] > 1e-4, axis=0)

    return {'net_idx': combo['net_idx'], 'o_val': combo['o_val'],
           'community_idx': combo['community_idx'], 't': sol.t,
           'survival_fraction': survival_t, 'max_abs_y': float(np.max(np.abs(sol.y))),
           'status': int(sol.status)}


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
    community.metabolic_network(energies=w, adjacency=adjacency, network_method='step',
                                resource_conversions={'p_s': 1}, growth_saturation=True,
                                saturation_kinetics=SATURATION_KINETICS, K_m=K_m,
                                production_method='constant', production_args={'p': p})

    resources_ic = np.random.uniform(1e-8, 2 / M, M)
    resources_ic[dominant] = R_star
    species_ic = np.random.uniform(1e-8, 2 / S, S)
    flat_ic = np.concatenate((resources_ic, species_ic))

    community.simulate_community(t_end=t_end_pulse, no_init_cond=1,
                                 init_cond_func='user-supplied', initial_conditions=[flat_ic])

    sol = community.ODE_sols[0]
    survival_t = np.mean(sol.y[M:, :] > 1e-4, axis=0)

    return {'net_idx': combo['net_idx'], 'R_star': R_star,
           'community_idx': combo['community_idx'], 't': sol.t,
           'survival_fraction': survival_t, 'max_abs_y': float(np.max(np.abs(sol.y))),
           'status': int(sol.status)}


if __name__ == '__main__':

    networks = [sample_connected_gamma_network(M, GAMMA_MEAN, GAMMA_VARIANCE, s) for s in GAMMA_SEEDS]

    for net_idx, (w, adjacency) in enumerate(networks):
        dominant = np.argmax(w)
        print(f"net {net_idx}: dominant out-degree={int(adjacency[dominant].sum())}, "
              f"total edges={int(adjacency.sum())}", flush=True)
        check_connectivity(w, adjacency, verbose=True)

    cont_combos = []
    pulse_combos = []

    for net_idx, (w, adjacency) in enumerate(networks):
        dominant = int(np.argmax(w))
        for community_idx in range(no_communities):
            rate_seed = 950000 + community_idx
            for o_val in o_values:
                cont_combos.append(dict(w=w, adjacency=adjacency, dominant=dominant,
                                        net_idx=net_idx, o_val=o_val,
                                        community_idx=community_idx, rate_seed=rate_seed))
            for R_star in R_star_values:
                pulse_combos.append(dict(w=w, adjacency=adjacency, dominant=dominant,
                                         net_idx=net_idx, R_star=R_star,
                                         community_idx=community_idx, rate_seed=rate_seed))

    print(f"Continuous-supply runs: {len(cont_combos)}", flush=True)

    with mp.Pool(processes=14) as pool:
        cont_results = list(pool.imap_unordered(_run_continuous, cont_combos, chunksize=1))

    n_failed_cont = sum(1 for r in cont_results if r['status'] != 0)
    print(f"Continuous-supply done: {len(cont_results)} ({n_failed_cont} non-zero status)", flush=True)

    print(f"Pulse runs: {len(pulse_combos)}", flush=True)

    with mp.Pool(processes=14) as pool:
        pulse_results = list(pool.imap_unordered(_run_pulse, pulse_combos, chunksize=1))

    n_failed_pulse = sum(1 for r in pulse_results if r['status'] != 0)
    print(f"Pulse done: {len(pulse_results)} ({n_failed_pulse} non-zero status)", flush=True)

    out_path = os.path.join(DATA_DIR, 'M25_sparse_boltzmann_trajectories_connected.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({'continuous': cont_results, 'pulse': pulse_results,
                    'o_values': o_values, 'R_star_values': R_star_values,
                    'no_communities': no_communities, 'M': M, 'S': S,
                    'K_m': K_m, 'd_continuous': d_continuous, 'd_pulse': d_pulse,
                    'saturation_kinetics': SATURATION_KINETICS}, f)

    print(f"Saved to {out_path}")
