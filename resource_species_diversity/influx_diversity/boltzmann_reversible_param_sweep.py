# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 2026

@author: jamil

Screening sweep across biologically-reasonable rate-parameter regimes for
'boltzmann' and 'reversible' saturation kinetics on the M=25 sparse (gamma
network_method, gated) topology, to check whether the diversity-vs-influx
trend (diversity RISES with continuous supply o, found for both kinetics on
this topology) ever REVERSES under some parameter combination.

Swept parameters (all kept strictly positive, varying their distributional
mean and/or spread):
  - mu_C in {40, 80} (total consumption budget; mu_c = mu_C/M). mu_C=20 was
    DROPPED after the first pass through this grid (see below) - combined
    with sigma_C>=1.6 it gives sigma_c/mu_c ratios high enough that
    Normal(mu_c,sigma_c) sampled negative consumption rates for a
    meaningful fraction of species-resource pairs, which is what actually
    caused nearly all of that run's failures/severe slowdowns (confirmed
    independent of K_m and kinetics) - not a biologically reasonable corner
    of parameter space to begin with, and excluding it lets the rest of
    the grid finish in the original fast regime.
  - sigma_C in {0.8, 1.6, 3.2} (consumption-rate spread; sigma_c = sigma_C/sqrt(M))
  - sigma_g in {0, 0.15, 0.3} (growth-yield spread; mu_g fixed at 1 - yield
    is sampled rue = mu_g + sigma_g*X, X~N(0,1), via growth_consumption_rates(),
    so this is already-existing model infrastructure, not a new feature)
  - K_m: variant-specific range, since the two variants use K_m completely
    differently (see models.py's saturation_kinetics docstring) -
    'reversible' (Michaelis-Menten-style half-saturation): {1e-3, 1e-2, 1e-1}
    'boltzmann' (Boltzmann-factor thermal scale): {0.1, 1.0, 5.0}
That's 2*3^3 = 54 parameter combinations per kinetics variant (108 total).

RESULT FROM THE FIRST (mu_C in {20,40,80}) PASS: across the 144 fully-
completed, non-degenerate combinations analysed (all mu_C, before dropping
20), the diversity-vs-o trend was increasing or flat in every single case -
weakest +0.25 species (o=0.1->1.1), strongest +8.08, never negative. No
reversal found. Two apparent "reversals" surfaced initially but turned out
to be data-completeness artefacts (unequal replicate counts across o
because the run was killed mid-grid, e.g. one network's low-diversity runs
averaged against a DIFFERENT, smaller set of networks at high o) - caught
by requiring all 12 replicates (3 networks x 4 communities) present at
every o value before accepting a trend as real. This run's job is to
extend that same (now more reliable, mu_C=20-free) analysis to full
completion.

NOT swept (fixed, to keep this screening tractable): network topology
(fixed at the established sparse gamma network, mean=0.04/variance=0.0014,
3 of the previous 5 seeds for speed) - some parameter combinations may
still land in a degenerate regime (~0 or ~50/50 survivors at every o),
which can't show a meaningful trend either way; these get flagged and
excluded from the trend verdict during analysis, not silently treated as
"no reversal".

RETROFITTED to sample_connected_gamma_network() (network_diagnostics.py)
instead of sample_shared_network_gamma() - see M25_sparse_continuous_
supply.py's docstring for why: the old sampler left the dominant resource
disconnected from most of the network in 10/10 tested seeds, regardless of
species dynamics. check_connectivity() confirms full connectivity right
after sampling. Death rate also re-tuned as a result: d=0.05, up from the
disconnected-network d=0.01 (which now under-selects badly - a spot check
at mu_C=40, o=1.1 on a connected network gave mean 11.7/50 at d=0.05 vs.
44-50/50 at d=0.01; mu_C=80 gave mean 22.3/50 at d=0.05, still a
reasonable, non-saturated spread across the full mu_C range tested here).

o_values = [0.1, 0.3, 0.7, 1.1] (4 points spanning the low-to-high range
used throughout this investigation). 3 networks x 4 communities = 12
replicates per (kinetics, param combo, o value). Only the FINAL survival
fraction is kept per run (not the full trajectory) to keep this large
screening sweep's result objects small - 81 combos x 4 o-values x 12
replicates x 2 kinetics = 7776 runs total.

Runs directly (no per-run subprocess timeout kill) inside a
multiprocessing.Pool, like the M=25 scripts before it - but unlike those,
this sweep covers untested parameter regions where LSODA stiffness is much
more severe, so results are CHECKPOINTED every 200 completions (overwriting
boltzmann_reversible_param_sweep_checkpoint.pkl) and this script is safe to
kill and rerun - on restart it loads the checkpoint and only runs whatever
combos are still missing, rather than starting over.

TIMING NOTE: 'reversible' with small K_m (1e-3) and high sigma_C is
genuinely, severely slow to integrate - confirmed by direct per-combo
timing at ~0.3-0.5s for sigma_C=0.8 vs. 7-28s (and apparently still
climbing) for sigma_C=1.6-3.2, otherwise identical parameters. This is NOT
a hang (verified via steady, undiminished CPU usage across all pool workers
throughout), just far more LSODA steps needed as K_m+R_alpha+R_beta gets
driven closer to its floor more often with wider consumption-rate variance
- but it means this sweep can take much longer in wall-clock time than the
naive 7776-runs-at-~0.3s estimate would suggest, concentrated in the
K_m=1e-3 x high-sigma_C corner of the 'reversible' grid. Expect this to
show up as long stretches between checkpoints while that corner is being
processed - check CPU usage across worker processes (not just checkpoint
recency) before concluding a run has genuinely stalled.
"""

import numpy as np
import sys
import os
import pickle
import warnings
import multiprocessing as mp
from datetime import datetime

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
b, p_eff = -0.001, 1
d = 0.05
t_end = 7000

GAMMA_SEEDS = [900201, 900202, 900204]
GAMMA_MEAN, GAMMA_VARIANCE = 0.04, 0.0014
no_communities = 4

o_values = [0.1, 0.3, 0.7, 1.1]

mu_C_values = [40, 80]
sigma_C_values = [0.8, 1.6, 3.2]
sigma_g_values = [0, 0.15, 0.3]
K_m_values = {'reversible': [1e-3, 1e-2, 1e-1], 'boltzmann': [0.1, 1.0, 5.0]}

kinetics_values = ['reversible', 'boltzmann']

CHECKPOINT_PATH = os.path.join(DATA_DIR, 'boltzmann_reversible_param_sweep_connected_checkpoint.pkl')
CHECKPOINT_EVERY = 200


def _key(combo):
    return (combo['kinetics'], combo['mu_C'], combo['sigma_C'], combo['sigma_g'],
           combo['K_m'], combo['net_idx'], combo['o_val'], combo['community_idx'])


def _run_one(combo):

    w, adjacency, dominant = combo['w'], combo['adjacency'], combo['dominant']
    mu_c = combo['mu_C'] / M
    sigma_c = combo['sigma_C'] / np.sqrt(M)

    o = np.zeros(M)
    o[dominant] = combo['o_val']

    np.random.seed(combo['rate_seed'])

    try:

        # small K_m (e.g. 1e-3) for 'reversible' combined with high sigma_C
        # is genuinely, severely slow to integrate (confirmed by direct
        # timing: ~0.3-0.5s at sigma_C=0.8 vs. 7-28s at sigma_C=1.6 for
        # otherwise-identical combos, apparently still climbing at
        # sigma_C=3.2) - NOT an infinite hang, just far more LSODA steps
        # needed as the K_m+R_alpha+R_beta denominator gets driven closer to
        # its floor more often with wider consumption-rate variance. An
        # early version of this sweep also hit sporadic RuntimeWarnings
        # (divide by zero / invalid value in the reversible flux) that were
        # initially (incorrectly) suspected to be causing a NaN-thrashing
        # hang; direct timing showed the true cause was just this severe
        # but finite slowdown, and killing that run was premature. Kept
        # RuntimeWarning promoted to an exception anyway as a harmless
        # defensive fail-fast for any genuinely pathological corner
        # elsewhere in the grid.
        with warnings.catch_warnings():

            warnings.filterwarnings('error', category=RuntimeWarning)

            community = Consumer_Resource_Model('Metabolic pathways', pool_sizes=(M, S))
            community.growth_consumption_rates('growth function of consumption',
                                               mu_c=mu_c, sigma_c=sigma_c,
                                               mu_g=1, sigma_g=combo['sigma_g'])
            community.model_specific_rates(death_args={'d': d},
                                           influx_method='user-supplied', influx_args={'o': o},
                                           resource_growth_args={'b': b}, resource_inhibition_args={'A': 0})
            community.metabolic_network(energies=w, adjacency=adjacency, network_method='step',
                                        resource_conversions={'p_s': 1}, growth_saturation=True,
                                        saturation_kinetics=combo['kinetics'], K_m=combo['K_m'],
                                        production_method='constant', production_args={'p': p_eff})
            community.simulate_community(t_end=t_end, no_init_cond=1)

            sol = community.ODE_sols[0]
            N_final = sol.y[M:, -1]

            result = {'survival_fraction': float(np.mean(N_final > 1e-4)),
                     'max_abs_y': float(np.max(np.abs(sol.y))), 'status': int(sol.status),
                     'failed': False}

    except Exception as e:

        result = {'failed': True, 'error': str(e)}

    result['key'] = _key(combo)

    return result


if __name__ == '__main__':

    networks = [sample_connected_gamma_network(M, GAMMA_MEAN, GAMMA_VARIANCE, s) for s in GAMMA_SEEDS]
    for w, adjacency in networks:
        check_connectivity(w, adjacency, verbose=True)

    combos = []

    for kinetics in kinetics_values:
        for mu_C in mu_C_values:
            for sigma_C in sigma_C_values:
                for sigma_g in sigma_g_values:
                    for K_m in K_m_values[kinetics]:
                        for net_idx, (w, adjacency) in enumerate(networks):
                            dominant = int(np.argmax(w))
                            for o_val in o_values:
                                for community_idx in range(no_communities):
                                    rate_seed = 950000 + community_idx
                                    combos.append(dict(
                                        kinetics=kinetics, mu_C=mu_C, sigma_C=sigma_C,
                                        sigma_g=sigma_g, K_m=K_m, w=w, adjacency=adjacency,
                                        dominant=dominant, net_idx=net_idx, o_val=o_val,
                                        community_idx=community_idx, rate_seed=rate_seed))

    print(f"Total combos: {len(combos)}", flush=True)

    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'rb') as f:
            results = pickle.load(f)
        print(f"Resuming from checkpoint: {len(results)} already done", flush=True)
    else:
        results = {}

    remaining = [c for c in combos if _key(c) not in results]
    print(f"Remaining to run: {len(remaining)}", flush=True)

    n_done = len(results)
    n_failed = sum(1 for r in results.values() if r.get('failed'))

    if remaining:

        with mp.Pool(processes=14) as pool:

            for res in pool.imap_unordered(_run_one, remaining, chunksize=1):

                results[res['key']] = res
                n_done += 1

                if res.get('failed'):
                    n_failed += 1
                    print(f"FAILED: {res['key']}: {res.get('error')}", flush=True)

                if n_done % CHECKPOINT_EVERY == 0:
                    with open(CHECKPOINT_PATH, 'wb') as f:
                        pickle.dump(results, f)
                    ts = datetime.now().strftime('%H:%M:%S')
                    print(f"[{ts}] {n_done}/{len(combos)} done ({n_failed} failed), "
                         f"checkpoint saved", flush=True)

    with open(CHECKPOINT_PATH, 'wb') as f:
        pickle.dump(results, f)

    print(f"All done: {n_done}/{len(combos)} ({n_failed} failed)", flush=True)
    print(f"Saved checkpoint to {CHECKPOINT_PATH}")
