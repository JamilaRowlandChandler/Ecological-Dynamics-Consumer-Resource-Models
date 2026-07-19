# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 2026

@author: jamil

Run a single MP_CRM simulation in its own subprocess with a hard wall-clock
timeout, killing the subprocess if it's exceeded. Needed because certain
(M, o, community) combinations have been found to make LSODA grind through
genuinely stiff dynamics for minutes at a time (see K_m investigation) - a
soft/thread-based timeout can't actually stop a C-level integration loop, so
this uses multiprocessing.Process (real OS-level termination) instead.

Each simulation gets a fresh subprocess (built from raw parameters, not a
pickled community object), which also sidesteps the separate
"long sequences of solve_ivp() calls in one process slow down" issue found
earlier - every call starts a clean process.
"""

import multiprocessing as mp
import numpy as np


def sample_shared_network(M, p_s, seed):

    '''

    Sample a single (w, adjacency) metabolic network - resource energies
    w_alpha and a (M, M) 0/1 'step'-method adjacency q_{alpha, beta} (link
    alpha -> beta with probability p_s iff w_alpha > w_beta) - to be reused
    identically (via metabolic_network(energies=w, adjacency=adjacency, ...))
    across many communities/simulations, instead of each community sampling
    its own network.

    '''

    rng = np.random.RandomState(seed)
    w = rng.uniform(0, 1, M)
    energy_differences = w[:, np.newaxis] - w[np.newaxis, :]
    link_probability = np.where(energy_differences > 0, p_s, 0)
    adjacency = rng.binomial(1, link_probability, size=(M, M))

    return w, adjacency


def _simulate_worker(params, queue):

    import sys
    sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/consumer_resource_modules')
    from models import Consumer_Resource_Model

    M = params['M']
    S = params['S']

    w, adjacency = params['w'], params['adjacency']

    if params['condition'] == 'single':
        o = np.zeros(M)
        o[np.argmax(w)] = params['o_val']
    else:
        o = np.full(M, params['o_val'])

    # only growth/consumption rates (c_ia, y_ia) are freshly sampled per
    # community - w and the metabolic network q are fixed (passed in above)
    np.random.seed(params['seed'])

    community = Consumer_Resource_Model('Metabolic pathways', pool_sizes=(M, S))
    community.growth_consumption_rates('growth function of consumption',
                                       mu_c=params['mu_c'], sigma_c=params['sigma_c'],
                                       mu_g=1, sigma_g=0)
    community.model_specific_rates(death_args={'d': params['d']},
                                   influx_method='user-supplied', influx_args={'o': o},
                                   resource_growth_args={'b': -1},
                                   resource_inhibition_args={'A': 0})
    community.metabolic_network(energies=w, adjacency=adjacency, network_method='step',
                                resource_conversions={'p_s': params['p_s']},
                                growth_saturation=True,
                                saturation_kinetics=params.get('saturation_kinetics', 'flux'),
                                K_m=params['K_m'], log_eps=params.get('log_eps', 1e-4),
                                production_method='constant', production_args={'p': 1})

    community.simulate_community(t_end=params['t_end'], no_init_cond=1)
    sol = community.ODE_sols[0]

    species_traj = sol.y[M:, :]
    survival_frac_t = np.mean(species_traj > 1e-4, axis=0)

    queue.put({'t': sol.t, 'survival_fraction': survival_frac_t, 'status': sol.status,
              'timed_out': False})


def simulate_with_timeout(M, S, mu_c, sigma_c, d, p_s, K_m, condition, o_val, seed,
                          w, adjacency, t_end=7000, timeout=60, saturation_kinetics='flux',
                          log_eps=1e-4):

    '''

    Run one MP_CRM simulation in a subprocess, killing it (and returning a
    'timed_out': True sentinel) if it exceeds `timeout` seconds.

    w, adjacency : the (M,) energies and (M, M) 0/1 metabolic network to use
        for this simulation - sample once with sample_shared_network() and
        pass the same arrays to every community that should share a network.
    saturation_kinetics : 'flux' (default) or 'thermodynamic' - which
        growth_saturation=True variant to use (see metabolic_network()'s
        docstring).

    Returns
    -------
    dict with keys 't', 'survival_fraction', 'status', 'timed_out' - if
    timed_out is True, 't'/'survival_fraction'/'status' are None.

    '''

    params = dict(M=M, S=S, mu_c=mu_c, sigma_c=sigma_c, d=d, p_s=p_s, K_m=K_m,
                  condition=condition, o_val=o_val, seed=seed, t_end=t_end,
                  w=w, adjacency=adjacency, saturation_kinetics=saturation_kinetics,
                  log_eps=log_eps)

    queue = mp.Queue()
    proc = mp.Process(target=_simulate_worker, args=(params, queue))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {'t': None, 'survival_fraction': None, 'status': None, 'timed_out': True}

    if not queue.empty():
        result = queue.get()
        return result

    # process ended without putting a result (crashed) - treat as a failure,
    # not silently as success
    return {'t': None, 'survival_fraction': None, 'status': None, 'timed_out': True}
