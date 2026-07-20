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


def sample_shared_network(M, p_s, seed, gated=True):

    '''

    Sample a single (w, adjacency) metabolic network - resource energies
    w_alpha and a (M, M) 0/1 'step'-method adjacency q_{alpha, beta} - to be
    reused identically (via metabolic_network(energies=w, adjacency=adjacency,
    ...)) across many communities/simulations, instead of each community
    sampling its own network.

    gated : bool
        If True (default), a link alpha -> beta exists with probability p_s
        only when w_alpha > w_beta (energy-descending links only, matching
        metabolic_network()'s gated=True/'step' default). If False, q is
        instead sampled Bernoulli(p_s) independently of the sign of
        w_alpha - w_beta - an unstructured random network with a fixed link
        probability, matching the mu_c_vs_M_gated_false* stability scripts.

    '''

    rng = np.random.RandomState(seed)
    w = rng.uniform(0, 1, M)

    if gated:

        energy_differences = w[:, np.newaxis] - w[np.newaxis, :]
        link_probability = np.where(energy_differences > 0, p_s, 0)

    else:

        link_probability = np.full((M, M), p_s)

    adjacency = rng.binomial(1, link_probability, size=(M, M))

    return w, adjacency


def _simulate_worker(params, queue):

    import sys
    sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/consumer_resource_modules')
    from models import Consumer_Resource_Model

    M = params['M']
    S = params['S']

    w, adjacency = params['w'], params['adjacency']
    R_star = params.get('R_star', None)

    if R_star is not None:

        # one-time resource pulse: o = 0 everywhere (no continuous supply),
        # the highest-energy resource starts at R_star instead of the usual
        # small Mallmin-range baseline
        o = np.zeros(M)

    elif params['condition'] == 'single':

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
                                   resource_growth_args={'b': params.get('b', -1)},
                                   resource_inhibition_args={'A': 0})
    growth_saturation = params.get('growth_saturation', True)
    metabolic_network_kwargs = dict(
        energies=w, adjacency=adjacency, network_method='step',
        resource_conversions={'p_s': params['p_s']},
        growth_saturation=growth_saturation,
        K_m=params['K_m'], log_eps=params.get('log_eps', 1e-4),
        production_method='constant', production_args={'p': params.get('p', 1)})

    if growth_saturation:

        metabolic_network_kwargs['saturation_kinetics'] = params.get('saturation_kinetics', 'flux')

        # consumer- and reaction-specific K_m/v_max (only meaningful for
        # saturation_kinetics='reversible' - see metabolic_network()'s
        # docstring); default to None/constant so every other variant is
        # unaffected
        if params.get('K_m_method') is not None:
            metabolic_network_kwargs['K_m_method'] = params['K_m_method']
            metabolic_network_kwargs['K_m_args'] = params['K_m_args']
        if params.get('v_max_method') is not None:
            metabolic_network_kwargs['v_max_method'] = params['v_max_method']
            metabolic_network_kwargs['v_max_args'] = params['v_max_args']

    community.metabolic_network(**metabolic_network_kwargs)

    if R_star is not None:

        # Mallmin-range baseline for every resource/species, then override
        # the highest-energy resource's initial abundance to R_star
        resources_ic = np.random.uniform(1e-8, 2 / M, M)
        resources_ic[np.argmax(w)] = R_star
        species_ic = np.random.uniform(1e-8, 2 / S, S)
        flat_ic = np.concatenate((resources_ic, species_ic))

        community.simulate_community(t_end=params['t_end'], no_init_cond=1,
                                     init_cond_func='user-supplied',
                                     initial_conditions=[flat_ic])

    else:

        community.simulate_community(t_end=params['t_end'], no_init_cond=1)

    sol = community.ODE_sols[0]

    species_traj = sol.y[M:, :]
    survival_frac_t = np.mean(species_traj > 1e-4, axis=0)

    queue.put({'t': sol.t, 'survival_fraction': survival_frac_t, 'status': sol.status,
              'max_abs_y': float(np.max(np.abs(sol.y))), 'timed_out': False})


def simulate_with_timeout(M, S, mu_c, sigma_c, d, p_s, K_m, condition, o_val, seed,
                          w, adjacency, t_end=7000, timeout=60, saturation_kinetics='flux',
                          log_eps=1e-4, b=-1, p=1, R_star=None, growth_saturation=True,
                          K_m_method=None, K_m_args=None, v_max_method=None, v_max_args=None):

    '''

    Run one MP_CRM simulation in a subprocess, killing it (and returning a
    'timed_out': True sentinel) if it exceeds `timeout` seconds.

    w, adjacency : the (M,) energies and (M, M) 0/1 metabolic network to use
        for this simulation - sample once with sample_shared_network() and
        pass the same arrays to every community that should share a network.
    growth_saturation : bool - if False, the original (unsaturated) MP_CRM
        model is used and saturation_kinetics is ignored.
    saturation_kinetics : 'flux', 'thermodynamic', or 'reversible' - which
        growth_saturation=True variant to use (see metabolic_network()'s
        docstring). Ignored if growth_saturation=False.
    K_m_method, K_m_args, v_max_method, v_max_args : only meaningful for
        saturation_kinetics='reversible' - sample K_m/v_max per (species,
        resource-pair) instead of using a single shared scalar (see
        metabolic_network()'s docstring). Left as None (metabolic_network()'s
        own defaults - a uniform scalar K_m, v_max=1 everywhere) unless set.
    b, p : resource self-decay rate and production (byproduct) efficiency -
        exposed so a run can check for unbounded growth (b=0/p=1 is "perfect
        recycling", with no loss channel besides species death) and back off
        to e.g. b=-1e-5 or p=1-1e-5 if it occurs.
    R_star : float or None - if given, o_val/condition are ignored, o is set
        to 0 everywhere, and the simulation starts from a one-time pulse:
        the highest-energy resource's initial abundance is set to R_star
        (every other resource/species starts at the usual small Mallmin-range
        baseline) instead of continuous influx.

    Returns
    -------
    dict with keys 't', 'survival_fraction', 'status', 'max_abs_y',
    'timed_out' - if timed_out is True, the rest are None.

    '''

    params = dict(M=M, S=S, mu_c=mu_c, sigma_c=sigma_c, d=d, p_s=p_s, K_m=K_m,
                  condition=condition, o_val=o_val, seed=seed, t_end=t_end,
                  w=w, adjacency=adjacency, saturation_kinetics=saturation_kinetics,
                  log_eps=log_eps, b=b, p=p, R_star=R_star,
                  growth_saturation=growth_saturation,
                  K_m_method=K_m_method, K_m_args=K_m_args,
                  v_max_method=v_max_method, v_max_args=v_max_args)

    queue = mp.Queue()
    proc = mp.Process(target=_simulate_worker, args=(params, queue))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {'t': None, 'survival_fraction': None, 'status': None, 'max_abs_y': None,
                'timed_out': True}

    if not queue.empty():
        result = queue.get()
        return result

    # process ended without putting a result (crashed) - treat as a failure,
    # not silently as success
    return {'t': None, 'survival_fraction': None, 'status': None, 'timed_out': True}
