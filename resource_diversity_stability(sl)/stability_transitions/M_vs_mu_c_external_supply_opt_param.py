# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:06:42 2026

@author: jamil
"""

import os
import sys
import time
from matplotlib import pyplot as plt
import numpy as np
from copy import deepcopy
from scipy.stats import pearsonr, norm
from scipy.optimize import direct, minimize

os.chdir("C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/consumer_resource_modules")

from models import Consumer_Resource_Model

def rho_calculation(M,
                    mu_c, sigma_c,
                    mu_y, sigma_y):
    
    return 1/np.sqrt(1 + ((sigma_y/mu_y)**2 * (1 + (mu_c**2 / (sigma_c**2 * M)))))

class SimulationTimeout(Exception):
    pass

class SimulationFailed(Exception):
    pass

class CommunityInfeasible(Exception):
    pass

EXTINCTION_THRESHOLD = 1e-4
MIN_SURVIVING_CONSUMERS = 5      # a community is infeasible if surviving consumers <= 5
MAX_SIGN_MISMATCH_PERCENT = 10   # ... or if >= 10% of growth/consumption pairs have opposite signs

def pos_neg_growth_consumption(consumption,
                               growth):

    tot_sign_mismatch = np.sum((growth.flatten() < 0) & (consumption.T.flatten() > 0)) + \
                        np.sum((growth.flatten() > 0) & (consumption.T.flatten() < 0))

    return tot_sign_mismatch

def instability_distance(M,
                         mu_c, sigma_c,
                         mu_y, sigma_y,
                         timeout = None):
    """
    timeout : wall-clock limit in seconds for the ODE integration of this one
    community (None = no limit). Raises SimulationTimeout if exceeded.

    Raises SimulationFailed / CommunityInfeasible if the community is infeasible.
    """
    
    community = Consumer_Resource_Model("Externally-supplied resources",
                                        M,
                                        M)
    
    community.growth_consumption_rates('growth function of consumption',
                                       mu_c = mu_c/M,
                                       sigma_c = sigma_c/np.sqrt(M),
                                       mu_g = mu_y,
                                       sigma_g = sigma_y)
    
    # checked before simulating, since it only depends on the sampled rates
    sign_mismatch_percent = 100 * pos_neg_growth_consumption(community.consumption,
                                                             community.growth) / M**2
    if sign_mismatch_percent >= MAX_SIGN_MISMATCH_PERCENT:

        raise CommunityInfeasible(f"{sign_mismatch_percent:.1f}% sign mismatches")

    community.model_specific_rates()

    if timeout is not None:
        # solve_ivp calls self.model at every step, so shadowing it with a wrapper
        # that checks the clock lets us abort a stalled (e.g. stiff) integration.
        deadline = time.time() + timeout
        model = community.model
        def model_with_deadline(*args, **kwargs):
            if time.time() > deadline:
                raise SimulationTimeout(f"simulation exceeded {timeout} s")
            return model(*args, **kwargs)
        community.model = model_with_deadline
    
    community.simulate_community(2000, 1)

    if community.ODE_sols[0].status != 0:
        
        raise SimulationFailed(f"solver stopped early: {community.ODE_sols[0].message}")
    
    if np.all(community.ODE_sols[0].y[:M, -1] < EXTINCTION_THRESHOLD):
        
        raise SimulationFailed("all consumers extinct")
    
    community.calculate_community_properties()
    
    rho = rho_calculation(M, mu_c, sigma_c, mu_y, sigma_y)
    packing_ratio = community.species_survival_fraction[0]/community.resource_survival_fraction[0]
    instability_distance = rho**2 - packing_ratio
    
    final_consumer_distribution, _ = np.histogram(community.ODE_sols[0].y[:M, -1],
                                                  bins=M)
    surviving_consumers = M - final_consumer_distribution[0]

    if surviving_consumers <= MIN_SURVIVING_CONSUMERS:

        raise CommunityInfeasible(f"{surviving_consumers} surviving consumers")

    return instability_distance, surviving_consumers

def instability_distance_M(mu_c, sigma_c,
                           mu_y, sigma_y,
                           Ms = np.arange(50, 275, 25),
                           no_communities = 5,
                           timeout = None,
                           eval_timeout = None):
    """
    An M is feasible only if all of its communities are feasible. At an M where a
    community is infeasible, the remaining communities aren't simulated, and
    min_is_M / min_sc_M are taken over the feasible Ms only.

    timeout : wall-clock limit (s) for each single community simulation.
    eval_timeout : wall-clock limit (s) for this whole call (all Ms x communities);
    once it runs out, the remaining Ms count as infeasible.

    Returns a dict with
        min_is_M : min over feasible Ms of mean |instability distance| (nan if none)
        min_sc_M : min over feasible Ms of mean surviving consumers (nan if none)
        feasible_Ms : list of feasible Ms
        max_feasible_fraction : largest fraction of communities feasible at any M
            (used to grade candidates with no feasible M)
    """

    deadline = None if eval_timeout is None else time.time() + eval_timeout

    is_M, sc_M, feasible_Ms, feasible_fractions = [], [], [], []

    for M in Ms:

        is_sc = []

        for _ in range(no_communities):

            if deadline is None:
                sim_timeout = timeout
            else:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                sim_timeout = remaining if timeout is None else min(timeout, remaining)

            try:
                is_sc.append(instability_distance(M, mu_c, sigma_c, mu_y, sigma_y,
                                                  timeout = sim_timeout))
            except Exception:
                # timeout, failed solve or infeasible community -> this M is infeasible
                break

        feasible_fractions.append(len(is_sc) / no_communities)

        if len(is_sc) == no_communities:

            feasible_Ms.append(M)
            # mean of |instability distance|, so signs from different communities can't cancel
            is_M.append(np.mean([np.abs(i) for i, _ in is_sc]))
            sc_M.append(np.mean([sc for _, sc in is_sc]))

        if deadline is not None and time.time() >= deadline:
            break

    return dict(min_is_M = np.min(is_M) if is_M else np.nan,
                min_sc_M = np.min(sc_M) if sc_M else np.nan,
                feasible_Ms = feasible_Ms,
                max_feasible_fraction = max(feasible_fractions, default = 0.0))

#%% Objective and constraint

PARAM_NAMES = ["mu_c", "sigma_c", "mu_y", "sigma_y"]
BOUNDS = [(1e-3, 400.0),  # mu_c
          (1e-3, 400.0),  # sigma_c
          (1e-3, 5.0),    # mu_y
          (1e-3, 3.0)]    # sigma_y

SIMULATION_TIMEOUT= 60       # seconds allowed per single community simulation
EVALUATION_TIMEOUT = 600      # seconds allowed per objective evaluation (all Ms x communities)
INFEASIBLE_BASE = 10.0        # score for candidates with no feasible M, so they always
                              # score worse than any feasible one (|instability
                              # distance| is O(1), so any value >> 1 works)

def constrained_objective(params,
                          Ms = np.arange(50, 275, 25),
                          no_communities = 5,
                          timeout = SIMULATION_TIMEOUT,
                          eval_timeout = EVALUATION_TIMEOUT):
    """
    Objective to minimise: min_is_M over the feasible Ms.

    A community is infeasible if its simulation fails or times out, all consumers go
    extinct, <= MIN_SURVIVING_CONSUMERS consumers survive, or >= MAX_SIGN_MISMATCH_PERCENT
    of its growth/consumption pairs have opposite signs. An M is feasible if all its
    communities are. A candidate is only rejected if no M is feasible: it then scores
    INFEASIBLE_BASE + (1 - max_feasible_fraction), so the search is still pushed
    towards candidates where more communities are feasible.
    """

    mu_c, sigma_c, mu_y, sigma_y = params

    result = instability_distance_M(mu_c, sigma_c,
                                    mu_y, sigma_y,
                                    Ms = Ms,
                                    no_communities = no_communities,
                                    timeout = timeout,
                                    eval_timeout = eval_timeout)

    if not result["feasible_Ms"] or not np.isfinite(result["min_is_M"]):
        return INFEASIBLE_BASE + (1 - result["max_feasible_fraction"])

    return result["min_is_M"]

#%% DIRECT, then Nelder-Mead (best protocol in trial runs)

# Search in log10(parameters): the bounds span 1e-3 .. 400, so equal steps in log
# space are far more sensible than equal steps in linear space.
LOG_BOUNDS = [(np.log10(lo), np.log10(hi)) for lo, hi in BOUNDS]
CRN_SEED = 12345   # common random numbers: same seed for every evaluation

def crn_objective(log_params, seed = CRN_SEED):
    """
    constrained_objective in log10 space with a fixed random seed (common random numbers).
    
    Fixing the seed makes the objective deterministic in the parameters, so DIRECT and
    Nelder-Mead see a consistent landscape instead of fresh noise at every call. The
    result is optimised for this one seed, so check it with validate_candidate().
    """
    
    log_params = np.clip(log_params, *np.array(LOG_BOUNDS).T)
    
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        return constrained_objective(10**log_params)
    finally:
        np.random.set_state(state)

def run_direct_nelder_mead(direct_maxfun = 50,
                           nm_maxfev = 50,
                           nm_step = 0.2,     # initial simplex size, in log10 units
                           seed = CRN_SEED):
    """
    DIRECT (deterministic global search) over the whole log10 box,
    then a bounded Nelder-Mead refinement starting from the best DIRECT point.
    
    Returns (best parameters in linear space, best objective, history), where history
    lists (stage, objective, parameters) for every evaluation.
    """
    
    history = []
    
    def f(log_params, stage):
        value = crn_objective(log_params, seed = seed)
        history.append((stage, value, 10**np.clip(log_params, *np.array(LOG_BOUNDS).T)))
        print(f"{stage} eval {len(history):3d} | objective = {value:.5f} | "
              + ", ".join(f"{n} = {v:.4f}" for n, v in zip(PARAM_NAMES, history[-1][2])),
              flush = True)
        return value
    
    direct_result = direct(f, LOG_BOUNDS, args = ("DIRECT",),
                           maxfun = direct_maxfun, locally_biased = True)
    
    x0 = direct_result.x
    lo, hi = np.array(LOG_BOUNDS).T
    # step each parameter towards the interior so the simplex isn't flattened against a bound
    steps = np.where(x0 + nm_step <= hi, nm_step, -nm_step)
    simplex = np.vstack([x0] + [x0 + steps[i] * np.eye(len(x0))[i] for i in range(len(x0))])
    
    minimize(f, x0, args = ("Nelder-Mead",), method = "Nelder-Mead",
             bounds = LOG_BOUNDS,
             options = dict(initial_simplex = simplex, maxfev = nm_maxfev,
                            xatol = 1e-3, fatol = 1e-5))
    
    best = min(history, key = lambda h: h[1])
    
    return best[2], best[1], history

def validate_candidate(params, seeds = (1, 2, 3, 4, 5)):
    """
    Re-evaluate a candidate with fresh random seeds (not the CRN seed it was optimised on).
    Returns the instability_distance_M result dict for each seed.
    """
    
    results = []
    for seed in seeds:
        np.random.seed(seed)
        results.append(instability_distance_M(*params,
                                              timeout = SIMULATION_TIMEOUT,
                                              eval_timeout = EVALUATION_TIMEOUT))
    
    return results

if __name__ == "__main__":
    best_params, best_f, history = run_direct_nelder_mead()

    print("\nBest parameters:", dict(zip(PARAM_NAMES, best_params)))
    print(f"Best objective (CRN seed {CRN_SEED}): {best_f:.5f}")

    # the optimum is tuned to one seed, so re-score it with fresh seeds
    for seed, result in zip((1, 2, 3, 4, 5), validate_candidate(best_params)):
        print(f"fresh seed {seed}: min_is_M = {result['min_is_M']:.5f}, "
              f"min_sc_M = {result['min_sc_M']:.1f}, feasible Ms = {result['feasible_Ms']}")

    plt.plot([h[1] for h in history], ".")
    plt.yscale("log")
    plt.xlabel("evaluation")
    plt.ylabel("objective (penalised if infeasible)")
    plt.show()
