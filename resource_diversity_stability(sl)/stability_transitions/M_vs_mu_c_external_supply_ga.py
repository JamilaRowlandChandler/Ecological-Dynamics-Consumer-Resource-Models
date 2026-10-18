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

EXTINCTION_THRESHOLD = 1e-4   # used only to detect communities where all consumers are extinct

def instability_distance(M,
                         mu_c, sigma_c,
                         mu_y, sigma_y,
                         timeout = None):
    """
    timeout : wall-clock limit in seconds for the ODE integration of this one
    community (None = no limit). Raises SimulationTimeout if exceeded.
    """
    
    community = Consumer_Resource_Model("Externally-supplied resources",
                                        M,
                                        M)
    
    community.growth_consumption_rates('growth function of consumption',
                                       mu_c = mu_c/M,
                                       sigma_c = sigma_c/np.sqrt(M),
                                       mu_g = mu_y,
                                       sigma_g = sigma_y)
    
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
    # status != 0 means solve_ivp stopped early (e.g. the unbounded-growth event).
    # With t_eval only t = 0 is stored in that case, so the community would look like
    # every species/resource survived and the metrics would be meaningless.
    if community.ODE_sols[0].status != 0:
        raise SimulationFailed(f"solver stopped early: {community.ODE_sols[0].message}")
    
    # discard communities where every consumer has gone extinct: rho**2 and the packing
    # ratio are then both ~0, and a survivor count that is relative to the abundance
    # range (e.g. a histogram of final abundances) would wrongly report ~M survivors
    if np.all(community.ODE_sols[0].y[:M, -1] < EXTINCTION_THRESHOLD):
        raise SimulationFailed("all consumers extinct")
    
    community.calculate_community_properties()
    
    rho = rho_calculation(M, mu_c, sigma_c, mu_y, sigma_y)
    packing_ratio = community.species_survival_fraction[0]/community.resource_survival_fraction[0]
    instability_distance = rho**2 - packing_ratio
    
    final_consumer_distribution, _ = np.histogram(community.ODE_sols[0].y[:M, -1],
                                                  bins=M)
    surviving_consumers = M - final_consumer_distribution[0]
    #t_end_act = community.ODE_sols[0].t[-1]
    
    return instability_distance, surviving_consumers #, t_end_act

def unpack_is_sc(is_sc_M):
    
    # mean of |instability distance| over communities, so that positive and negative
    # values from different communities cannot cancel each other out
    is_M = [np.mean([np.abs(is_sc_comm[0])
                     for is_sc_comm in is_cs])
            for is_cs in is_sc_M]
    
    sc_M = [np.mean([is_sc_comm[1]
                     for is_sc_comm in is_cs])
            for is_cs in is_sc_M]
    
    return is_M, sc_M

def instability_distance_M(mu_c, sigma_c,
                           mu_y, sigma_y,
                           Ms = np.arange(50, 275, 25),
                           no_communities = 5,
                           timeout = None,
                           eval_timeout = None):
    """
    timeout : wall-clock limit (s) for each single community simulation.
    eval_timeout : wall-clock limit (s) for this whole call (all Ms x communities);
    each simulation gets whatever is left of it, so a call can't run for hours
    when many simulations each take just under `timeout`.
    """
    
    deadline = None if eval_timeout is None else time.time() + eval_timeout
    
    def sim_timeout():
        if deadline is None:
            return timeout
        remaining = deadline - time.time()
        if remaining <= 0:
            raise SimulationTimeout(f"evaluation exceeded {eval_timeout} s")
        return remaining if timeout is None else min(timeout, remaining)
    
    is_sc_M = [[instability_distance(M, mu_c, sigma_c, mu_y, sigma_y,
                                     timeout = sim_timeout())
                for _ in range(no_communities)]
               for M in Ms]
    
    is_M, sc_M = unpack_is_sc(is_sc_M)
    
    min_is_M = np.min(is_M)
    min_sc_M = np.min(sc_M)

    return min_is_M, min_sc_M

#%% Objective and constraint

# Parameter order: [mu_c, sigma_c, mu_y, sigma_y].
# All four must be > 0, enforced through strictly positive lower bounds.
# The upper bounds are placeholders - adjust to the range you want to explore.
PARAM_NAMES = ["mu_c", "sigma_c", "mu_y", "sigma_y"]
BOUNDS = [(1e-3, 400.0),  # mu_c
          (1e-3, 400.0),  # sigma_c
          (1e-3, 5.0),    # mu_y
          (1e-3, 3.0)]    # sigma_y

MIN_SURVIVING_CONSUMERS = 4   # constraint: min_sc_M > 4
SIMULATION_TIMEOUT = 60       # seconds allowed per single community simulation
EVALUATION_TIMEOUT = 600      # seconds allowed per objective evaluation (all Ms x communities)
INFEASIBLE_BASE = 10.0        # added to infeasible candidates so that they always
                              # score worse than any feasible one (|instability
                              # distance| is O(1), so any value >> 1 works)

def constrained_objective(params,
                          Ms = np.arange(50, 275, 25),
                          no_communities = 5,
                          timeout = SIMULATION_TIMEOUT,
                          eval_timeout = EVALUATION_TIMEOUT):
    """
    Objective to minimise: min_is_M, subject to min_sc_M > 4.

    The constraint is handled with a penalty rather than scipy's
    NonlinearConstraint, because a NonlinearConstraint evaluates the (expensive)
    simulation a second time per candidate. Here each candidate is simulated once.
    Infeasible candidates get INFEASIBLE_BASE + shortfall, so the search is
    still pushed towards the feasible region (larger min_sc_M) rather than
    seeing a flat plateau.
    """

    mu_c, sigma_c, mu_y, sigma_y = params

    try:
        min_is_M, min_sc_M = instability_distance_M(mu_c, sigma_c,
                                                    mu_y, sigma_y,
                                                    Ms = Ms,
                                                    no_communities = no_communities,
                                                    timeout = timeout,
                                                    eval_timeout = eval_timeout)
    except Exception:
        # timed-out simulation (SimulationTimeout), failed ODE solve or other
        # numerical problem -> treat as worst-case candidate
        return INFEASIBLE_BASE + MIN_SURVIVING_CONSUMERS

    if not np.isfinite(min_is_M) or not np.isfinite(min_sc_M):
        return INFEASIBLE_BASE + MIN_SURVIVING_CONSUMERS

    if min_sc_M <= MIN_SURVIVING_CONSUMERS:
        return INFEASIBLE_BASE + (MIN_SURVIVING_CONSUMERS - min_sc_M)

    return min_is_M

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
    Stage 1 protocol: DIRECT (deterministic global search) over the whole log10 box,
    then a bounded Nelder-Mead refinement starting from the best DIRECT point.
    Both stages use the same CRN seed. Runs serially (each stage is sequential).
    
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
    Returns a list of (min_is_M, min_sc_M) per seed, or the exception name if it failed.
    """
    
    results = []
    for seed in seeds:
        np.random.seed(seed)
        try:
            results.append(instability_distance_M(*params,
                                                  timeout = SIMULATION_TIMEOUT,
                                                  eval_timeout = EVALUATION_TIMEOUT))
        except Exception as e:
            results.append(type(e).__name__)
    
    return results

if __name__ == "__main__":
    best_params, best_f, history = run_direct_nelder_mead()

    print("\nBest parameters:", dict(zip(PARAM_NAMES, best_params)))
    print(f"Best objective (CRN seed {CRN_SEED}): {best_f:.5f}")

    # the optimum is tuned to one seed, so re-score it with fresh seeds
    for seed, result in zip((1, 2, 3, 4, 5), validate_candidate(best_params)):
        print(f"fresh seed {seed}: (min_is_M, min_sc_M) = {result}")

    plt.plot([h[1] for h in history], ".")
    plt.yscale("log")
    plt.xlabel("evaluation")
    plt.ylabel("objective (penalised if infeasible)")
    plt.show()
