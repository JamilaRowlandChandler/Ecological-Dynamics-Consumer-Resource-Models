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
from scipy.optimize import differential_evolution, dual_annealing

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

#%% Genetic algorithm (differential evolution)

# Parameter order in the DE vector: [mu_c, sigma_c, mu_y, sigma_y].
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

def ga_objective(params,
                 Ms = np.arange(50, 275, 25),
                 no_communities = 5,
                 timeout = SIMULATION_TIMEOUT,
                 eval_timeout = EVALUATION_TIMEOUT):
    """
    Objective for differential_evolution: min_is_M, subject to min_sc_M > 4.

    The constraint is handled with a penalty rather than scipy's
    NonlinearConstraint, because a NonlinearConstraint evaluates the (expensive)
    simulation a second time per candidate. Here each candidate is simulated once.
    Infeasible candidates get INFEASIBLE_BASE + shortfall, so the population is
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

def run_ga(bounds = BOUNDS,
           popsize = 10,          # population = popsize * 4 parameters
           maxiter = 50,
           seed = None,
           workers = 1,           # >1 parallelises candidates (needs the __main__ guard below)
           polish = False):       # L-BFGS-B polish is pointless on a noisy stochastic objective

    history = []   # best objective value per generation

    def callback(intermediate_result):
        history.append(intermediate_result.fun)
        print(f"gen {len(history):3d} | best objective = {intermediate_result.fun:.5f} | "
              + ", ".join(f"{n} = {v:.4f}"
                          for n, v in zip(PARAM_NAMES, intermediate_result.x)))

    result = differential_evolution(ga_objective,
                                    bounds,
                                    strategy = "best1bin",
                                    popsize = popsize,
                                    maxiter = maxiter,
                                    mutation = (0.5, 1.0),
                                    recombination = 0.7,
                                    seed = seed,
                                    init = "sobol",
                                    polish = polish,
                                    workers = workers,
                                    updating = "deferred" if workers != 1 else "immediate",
                                    callback = callback)

    return result, history

#%% Simulated annealing (dual_annealing)

def run_sa(seed = None,
           maxfun = 60,      # budget of objective evaluations for this chain
           bounds = BOUNDS):
    """
    One simulated-annealing chain (scipy's generalised SA, dual_annealing).
    
    no_local_search = True: dual_annealing normally finishes with L-BFGS-B local
    searches, which use finite-difference gradients and are meaningless on a noisy,
    discontinuous objective (and would burn evaluations).
    
    dual_annealing has no `workers` option, so run several independent chains in
    parallel with run_sa_chains().
    """
    
    history = []   # best objective value found so far, at each improvement
    
    def callback(x, f, context):
        history.append(f)
        print(f"[seed {seed}] new best {f:.5f} | "
              + ", ".join(f"{n} = {v:.4f}" for n, v in zip(PARAM_NAMES, x)),
              flush = True)
    
    result = dual_annealing(ga_objective,
                            bounds,
                            maxfun = maxfun,
                            seed = seed,
                            no_local_search = True,
                            callback = callback)
    
    return result, history

def run_sa_chains(seeds = range(8),
                  maxfun = 60):
    """Independent SA chains in parallel; returns the results sorted best-first."""
    
    from concurrent.futures import ProcessPoolExecutor
    
    with ProcessPoolExecutor(max_workers = len(seeds)) as pool:
        futures = [pool.submit(run_sa, seed, maxfun) for seed in seeds]
        results = [future.result()[0] for future in futures]
    
    return sorted(results, key = lambda r: r.fun)

if __name__ == "__main__":
    # guard is required on Windows when workers != 1 (multiprocessing re-imports this file)
    result, history = run_ga(seed = 0, workers = 1)

    best = dict(zip(PARAM_NAMES, result.x))
    print("\nBest parameters:", best)

    # re-evaluate the best candidate to report both outputs (stochastic, so values will differ slightly)
    min_is_M, min_sc_M = instability_distance_M(*result.x)
    print(f"Re-evaluated: min_is_M = {min_is_M:.5f}, min_sc_M = {min_sc_M:.2f}")

    plt.plot(history)
    plt.xlabel("generation")
    plt.ylabel("best min_is_M")
    plt.show()
