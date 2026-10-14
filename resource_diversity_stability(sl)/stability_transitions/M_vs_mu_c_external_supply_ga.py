# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:06:42 2026

@author: jamil
"""

import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from copy import deepcopy
from scipy.stats import pearsonr, norm
from scipy.optimize import differential_evolution

os.chdir("C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/consumer_resource_modules")

from models import Consumer_Resource_Model

def rho_calculation(M,
                    mu_c, sigma_c,
                    mu_y, sigma_y):
    
    return 1/np.sqrt(1 + ((sigma_y/mu_y)**2 * (mu_c**2 / (sigma_c**2 * M))))

def instability_distance(M,
                         mu_c, sigma_c,
                         mu_y, sigma_y):
    
    community = Consumer_Resource_Model("Externally-supplied resources",
                                        M,
                                        M)
    
    community.growth_consumption_rates('growth function of consumption',
                                       mu_c = mu_c/M,
                                       sigma_c = sigma_c/np.sqrt(M),
                                       mu_g = mu_y,
                                       sigma_g = sigma_y)
    
    community.model_specific_rates()
    
    community.simulate_community(2000, 1)
    community.calculate_community_properties()
    
    rho = rho_calculation(M, mu_c, sigma_c, mu_y, sigma_y)
    packing_ratio = community.species_survival_fraction[0]/community.resource_survival_fraction[0]
    instability_distance = rho**2 - packing_ratio
    
    surviving_consumers = community.species_survival_fraction[0]*M
    #t_end_act = community.ODE_sols[0].t[-1]
    
    return instability_distance, surviving_consumers #, t_end_act

def unpack_is_sc(is_sc_M):
    
    is_M = [np.mean([is_sc_comm[0]
                     for is_sc_comm in is_cs])
            for is_cs in is_sc_M]
    
    sc_M = [np.mean([is_sc_comm[1]
                     for is_sc_comm in is_cs])
            for is_cs in is_sc_M]
    
    return is_M, sc_M

def instability_distance_M(mu_c, sigma_c,
                           mu_y, sigma_y,
                           Ms = np.arange(50, 275, 25),
                           no_communities = 5):
    
    is_sc_M = [[instability_distance(M, mu_c, sigma_c, mu_y, sigma_y)
                for _ in range(no_communities)]
               for M in Ms]
    
    is_M, sc_M = unpack_is_sc(is_sc_M)
    
    min_is_M = np.min(np.abs(is_M))
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
INFEASIBLE_BASE = 10.0        # added to infeasible candidates so that they always
                              # score worse than any feasible one (|instability
                              # distance| is O(1), so any value >> 1 works)

def ga_objective(params,
                 Ms = np.arange(50, 275, 25),
                 no_communities = 5):
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
                                                    no_communities = no_communities)
    except Exception:
        # failed ODE solve / numerical problem -> treat as worst-case candidate
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
