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
    
    community.simulate_community(7000, 1)
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
                           no_communities = 10):
    
    is_sc_M = [[instability_distance(M, mu_c, sigma_c, mu_y, sigma_y)
                for _ in range(no_communities)]
               for M in Ms]
    
    is_M, sc_M = unpack_is_sc(is_sc_M)
    
    min_is_M = np.min(np.abs(is_M))
    min_sc_M = np.min(sc_M)
    
    return min_is_M, min_sc_M
    