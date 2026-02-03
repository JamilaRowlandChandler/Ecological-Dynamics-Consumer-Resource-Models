# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 16:30:48 2026

@author: jamil
"""

import numpy as np
import sys
import os
from copy import deepcopy
import pandas as pd
from matplotlib import pyplot as plt

# %%

abspath = os.path.abspath(__file__)
file_directory_name = os.path.dirname(abspath)
os.chdir(file_directory_name)

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/consumer_resource_modules')
from models import Consumer_Resource_Model
from community_level_properties import max_le

# %%

no_species = 500
no_resources = 100
mu = 3
sigma = 5
b = 5

t_end = 1000
no_init_conds = 1

# %%

community = Consumer_Resource_Model("Externally-supplied resources",
                                    no_species, no_resources)

# generate model parameters
community.growth_consumption_rates(method = 'coupled by rho',
                                   mu_c = mu/no_resources,
                                   sigma_c = sigma/np.sqrt(no_resources),
                                   mu_g = mu/no_resources,
                                   sigma_g = sigma/np.sqrt(no_resources),
                                   rho = 0.75)
community.model_specific_rates(influx_method = 'constant',
                               influx_args = {'b' : b})

# simulate commmunity dynamics
community.simulate_community(t_end, no_init_conds)

# estimate community properties, including the max. lyapunov exponent
community.calculate_community_properties()
community.lyapunov_exponent = max_le(community, community.ODE_sols[0].y[:, -1],
                                     T = 1000, perturbation = 1e-6)

print(np.array(community.species_survival_fraction) * (no_species/no_resources))

# %%

print(dict(phi_N = community.species_survival_fraction[0],
           N_mean = community.species_avg_abundance[0],
           q_N = community.species_abundance_fluctuations[0],
           R_mean = community.resource_avg_abundance[0],
           q_R = community.resource_abundance_fluctuations[0]))
    
# %%

plt.plot(community.ODE_sols[0].t,
         community.ODE_sols[0].y[:no_species, :].T)

plt.show()

# %%

community_sl = Consumer_Resource_Model("Self-limiting resource supply",
                                       no_species, no_resources)

# generate model parameters
community_sl.growth_consumption_rates(method = 'coupled by rho',
                                   mu_c = mu/no_resources,
                                   sigma_c = sigma/np.sqrt(no_resources),
                                   mu_g = mu/no_resources,
                                   sigma_g = sigma/np.sqrt(no_resources),
                                   rho = 0.75)
community_sl.model_specific_rates(resource_growth_method = 'constant',
                                  resource_growth_args = {'b' : b})

# simulate commmunity dynamics
community_sl.simulate_community(t_end, no_init_conds)

# estimate community_sl properties, including the max. lyapunov exponent
community_sl.calculate_community_properties()
community_sl.lyapunov_exponent = max_le(community_sl,
                                        community_sl.ODE_sols[0].y[:, -1],
                                        T = 1000, perturbation = 1e-6)

print(np.array(community_sl.species_survival_fraction) * (no_species/no_resources))
    
# %%

plt.plot(community_sl.ODE_sols[0].t,
         community_sl.ODE_sols[0].y[:no_species, :].T)

plt.show()