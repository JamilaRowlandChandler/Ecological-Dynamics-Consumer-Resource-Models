# -*- coding: utf-8 -*-
"""
Created on Mon May  5 18:08:45 2025

@author: jamil

"""

import os
import sys

os.chdir("C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/consumer_resource_modules")

from models import Consumer_Resource_Model
from community_level_properties import max_le

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/resource_diversity_stability(sl)')
from simulation_functions import simulation_df_from_communities
from matplotlib import pyplot as plt
import numpy as np

# %%

community = Consumer_Resource_Model("Self-limiting resource supply", pool_sizes=[150, 150])

community.growth_consumption_rates('coupled by rho',
                                   50/150, 3/np.sqrt(150),
                                   50/150, 3/np.sqrt(150), rho = 0.5)
community.model_specific_rates(death_method = "normal",
                               death_args = {'mu' : 1, 'sigma' : 0.1},
                               resource_growth_method = "normal",
                               resource_growth_args = {'mu' : 1, 'sigma' : 0.1})

community.simulate_community(1000, 1)

#community.calculate_community_properties() 

#community.lyapunov_exponent = max_le(community, community.ODE_sols[0].y[:, -1],
#                                         T = 1000, perturbation = 1e-6)
#print(community.lyapunov_exponent)

plt.plot(community.ODE_sols[0].t, community.ODE_sols[0].y[:150, :].T)
plt.show()

# %%

M = 150
mu = 100
sigma = 3

community = Consumer_Resource_Model("Self-limiting resource supply, leached",
                                    pool_sizes = [M, M])

community.growth_consumption_rates('coupled by rho',
                                   mu/M, sigma/np.sqrt(M),
                                   mu/M, sigma/np.sqrt(M),
                                   rho = 1)
community.model_specific_rates(death_method = "normal",
                               death_args = {'mu' : 1.0, 'sigma' : 0.0},
                               resource_growth_method = "normal",
                               resource_growth_args = {'mu' : 1.0,
                                                       'sigma' : 0.0},
                               resource_interaction_method = "normal",
                               resource_interaction_args = {'mu' : 0.3,
                                                            'sigma' : 0.3})

community.simulate_community(7000, 1)

#community.calculate_community_properties() 

#community.lyapunov_exponent = max_le(community, community.ODE_sols[0].y[:, -1],
#                                         T = 1000, perturbation = 1e-6)
#print(community.lyapunov_exponent)

plt.plot(community.ODE_sols[0].t,
         community.ODE_sols[0].y[:M, :].T)
plt.show()


# %%

M = 180
mu_c = 200

community = Consumer_Resource_Model("Self-limiting resource supply, multi-trophic level",
                                    pool_sizes = [M, M, M])

community.growth_consumption_rates('growth function of consumption',
                                   mu_c/M,
                                   1.6/np.sqrt(M),
                                   1,
                                   0.13069,
                                   trophic_level = 2)
community.growth_consumption_rates('growth function of consumption',
                                   mu_c/M,
                                   1.6/np.sqrt(M),
                                   1.0,
                                   0.0,
                                   trophic_level = 3)

community.model_specific_rates(death_methods = ['constant', 'constant'],
                               death_args = [{'d_2' : 1}, {'d_3' : 0.93}],
                               resource_interaction_method = 'normal',
                               resource_interaction_args = {'mu' : 10/M, 
                                                            'sigma' : 0.5/np.sqrt(M)}) 

community.simulate_community(9000, 1)

fig, axs = plt.subplots(1, 3, figsize = (8, 2.5), layout = "constrained")
    
axs[0].plot(community.ODE_sols[0].t, community.ODE_sols[0].y[:M, :].T)
axs[1].plot(community.ODE_sols[0].t, community.ODE_sols[0].y[M:-M, :].T)
axs[2].plot(community.ODE_sols[0].t, community.ODE_sols[0].y[-M:, :].T)

plt.show()


community.calculate_community_properties() 

community.lyapunov_exponent = max_le(community, community.ODE_sols[0].y[:, -1],
                                         T = 1000, perturbation = 1e-6)

print(community.lyapunov_exponent)

# %%

community_df = simulation_df_from_communities([community, community],
                                              "Self-limiting resource supply, multi-trophic level",
                                              "growth function of consumption")
