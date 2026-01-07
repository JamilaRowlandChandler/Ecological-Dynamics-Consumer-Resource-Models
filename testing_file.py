# -*- coding: utf-8 -*-
"""
Created on Mon May  5 18:08:45 2025

@author: jamil

"""

import os

os.chdir("C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
         "Ecological-Dynamics/Consumer-Resource Models/consumer_resource_modules")

from models import Consumer_Resource_Model
from community_level_properties import max_le, max_le_2

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# %%

community = Consumer_Resource_Model("Self-limiting resource supply", 100, 100)

community.growth_consumption_rates('coupled by rho',
                                   1, 0.1, 1, 0.1, rho = 1)
community.model_specific_rates()

community.simulate_community(3500, 1)

community.calculate_community_properties()
#community.lyapunov_exponent = max_le(community, 500, community.ODE_sols[0].y[:, -1],
#                                     1e-3, dt = 20, separation = 1e-3)

# %%

M = 100
S = M

community = Consumer_Resource_Model("Self-limiting resource supply",
                                    S, M)

community.growth_consumption_rates('growth function of consumption',
                                   160/M, 1.6/np.sqrt(M), 1, 0.131)
community.model_specific_rates('constant', {'d' : 1},
                               'constant', {'b' : 1})

# %%

community.simulate_community(7000, 1)

# %%

plt.plot(community.ODE_sols[0].t, community.ODE_sols[0].y[:S, :].T)
plt.show()

# %%

window_size = 75
weights = np.ones(window_size) / window_size

plt.plot(community.ODE_sols[0].t,
         np.array([np.convolve(community.ODE_sols[0].y[:S, i].T, weights, mode = "valid") 
                   for i in range(len(community.ODE_sols[0].t))]))
plt.show()

# %%

#community.calculate_community_properties()
community.lyapunov_exponent = max_le_2(community, community.ODE_sols[0].y[:, -1],
                                       T = 600)

# %%

M = 100

community = Consumer_Resource_Model("Externally-supplied resources, toxins",
                                    M, M, M)

community.growth_consumption_rates('coupled by rho',
                                   1, 0.1, 1, 0.1,
                                   rho = 0.3)
community.model_specific_rates(produce_method = 'normal',
                               produce_args = {'mu' : 0.3/M,
                                               'sigma' : 0.01/np.sqrt(M)},
                               attack_method = 'normal',
                               attack_args = {'mu' : 200/M,
                                             'sigma' : 3/np.sqrt(M)},
                               influx_method = 'constant',
                               influx_args = {'b' : 1})

# %%
community.simulate_community(3500, 1)

# %%

plt.plot(community.ODE_sols[0].t, community.ODE_sols[0].y[200:, :].T)
plt.show()

# %%

community.calculate_community_properties()
community.lyapunov_exponent = max_le(community, community.ODE_sols[0].y[:, -1],
                                     T = 1000, perturbation = 1e-6)
print(community.lyapunov_exponent)
