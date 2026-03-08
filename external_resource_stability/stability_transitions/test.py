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
from typing import Literal

# %%

abspath = os.path.abspath(__file__)
file_directory_name = os.path.dirname(abspath)
os.chdir(file_directory_name)

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/consumer_resource_modules')
from models import Consumer_Resource_Model
from effective_LV_models import Effective_LV_Model
from community_level_properties import max_le

# %%

no_species = 100
no_resources = 100
mu = 1
sigma = 3.1
mu_d = 1
sigma_d = 0.1
mu_b = 1
sigma_b = 0.1
o = 1
rho = 0.9

t_end = 4000
no_init_conds = 2


crm_community = Consumer_Resource_Model("Externally-supplied resources",
                                        no_species, no_resources)

# generate model parameters
crm_community.growth_consumption_rates(method = 'coupled by rho',
                                       mu_c = mu/no_resources,
                                       sigma_c = sigma/np.sqrt(no_resources),
                                       mu_g = mu/no_resources,
                                       sigma_g = sigma/np.sqrt(no_resources),
                                       rho = rho)

crm_community.model_specific_rates(death_method='normal',
                                   death_args={'mu' : mu_d, 'sigma' : sigma_d},
                                   influx_method='normal',
                                   influx_args={'mu' : mu_b, 'sigma' : sigma_b},
                                   outflux_method='constant',
                                   outflux_args={'o' : o})

crm_community.simulate_community(t_end, no_init_conds)

# estimate community properties, including the max. lyapunov exponent
#crm_community.calculate_community_properties()
#crm_community.lyapunov_exponent = max_le(crm_community,
#                                         crm_community.ODE_sols[0].y[:, -1],
#                                         T = 1000,
#                                         perturbation = 1e-6)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8, 5))

ax1.plot(crm_community.ODE_sols[0].t,
         crm_community.ODE_sols[0].y[:no_species, :].T)

ax2.plot(crm_community.ODE_sols[1].t,
         crm_community.ODE_sols[1].y[:no_species, :].T)

plt.show()


# %%

def CRM_eLV(rho):
    
    ### CRM ###

    crm_community = Consumer_Resource_Model("Externally-supplied resources",
                                            no_species, no_resources)

    # generate model parameters
    crm_community.growth_consumption_rates(method = 'coupled by rho',
                                           mu_c = mu/no_resources,
                                           sigma_c = sigma/np.sqrt(no_resources),
                                           mu_g = mu/no_resources,
                                           sigma_g = sigma/np.sqrt(no_resources),
                                           rho = rho)
    crm_community.model_specific_rates()

    # simulate commmunity dynamics
    crm_community.simulate_community(t_end, no_init_conds)

    # estimate community properties, including the max. lyapunov exponent
    #crm_community.calculate_community_properties()
    #crm_community.lyapunov_exponent = max_le(crm_community,
    #                                         crm_community.ODE_sols[0].y[:, -1],
    #                                         T = 1000,
    #                                         perturbation = 1e-6)
    
    

    ### eLV ###

    elv_community = Effective_LV_Model("Externally-supplied resources") # ,
    #                                   no_species = no_species,
    #                                   no_resources = no_resources)
    
    elv_community.elv_from_crm(crm_community)
    #elv_community.growth_consumption_rates(method = 'coupled by rho',
    #                                       mu_c = mu/no_resources,
    #                                       sigma_c = sigma/np.sqrt(no_resources),
    #                                       mu_g = mu/no_resources,
    #                                       sigma_g = sigma/np.sqrt(no_resources),
    #                                       rho = rho)
    #elv_community.model_specific_rates()
    
    elv_community.generate_elv_parameters()

    elv_community.simulate_community(t_end, 1, "user-supplied",
                                     user_supplied_init_cond = {'species' : crm_community.ODE_sols[0].y[:crm_community.no_species, -1]})

    return crm_community, elv_community
    
    #return elv_community

# %%

#elv_highrho = CRM_eLV(rhos[0])
#elv_lowrho = CRM_eLV(rhos[1])
crm_highrho, elv_highrho = CRM_eLV(rhos[0])

# %%

np.mean(crm_highrho.ODE_sols[0].y[crm_highrho.no_species:, -1] * \
        np.sum(crm_highrho.consumption * \
               crm_highrho.ODE_sols[0].y[:crm_highrho.no_species, -1], axis=1))

# %%
crm_lowrho, elv_lowrho = CRM_eLV(rhos[1])

# %%

fig, axs = plt.subplots(2, 2, figsize = (8, 5))

for model, ax in zip([crm_highrho, elv_highrho], #crm_lowrho, elv_lowrho],
                     axs.flatten()):
    
    ax.plot(model.ODE_sols[0].t, model.ODE_sols[0].y.T)

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

# %%

#############################

def rep():

    no_species = 200
    no_resources = 200
    mu = 1.5
    sigma = 1
    rho = 1
    mu_d = 1
    sigma_d = 0.1
    mu_b = 1
    sigma_b = 0.1
    
    
    crm_community = Consumer_Resource_Model("Externally-supplied resources",
                                            no_species, no_resources)
    
    # generate model parameters
    crm_community.growth_consumption_rates(method = 'coupled by rho',
                                           mu_c = mu/no_resources,
                                           sigma_c = sigma/np.sqrt(no_resources),
                                           mu_g = mu/no_resources,
                                           sigma_g = sigma/np.sqrt(no_resources),
                                           rho = rho)
    
    crm_community.model_specific_rates(death_method = "normal",
                                       death_args = {'mu' : mu_d, 'sigma' : sigma_d},
                                       influx_method = "normal",
                                       influx_args = {'mu' : mu_b, 'sigma' : sigma_b})
    
    # simulate commmunity dynamics
    crm_community.simulate_community(4000, 1)

    # estimate community properties, including the max. lyapunov exponent
    crm_community.calculate_community_properties()
    
    return dict(phi_N = crm_community.species_survival_fraction[0],
                N_mean = crm_community.species_avg_abundance[0],
                qN = crm_community.species_abundance_fluctuations[0],
                Rmean = crm_community.resource_avg_abundance[0],
                qR = crm_community.resource_abundance_fluctuations[0])

# %%

community_rep = pd.DataFrame([rep() for _ in range(10)])

print(community_rep.mean())

# %%

A = pd.read_csv("C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/external_resource_stability/stability_transitions/TestDF.csv",
                index_col=False)





























