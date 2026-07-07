# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:42:39 2026

@author: jamil
"""

import numpy as np
import sys
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# %%

abspath = os.path.abspath(__file__)
file_directory_name = os.path.dirname(abspath)
os.chdir(file_directory_name)

sys.path.insert(0, file_directory_name.removesuffix("\\external_resource_stability\\figures") + \
                "\\consumer_resource_modules")
from models import Consumer_Resource_Model

# %%

def load_clean_simulations(data_location):

    def prop_feasible(x,
                      feasibility_threshold = 1000):
        
        return np.count_nonzero(x == feasibility_threshold)/len(x)
        
            
    def prop_stable(x,
                    stability_threshold = 0):
        
        return np.count_nonzero(x < stability_threshold)/len(x)
    
    full_location = "C:/Users/jamil/Documents/PhD/Data/external_resource_stability/simulations/" + \
                        data_location
    
    if full_location.endswith(".csv"):
    
        df = pd.read_csv(full_location, index_col=False)
            
    else: 
       
        df = pd.concat([pd.read_csv(full_location + "/" + file, index_col=False) 
                       for file in os.listdir(full_location)],
                       axis = 0, ignore_index = True) 
        
    df = df.apply(pd.to_numeric, errors="coerce")
    
    df.rename(columns = {"maxLe" : "Max. lyapunov exponent"}, inplace = True)
    df = np.round(df, 7)

    stable_feasible = df.groupby('b_val').agg({'Divergence measure' : prop_feasible,
                                               'Max. lyapunov exponent' : prop_stable}).reset_index().rename(columns = {'b_val' : 'influx',
                                                                                                                        'Divergence measure' : 'P(Feasible)',
                                                                                                                        'Max. lyapunov exponent' : 'P(Stable)'})
    
    return df, stable_feasible

# %%

simulations, stable_feasible = load_clean_simulations("influx_outflux")

# %%

no_species = 150
no_resources = 150
mu = 50
sigma = 10.5
mu_d = 1
sigma_d = 0
mu_b = 10**-5
sigma_b = 0.0
o = 10**-5
rho = 0.8

t_end = 7000
no_init_conds = 1


crm_community = Consumer_Resource_Model("Externally-supplied resources",
                                        pool_sizes = [no_resources, no_species])

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

# %%

mosaic = [["P(stablefeasible)", ".", "sim_R"],
          ["P(stablefeasible)", ".", "sim_C"]]
   
fig, axs = plt.subplot_mosaic(mosaic,
                              figsize = (5, 2.5),
                              width_ratios = [1, 0.3, 1],
                              height_ratios = [1, 1],
                              gridspec_kw = {'hspace' : 0.2, 'wspace' : 0})

sns.lineplot(x = np.log10(stable_feasible['influx']),
             y = stable_feasible['P(Feasible)'],
             ax = axs["P(stablefeasible)"],
             color = 'black', linewidth = 3, linestyle = "--")

sns.lineplot(x = np.log10(stable_feasible['influx']),
             y = stable_feasible['P(Stable)'],
             ax = axs["P(stablefeasible)"],
             color = 'black', linewidth = 3)

axs["sim_R"].plot(crm_community.ODE_sols[0].t,
                  crm_community.ODE_sols[0].y[no_species:, :].T)
axs["sim_C"].plot(crm_community.ODE_sols[0].t,
                  crm_community.ODE_sols[0].y[:no_species, :].T)

axs["sim_R"].set_xlim([2000, 7000])
axs["sim_C"].set_xlim([2000, 7000])

plt.savefig("C:/Users/jamil/Documents/PhD/Figures/externally_supplied_resources/external_supply_chaos.png",
            dpi=300, bbox_inches='tight')
plt.savefig("C:/Users/jamil/Documents/PhD/Figures/externally_supplied_resources/external_supply_chaos.svg",
            bbox_inches='tight')

plt.show()
    
