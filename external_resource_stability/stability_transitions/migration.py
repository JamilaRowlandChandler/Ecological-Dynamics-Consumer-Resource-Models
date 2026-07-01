# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:28:07 2026

@author: jamil
"""

import numpy as np
import sys
import os
from copy import deepcopy
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# %%

abspath = os.path.abspath(__file__)
file_directory_name = os.path.dirname(abspath)
os.chdir(file_directory_name)

sys.path.insert(0, file_directory_name.removesuffix("\\stability_transitions"))
from simulation_functions_new import CRM_across_parameter_space

sys.path.insert(0,  file_directory_name.removesuffix("\\external_resource_stability\\stability_transitions") + \
                "\\cavity_method_functions")
from self_consistency_equation_functions import variable_fixed_parameters

# %%

def migration(migration_exponents,
              fixed_parameters,
              subdirectory = 'external_resource_stability/simulations/influx_outflux',
              save_method = 'v3',
              **kwargs):
    
    parameters = generate_parameters(migration_exponents,
                                     fixed_parameters)
    
    CRM_across_parameter_space(parameters,
                               subdirectory,
                               ['exponent', 'exponent'],
                               save_method=save_method,
                               model = 'Externally-supplied resources',
                               **kwargs)
    
# %%

def generate_parameters(migration_exponents, fixed_parameters):
    
    migration_rates = 10**(-migration_exponents)
    
    fixed_parameters_mod = deepcopy(fixed_parameters)
    
    fixed_parameters_mod['mu_c'] *= 1/fixed_parameters_mod['M']
    fixed_parameters_mod['mu_g'] *= 1/fixed_parameters_mod['M']
    fixed_parameters_mod['sigma_c'] *= 1/np.sqrt(fixed_parameters_mod['M'])
    fixed_parameters_mod['sigma_g'] *= 1/np.sqrt(fixed_parameters_mod['M'])

    # array of all parameter combinations
    parameters = variable_fixed_parameters(np.vstack([migration_exponents,
                                                      migration_rates,
                                                      migration_rates]),
                                           fixed_parameters_mod,
                                           ['exponent', 'b', 'o'])

    return parameters

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

migration_exponents = np.arange(0, 8, 0.5)
mu = 50
sigma = 10.5
rho = 0.4

# %%

migration(migration_exponents,
          dict(mu_c = mu, sigma_c = sigma,
               mu_g = mu, sigma_g = sigma, rho = rho,
               d = 1, M = 150, S = 150),
          no_communities = 20,
          t_end = 1000,
          no_init_conds = 1)

# %%

simulations, stable_feasible = load_clean_simulations("influx_outflux")

# %%

fig, ax = plt.subplots(1, 1)

sns.lineplot(x = np.log10(stable_feasible['influx']),
             y = stable_feasible['P(Feasible)'],
             color = 'gray', ax = ax)

sns.lineplot(x = np.log10(stable_feasible['influx']),
             y = stable_feasible['P(Stable)'],
             color = 'black', ax = ax)

plt.show()
    
    