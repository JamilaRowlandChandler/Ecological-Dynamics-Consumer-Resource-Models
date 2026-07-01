# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:30:12 2026

@author: jamil
"""

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
from simulation_functions_new import CRM_across_parameter_space, le_pivot_r

sys.path.insert(0,  file_directory_name.removesuffix("\\external_resource_stability\\stability_transitions") + \
                "\\cavity_method_functions")
from self_consistency_equation_functions import variable_fixed_parameters, \
    parameter_combinations

# %%

def migration(migration_exponents,
              sigmas,
              fixed_parameters,
              subdirectory,
              save_method = 'v3',
              **kwargs):

    parameters = generate_parameters(migration_exponents,
                                     sigmas,
                                     fixed_parameters)
    
    CRM_across_parameter_space(parameters,
                               subdirectory,
                               ['exponent', 'sigma_M'],
                               save_method=save_method,
                               model = 'Hybrid resource supply',
                               **kwargs)
    
# %%

def generate_parameters(migration_exponents,
                        sigmas,
                        fixed_parameters):
    
    migration_rates = 10**(-migration_exponents)
    
    sigma_migration_combos = np.unique(parameter_combinations([sigmas,
                                                               migration_rates],
                                                              1),
                                       axis = 1)
    
    fixed_parameters_mod = deepcopy(fixed_parameters)
    
    fixed_parameters_mod['mu_c'] *= 1/fixed_parameters_mod['M']
    fixed_parameters_mod['mu_g'] *= 1/fixed_parameters_mod['M']

    # array of all parameter combinations
    parameters = variable_fixed_parameters(np.vstack([sigma_migration_combos,
                                                      sigma_migration_combos[0, :]/np.sqrt(fixed_parameters_mod['M']),
                                                      sigma_migration_combos[0, :]/np.sqrt(fixed_parameters_mod['M']),
                                                      np.abs(np.log10(sigma_migration_combos[1, :]))]),
                                           fixed_parameters_mod,
                                           ['sigma_M',
                                            'b',
                                            'sigma_c',
                                            'sigma_g',
                                            'exponent'])
    
    return parameters

# %%

def load_clean_simulations(data_location):

    def prop_feasible(x,
                      feasibility_threshold = 1000):
        
        return np.count_nonzero(x == feasibility_threshold)/len(x)
    
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

    stable_pivot = le_pivot_r(df,
                              index = "b_val",
                              columns = "sigma_c")[0]
    
    feasible_pivot = pd.pivot_table(df,
                                    index = "b_val",
                                    columns = "sigma_c",
                                    values = "Divergence measure",
                                    aggfunc = prop_feasible)                                                                                                             
    
    return df, stable_pivot, feasible_pivot

# %%

migration_exponents = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0])
mu = 50
sigmas = np.arange(2, 13, 1.0)
rhos = [0.4, 0.8]

# %%

migration(migration_exponents,
          sigmas,
          dict(mu_c = mu, mu_g = mu, rho = rhos[0],
               d = 1, o = 1, M = 150, S = 150),
          "external_resource_stability/simulations/influx_hybrid_infeasible",
          no_communities = 20,
          t_end = 1000,
          no_init_conds = 1)

migration(migration_exponents,
          sigmas,
          dict(mu_c = mu, mu_g = mu, rho = rhos[1],
               d = 1, o = 1, M = 150, S = 150),
          "external_resource_stability/simulations/influx_hybrid_unstable",
          no_communities = 20,
          t_end = 1000,
          no_init_conds = 1)

# %%

simulations_in, stable_in, feasible_in = \
    load_clean_simulations("influx_hybrid_infeasible")

simulations_un, stable_un, feasible_un = \
    load_clean_simulations("influx_hybrid_unstable")
    
# %%

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True,
                               layout="constrained", figsize=(10, 3))

sns.heatmap(stable_in.mask(feasible_in < 1),
            cmap = "Purples_r",
            ax = ax1)
ax1.invert_yaxis()
ax1.set_facecolor('grey')

sns.heatmap(stable_un.mask(feasible_un < 1),
            cmap = "Purples_r",
            ax = ax2)
ax2.invert_yaxis()
ax2.set_facecolor('grey')

plt.show()