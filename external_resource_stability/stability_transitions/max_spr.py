# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 18:04:36 2026

@author: jamil
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 14:24:01 2025

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
from simulation_functions import CRM_across_parameter_space, \
    generate_simulation_df, load_in_communities, le_pivot_r

sys.path.insert(0,  file_directory_name.removesuffix("\\external_resource_stability\\stability_transitions") + \
                "\\cavity_method_functions")
import self_consistency_equation_functions as sce

sys.path.insert(0,  file_directory_name.removesuffix("\\external_resource_stability\\stability_transitions") + \
                "\\consumer_resource_modules")
from community_level_properties import max_le

# %%

def rho_sigma(rho_range, sigma_range, n, fixed_parameters):
    
    parameters = generate_parameters(rho_range, sigma_range, n, fixed_parameters)
    
    CRM_across_parameter_space(parameters,
                               "external_resource_stability/simulations/max_spr_reduce_rho_new_solve",
                               ['rho', 'sigma_M'])
                    
# %%

def generate_parameters(rho_range, sigma_range, n, fixed_parameters):
    
    rho_sigma_combos = np.unique(sce.parameter_combinations([rho_range,
                                                             sigma_range],
                                                            n),
                                    axis = 1)
    
    variable_parameters = np.vstack([rho_sigma_combos,
                                     rho_sigma_combos[1, :]/np.sqrt(fixed_parameters['M']),
                                     rho_sigma_combos[1, :]/np.sqrt(fixed_parameters['M'])])
    
    fixed_parameters_mod = deepcopy(fixed_parameters)
    
    fixed_parameters_mod['mu_c'] *= 1/fixed_parameters_mod['M']
    fixed_parameters_mod['mu_g'] *= 1/fixed_parameters_mod['M']

    # array of all parameter combinations
    parameters = sce.variable_fixed_parameters(variable_parameters,
                                               fixed_parameters_mod,
                                               ['rho', 'sigma_M',
                                                'sigma_c', 'sigma_g'])
    
    return parameters

# %%

rhos = np.linspace(1, 0.3, 8) #np.concat([np.linspace(0.79, 0.71, 9), np.linspace(0.69, 0.61, 9)]) #np.linspace(1, 0.3, 8) 
sigmas = np.array([8]) #np.array([4, 5, 6, 7]) 

# %%

rho_sigma(rhos, sigmas, 8,
          dict(mu_c = 3, mu_g = 3, d = 1, b = 10, o = 1,
               M = 100, S = 300))

# %%

### Data inspection ###

max_spr_df = generate_simulation_df("C:/Users/jamil/Documents/PhD/Data/" + \
                                    "external_resource_stability/simulations/max_spr_reduce_rho_new_solve")
    
# %%

feasible_df = max_spr_df.iloc[np.where(max_spr_df["Divergence measure"] == 7000)]
    
# %%

stability_pivot = le_pivot_r(feasible_df.loc[~np.isnan(feasible_df["Max. lyapunov exponent"]), :], 
                             index = 'rho', columns = 'sigma_c')[0]

feasibility_pivot = pd.pivot_table(max_spr_df, index = 'rho', columns = 'sigma_c',
                                   values = 'Divergence measure',
                                   aggfunc = lambda x : 1 - np.count_nonzero(x < np.max(max_spr_df['Divergence measure']))/len(x))

# %%

stability_cond_df = \
    pd.melt(feasible_df.iloc[np.where(feasible_df['sigma_c'] == 3.0)][['rho',
                                                                       'Species packing']],
                            ['rho'])

stability_cond_df.rename(columns = {'rho' : 'rho_x'}, inplace = True)

stability_cond_df.loc[:, 'value'] = 0.5/np.sqrt(stability_cond_df.loc[:, 'value'])

stability_cond_df = pd.concat([stability_cond_df,
                              pd.DataFrame.from_dict({"rho_x" : stability_cond_df['rho_x'].to_list(),
                                                      "variable" : np.repeat("rho",
                                                                             stability_cond_df.shape[0]),
                                                      "value" : stability_cond_df['rho_x'].to_list()},
                                                     orient='index').transpose()])

sns.lineplot(data = stability_cond_df, 
             x = 'rho_x', y = 'value', hue = 'variable',
             errorbar=("ci", 95),
             legend = False)

plt.legend(labels = [r'$\frac{1}{2 \sqrt{\phi_N \gamma^{-1}}}$', '',
                     r'$\rho$', ''])
plt.show()

# %%

communities_08 = load_in_communities("C:/Users/jamil/Documents/PhD/Data/" + \
                                    "external_resource_stability/simulations/max_spr_reduce_rho_new_solve/" + \
                                    "simulations_0.7_8.0.bz2")
    
# %%
    
communities_071_5 = load_in_communities("C:/Users/jamil/Documents/PhD/Data/" + \
                                       "external_resource_stability/simulations/max_spr_reduce_rho/" + \
                                       "simulations_0.71_5.0.bz2") 
    
    # %%
    
for i, community in enumerate(communities_08):
    
    print({'community' : i,
          "max. le" : community.lyapunov_exponent,
          "divergence measure " : community.ODE_sols[0].t[-1]})

    if community.lyapunov_exponent > 0:
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (13, 7))
        
        ax1.plot(community.ODE_sols[0].t,
                 community.ODE_sols[0].y[:community.no_species, :].T)
        
        ax2.plot(community.ODE_sols[1].t,
                 community.ODE_sols[1].y[:community.no_species, :].T)

        plt.show()
        
# %%

nonstable_community = communities_08[0]

# %%

sol = nonstable_community.simulate_community(2000, 1, assign = False)

# %%

le_infinity = [max_le(nonstable_community,
                      np.concatenate((np.random.uniform(1e-8,
                                                        2/nonstable_community.no_species,
                                                        nonstable_community.no_species),
                                      np.random.uniform(1e-8,
                                                        2/nonstable_community.no_resources,
                                                        nonstable_community.no_resources))),
                      #nonstable_community.ODE_sols[0].y[:, -1],
                      T = 3000, perturbation = 1e-6)
               for _ in range(5)]

# %%

fig, axs = plt.subplots(1, 3, figsize = (17, 5))

for sol_init, ax in zip(sol, axs):
    
    ax.plot(sol_init.t,
             sol_init.y[:nonstable_community.no_species, :].T)


#plt.savefig("C:/Users/jamil/Documents/PhD/Figures/externally_supplied_resources/" + \
#            "maybe_nonstable_dynamics.png", dpi = 300)

plt.show()









