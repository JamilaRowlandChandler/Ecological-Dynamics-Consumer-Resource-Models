# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 00:45:37 2025

@author: jamil
"""

import os
import sys

os.chdir("C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
         "Ecological-Dynamics/Consumer-Resource Models/resource_diversity_stability(sl)")
from simulation_functions import community_dynamics_df, prop_chaotic
    
sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Consumer-Resource Models/consumer_resource_modules')
from models import Consumer_Resource_Model
from community_level_properties import max_le

import numpy as np
import pandas as pd
from tqdm import tqdm

# %%

def communities_M(M, no_communities = 20):
    
    def community_dynamics(M):
        
        community = Consumer_Resource_Model("Self-limiting resource supply",
                                            M, M)
        
        community.growth_consumption_rates('growth function of consumption',
                                           160/M, 1.6/np.sqrt(M), 1, 1.6/np.sqrt(150),
                                           no_negative = True)
        community.model_specific_rates('constant', {'d' : 1},
                                       'constant', {'b' : 1})
        
        community.simulate_community(7000, 1)
        community.calculate_community_properties()
        
        community.lyapunov_exponent = max_le(community,
                                             community.ODE_sols[0].y[:, -1],
                                             T = 1000, perturbation = 1e-6)
        
        return community 
    
    communities = np.array([community_dynamics(M) for _ in range(no_communities)])
    
    return communities

# %%

def communities_M_gamma(M, no_communities = 20):
    
    def community_dynamics(M):
        
        community = Consumer_Resource_Model("Self-limiting resource supply",
                                            M, M)
        
        community.growth_consumption_rates_noneg('growth function of consumption',
                                                 160/M, 1.6/np.sqrt(M), 1,
                                                 1.6/np.sqrt(150))
        community.model_specific_rates('constant', {'d' : 1},
                                       'constant', {'b' : 1})
        
        community.simulate_community(7000, 1)
        community.calculate_community_properties()
        
        community.lyapunov_exponent = max_le(community,
                                             community.ODE_sols[0].y[:, -1],
                                             T = 1000, perturbation = 1e-6)
        
        return community 
    
    communities = np.array([community_dynamics(M) for _ in range(no_communities)])
    
    return communities

# %%

def rough_df(communities):
    
    df = pd.concat([community_dynamics_df(communities_set, ['no_species',
                                                            'no_resources',
                                                            'mu_c',
                                                            'sigma_c',
                                                            'mu_y',
                                                            'sigma_y',
                                                            'd_val',
                                                            'b_val'])
                    for communities_set in communities])

    # set species and reosurce pool size to the correct type - int
    df.rename(columns = {'no_species' : 'S', 'no_resources' : 'M'},
              inplace = True)
    
    df['M'] = np.int32(df['M'])
    df['S'] = np.int32(df['S'])
    
    return df

# %%

resource_pool_sizes = np.arange(50, 275, 25)

communities_absgauss = [communities_M(M) for M in tqdm(resource_pool_sizes,
                                                       leave = True, position = 0)]

# %%

df_absgauss = rough_df(communities_absgauss)

df_absgauss.groupby("M")['Max. lyapunov exponent'].apply(prop_chaotic)

# %%

resource_pool_sizes = np.arange(50, 275, 25)

communities_gamma = [communities_M_gamma(M) for M in tqdm(resource_pool_sizes,
                                                          leave = True, position = 0)]

# %%

df_gamma = rough_df(communities_gamma)

# %%

df_gamma.groupby("M")['Max. lyapunov exponent'].apply(prop_chaotic)
