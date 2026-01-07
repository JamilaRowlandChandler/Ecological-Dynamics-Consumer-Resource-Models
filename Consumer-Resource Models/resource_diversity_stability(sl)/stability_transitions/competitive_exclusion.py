# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 16:21:22 2025

@author: jamil
"""

import os

os.chdir("C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
         "Ecological-Dynamics/Consumer-Resource Models/consumer_resource_modules")

from models import Consumer_Resource_Model

from matplotlib import pyplot as plt
import seaborn as sns 
import numpy as np
import pandas as pd

# %%

def communities_M(M, no_communities = 20):
    
    def community_S_star(M):
        
        community = Consumer_Resource_Model("Self-limiting resource supply",
                                            M, M)
        
        community.growth_consumption_rates('growth function of consumption',
                                           160/M, 1.6/np.sqrt(M), 1, 0)
        community.model_specific_rates('constant', {'d' : 1},
                                       'constant', {'b' : 1})
        
        community.simulate_community(7000, 1)
        community.calculate_community_properties()
        
        return community.species_survival_fraction[0] * community.no_species  
    
    S_stars = np.array([community_S_star(M) for _ in range(no_communities)])
    
    return {'M' : np.repeat(M, no_communities), 'S*' : S_stars}

# %%

resource_pool_sizes = np.arange(50, 275, 25)

M_vs_S_star = [communities_M(M) for M in resource_pool_sizes]

# %%

M_S_star_df = pd.concat([pd.DataFrame(M_vs_S) for M_vs_S in M_vs_S_star])

# %%

plot_df = M_S_star_df.groupby('M').apply("mean")

fig, ax = plt.subplots(1, 1, figsize = (2, 2))

sns.lineplot(data = plot_df, x = 'M', y = 'S*',
             ax = ax, linewidth = 1.5, color = 'black', err_style = "bars",
             errorbar = ("pi", 20))

ax.set_xticks(resource_pool_sizes[::2],
              labels = resource_pool_sizes[::2], fontsize = 10, rotation = 0)

ax.yaxis.set_tick_params(labelsize = 10)

ax.set_ylabel('')
ax.set_xlabel('')

sns.despine(ax = ax)

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/competitive_exclusion.png",
            bbox_inches='tight', dpi = 400)

plt.show()


