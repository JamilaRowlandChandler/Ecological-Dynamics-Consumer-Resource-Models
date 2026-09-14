# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 23:04:18 2026

@author: jamil
"""

import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys

from matplotlib import pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.patheffects as patheffects

os.chdir('C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/resource_diversity_stability(sl)/cavity_solutions_vs_simulations')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/" + \
                    "resource_diversity_stability(sl)")
from complete_simulation_functions import generate_simulation_df, le_pivot_r

# %%

def resource_eigennorm_pivot(df):
    
    return pd.pivot_table(data=df,
                          columns='M',
                          index='timescalar',
                          values='eigenvec_resource_mag',
                          aggfunc='mean')

# %%

def resource_eigenvec_plot(eigennorm_pivot,
                           stable_pivot):
    
    fig, (ax1, ax2)  = plt.subplots(1, 2,
                                    sharex=True,
                                    layout='constrained',
                                    figsize=(7,2.5))

    sns.heatmap(stable_pivot,
                cmap="Purples_r",
                vmin=0.0, #0,
                vmax=1.0, # 1,
                cbar_kws={'label' : 'Prob. (stability)'},
                ax=ax1)

    sns.heatmap(eigennorm_pivot.mask(stable_pivot < 0.7),
                cmap="Greens_r",
                vmin=0, #0.3, #0,
                vmax=1, #0.6, # 1,
                cbar_kws={'label' : 'resource contribution to\nleading eigenvector, ' + \
                          r'$\frac{||V_R||}{||V_R|| + ||V_N||}$'},
                ax=ax2)
        
    fig.supylabel("time-scalar, " + r'$\epsilon$',
                  fontsize=10, weight='bold')

    fig.supxlabel("resource pool size, " + r'$M$',
              fontsize=10, weight='bold')


    for ax in (ax1, ax2):

        ax.axhline(0, 0, 1, color = 'black', linewidth = 2)
        ax.axhline(stable_pivot.shape[0], 0, 1,
                   color = 'black', linewidth = 2)
        ax.axvline(0, 0, 1, color = 'black', linewidth = 2)
        ax.axvline(stable_pivot.shape[1], 0, 1,
                   color = 'black', linewidth = 2)
        ax.set_facecolor('grey')
        
        ax.set_yticks(np.arange(0.5, len(eigennorm_pivot.index.to_numpy()) + 0.5, 2),
                            labels = eigennorm_pivot.index.to_numpy()[::2], fontsize = 10,
                            rotation = 0)
        ax.set_ylabel("")

        ax.set_xticks(np.arange(0.5, len(eigennorm_pivot.columns.to_numpy()) + 0.5, 2),
                      labels = eigennorm_pivot.columns.to_numpy()[::2],
                      fontsize = 10, rotation = 0)
        ax.set_xlabel("")

    ax1.invert_yaxis()
    ax2.invert_yaxis()

    #plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/eigenvec_contr.png",
    #            bbox_inches='tight')
    #plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/eigenvec_contr.svg",
    #            bbox_inches='tight')
    
    

# %%

df_simulation = generate_simulation_df("C:/Users/jamil/Documents/PhD/Data/" \
                                       + 'resource_diversity_stability/simulations/CRM_TS/M_vs_mu_c')

# %%

resource_eigennorm_pivot(df_simulation)