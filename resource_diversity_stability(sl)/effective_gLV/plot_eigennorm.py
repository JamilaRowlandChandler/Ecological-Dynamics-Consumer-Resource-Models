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

os.chdir('C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/resource_diversity_stability(sl)/effective_gLV')

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
                           stable_pivot,
                           logged):
    
    fig, (ax1, ax2)  = plt.subplots(1, 2,
                                    sharex=True,
                                    layout='constrained',
                                    figsize=(7,2.5))

    sns.heatmap(stable_pivot,
                cmap="Purples_r",
                vmin=0.0,
                vmax=1.0,
                cbar_kws={'label' : 'Prob. (stability)'},
                ax=ax1,
                square=True)
    
    if logged is True: 
        
        plot_eig = np.log10(np.abs(eigennorm_pivot))
        
    else:
        
        plot_eig = eigennorm_pivot

    sns.heatmap(plot_eig,
                cmap="Greens_r",
                #vmin=0, #0.3, #0,
                #vmax= np.max(eigennorm_pivot),# 1, #0.6, # 1,
                cbar_kws={'label' : 'resource contribution to\nleading eigenvector, ' + \
                          r'$\frac{||V_R||}{||V_R|| + ||V_N||}$'},
                ax=ax2,
                square=True)
        
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
        
        if logged is True: 
        
            ax.set_yticks(np.arange(0.5,
                                    len(eigennorm_pivot.index.to_numpy()) + 0.5,
                                    1),
                                labels = [f"$10^{{{e}}}$" 
                                          for e in 
                                          np.int64(np.round(np.log10(1/eigennorm_pivot.index.to_numpy()),
                                                            1))],
                                fontsize = 10,
                                rotation = 0)
            
        else:
            
            ax.set_yticks(np.arange(0.5,
                                    len(eigennorm_pivot.index.to_numpy()) + 0.5,
                                    1),
                                labels = [f"$1/{{{e}}}$" 
                                          for e in 
                                          np.round(eigennorm_pivot.index.to_numpy(),
                                                   1)],
                                fontsize = 10,
                                rotation = 0)
                 
        ax.set_ylabel("")

        ax.set_xticks(np.arange(0.5, len(eigennorm_pivot.columns.to_numpy()) + 0.5, 2),
                      labels = eigennorm_pivot.columns.to_numpy()[::2],
                      fontsize = 10, rotation = 0)
        ax.set_xlabel("")

    ax1.invert_yaxis()
    ax2.invert_yaxis()
    
    return fig, (ax1, ax2)
    
# %%

def load_data_plot(data_directory : str,
                   fig_filename : str,
                   logged : bool = True) -> None:
    
    base_data_directory = "C:/Users/jamil/Documents/PhD/Data/" \
                           + "resource_diversity_stability/simulations/"
                           
    full_data_directory = base_data_directory + data_directory
                           
    base_fig_directory = "C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/"
    
    full_fig_directory = base_fig_directory + fig_filename
    
    df = pd.concat([pd.read_csv(full_data_directory + "/" + filename)
                               for filename in os.listdir(full_data_directory)])
    
    eigennorm_pivot = resource_eigennorm_pivot(df)
    stable_pivot = le_pivot_r(df,
                              columns='M',
                              index='timescalar')[0]
    
    fig, axs = resource_eigenvec_plot(eigennorm_pivot,
                                      stable_pivot,
                                      logged)
    
    plt.savefig(full_fig_directory + ".png",
                bbox_inches='tight')
    plt.savefig(full_fig_directory + ".svg",
                bbox_inches='tight')
    
# %%

load_data_plot("CRM_TS/M_vs_mu_c",
               "eigenvec_contr")
    
load_data_plot("CRM_TS/small_separation",
               "eigenvec_contr_smaller",
               logged=False)
