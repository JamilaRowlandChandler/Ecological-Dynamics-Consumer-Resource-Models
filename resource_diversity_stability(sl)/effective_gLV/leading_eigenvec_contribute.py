# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 10:47:11 2026

@author: jamil
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
from typing import Literal, Union
import numpy.typing as npt
from matplotlib import pyplot as plt
import seaborn as sns

os.chdir('C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/resource_diversity_stability(sl)/effective_gLV')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/consumer_resource_modules")
from models import Consumer_Resource_Model
from community_level_properties import eigenspectrum
    
sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/resource_diversity_stability(sl)")
from complete_simulation_functions import pickle_dump, generate_simulation_df, le_pivot_r

# %%

def calculate_eigenspec(CRM_community):
    
    eigenspec_stats = [eigenspectrum(CRM_community,
                                    ode_sol.y[:, -1])
                       for ode_sol in CRM_community.ODE_sols]
    
    eigenspec_mag = [eig_stat['magnitude_ratios'] 
                     for eig_stat in eigenspec_stats]
    
    return eigenspec_mag
    
# %%

def CRM_eigenspec(CRM_directory : str):


    def read_eig_CRM(full_CRM_directory : str,
                     filename : str):
        
        # read in consumer-resource model (CRM) communities
        CRM_communities = pd.read_pickle(full_CRM_directory + "/" + filename)
        
        # generate eLV from CRM communities, run simulations
        eigenvec_mag = [calculate_eigenspec(CRM_community)
                        for CRM_community in
                        tqdm(CRM_communities,
                             leave = False,
                             position = 0,
                             total = len(CRM_communities))]
       
        return eigenvec_mag
    
    ###################################################################################
                           
    full_CRM_directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/" + \
                          CRM_directory
            
    filenames = os.listdir(full_CRM_directory)
    
    df_simulation = generate_simulation_df(full_CRM_directory)
    
    eigenvec_mags = [read_eig_CRM(full_CRM_directory, 
                                  filename)
                     for filename in tqdm(filenames,
                                          leave = True,
                                          position = 1,
                                          total = len(filenames))]
    
    eigenvec_df = (pd.DataFrame(np.array(eigenvec_mags).flatten().tolist())
                   .rename(columns={'resources' : 'resource_eigvec_mag',
                                    'species' : 'species_eigvec_mag'}))
    
    df = df_simulation.join(eigenvec_df, how='right')
        
    return df

# %%

df_eigenvec_mag = CRM_eigenspec(CRM_directory = "M_vs_mu_c")

# %%

vec_mag_pivot = pd.pivot_table(data=df_eigenvec_mag[df_eigenvec_mag['mu_c'] < 210],
                               columns='M',
                               index='mu_c',
                               values='resource_eigvec_mag',
                               aggfunc='mean')

# %%

fig, (ax1, ax2)  = plt.subplots(1, 2,
                                sharex=True,
                                #sharey=True,
                                layout='constrained',
                                figsize=(7,2.5))

stable_pivot = le_pivot_r(df_eigenvec_mag[df_eigenvec_mag['mu_c'] < 210],
                          columns='M',
                          index='mu_c')[0]

sns.heatmap(stable_pivot,
            cmap="Purples_r",
            vmin=0.0, #0,
            vmax=1.0, # 1,
            cbar_kws={'label' : 'Prob. (stability)'},
            ax=ax1)

sns.heatmap(vec_mag_pivot.mask(stable_pivot < 0.7),
            cmap="Greens_r",
            vmin=0, #0.3, #0,
            vmax=1, #0.6, # 1,
            cbar_kws={'label' : 'resource contribution to\nleading eigenvector, ' + \
                      r'$\frac{||V_R||}{||V_R|| + ||V_N||}$'},
            ax=ax2)
    
fig.supylabel("avg. tot. consumption coeff., " + r'$\mu_c$',
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
    
    ax.set_yticks(np.arange(0.5, len(vec_mag_pivot.index.to_numpy()) + 0.5, 2),
                        labels = vec_mag_pivot.index.to_numpy()[::2], fontsize = 10,
                        rotation = 0)
    ax.set_ylabel("")

    ax.set_xticks(np.arange(0.5, len(vec_mag_pivot.columns.to_numpy()) + 0.5, 2),
                  labels = vec_mag_pivot.columns.to_numpy()[::2],
                  fontsize = 10, rotation = 0)
    ax.set_xlabel("")

ax1.invert_yaxis()
ax2.invert_yaxis()

plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/eigenvec_contr.png",
            bbox_inches='tight')
plt.savefig("C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability/eigenvec_contr.svg",
            bbox_inches='tight')
