# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 08:59:34 2026

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

os.chdir('C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/resource_diversity_stability(sl)/effective_gLV')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/consumer_resource_modules")
from effective_LV_models import gLV
from community_level_properties import max_le
    
sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/resource_diversity_stability(sl)")
from complete_simulation_functions import eLV_gLV_df_from_communities

# %%

def gLV_dynamics(eLV_community : Literal["eLV_SL"],
                 include_rho : Union[list[Literal["rho_D",
                                                  "rho_R",
                                                  "rho_C",
                                                  "rho_1dix"]],
                                     None] =
                 ["rho_D", "rho_R", "rho_C", "rho_1idx"]):
    
    if include_rho is not None:

        rhos = {rho_key : getattr(eLV_community,
                                  rho_key)
                for rho_key in include_rho}

        # correlations between A_ij and r, and A_ij and A_ii
        rhos["rho_r_Aij"] = eLV_community.rho_r_Aij
        rhos["rho_Aii_Aij"] = eLV_community.rho_Aii_Aij

        interaction_args = dict(mu = eLV_community.mu_Aij,
                                sigma = eLV_community.sigma_Aij,
                                rhos = rhos)
        
    else:
        
        interaction_args = dict(mu = eLV_community.mu_Aij,
                                sigma = eLV_community.sigma_Aij)
        
    gLV_community = gLV(eLV_community.no_species)
    gLV_community.model_specific_rates(growth_method='normal',
                                       growth_args=dict(mu = eLV_community.mu_r,
                                                        sigma = eLV_community.sigma_r),
                                       interaction_method='normal',
                                       interaction_args=interaction_args,
                                       self_inhibition_method='normal',
                                       self_inhibition_args=dict(mu = eLV_community.mu_Aii,
                                                                 sigma = eLV_community.sigma_Aii))
    
    gLV_community.calculate_interaction_stats()
    
    # run simulations from randomly generated initial abundances
    gLV_community.simulate_community(t_end = 7000,
                                     no_init_cond = 1)
       
    # numerically estimate the max. lyapunov exponent
    gLV_community.max_lyapunov_exponent = max_le(gLV_community,
                                                 gLV_community.ODE_sols[0].y[:, -1],
                                                 T = 1000,
                                                 perturbation = 1e-6)
    
    for attribute in ["no_resources", "mu_c", "sigma_c", "mu_g", "sigma_g"]:
        
        setattr(gLV_community, attribute, getattr(eLV_community, attribute))

    return gLV_community
    
# %%

def gLV_M(gLV_directory : str,
          eLV_directory : str,
          include_rho : Union[list[Literal["rho_D",
                                           "rho_R",
                                           "rho_C",
                                           "rho_1dix"]],
                              None] =
          ["rho_D", "rho_R", "rho_C", "rho_1idx"]):
    
    '''
    
    ...

    Parameters
    ----------
    resource_pool_sizes : npt.NDArray, optional
        The default is np.arange(50, 275, 25).
    mu_c : float, optional
        The default is 145.
    eLV_directory : str, optional
        File directory to save elVs in. The default is "eLV/M_vs_mu_c".
    all_resource_survive : bool, optional
        Do we assume all resources survive or not. The default is False.

    Returns
    -------
    dict
        resource pool size vs max. lyapunov exponents for all eLVs.

    '''

    def read_call_gLV(full_eLV_directory : str,
                      full_gLV_directory : str,
                      filename : str,
                      include_rho : bool):
        
        # read in eLV communities
        eLV_communities = pd.read_pickle(full_eLV_directory + "/" + filename)
        
        # generate gLV from eLV communities, run simulations
        gLV_communities = [gLV_dynamics(eLV_community,
                                        include_rho) 
                            for eLV_community in 
                            tqdm(eLV_communities,
                                 leave = False,
                                 position = 0,
                                 total = len(eLV_communities))]
        
        cleaned_gLV_communities = [comm
                                   for comm in gLV_communities
                                   if comm is not None]

        # save gLV interaction statistics as a csv (v3 method), rather than
        # pickling the whole community objects
        gLV_df = eLV_gLV_df_from_communities(cleaned_gLV_communities)

        csv_filename = os.path.splitext(filename)[0] + ".csv"

        gLV_df.to_csv(full_gLV_directory + "/" + csv_filename, index = False)
    
    ###################################################################################
    
    full_eLV_directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/eLV/" + \
                           eLV_directory
                           
    full_gLV_directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/gLV/" + \
                          gLV_directory
    
    # make file directory for gLVs
    if not os.path.exists(full_gLV_directory):
        
        os.makedirs(full_gLV_directory)
            
    # generate filenames based on mu_c
    filenames = os.listdir(full_eLV_directory)
        
    # generate gLVs assuming phi_R = 1
    for filename in tqdm(filenames,
                         leave = True,
                         position = 1,
                         total = len(filenames)):
        
        read_call_gLV(full_eLV_directory,
                      full_gLV_directory,
                      filename,
                      include_rho)
        
# %%

gLV_directories = ["M_vs_mu_c",
                   "M_vs_mu_c_drc",
                   "M_vs_mu_c_dr",
                   "M_vs_mu_c_d",
                   "M_vs_mu_c_norho"]

incl_rhos = [["rho_D", "rho_R", "rho_C", "rho_1idx"],
             ["rho_D", "rho_R", "rho_C"],
             ["rho_D", "rho_R"],
             ["rho_D"],
             None]

for gLV_directory, include_rho in zip(gLV_directories, incl_rhos):
    
    gLV_M(eLV_directory="M_vs_mu_c",
          gLV_directory=gLV_directory,
          include_rho=include_rho)
    
# %%

gLV_directories = ["M_vs_mu_c(all_resource)",
                   "M_vs_mu_c_drc(all_resource)",
                   "M_vs_mu_c_dr(all_resource)",
                   "M_vs_mu_c_d(all_resource)",
                   "M_vs_mu_c_norho(all_resource)"]

incl_rhos = [["rho_D", "rho_R", "rho_C", "rho_1idx"],
             ["rho_D", "rho_R", "rho_C"],
             ["rho_D", "rho_R"],
             ["rho_D"],
             None]

for gLV_directory, include_rho in zip(gLV_directories, incl_rhos):
    
    gLV_M(eLV_directory="M_vs_mu_c(all_resource)",
          gLV_directory=gLV_directory,
          include_rho=include_rho)
    