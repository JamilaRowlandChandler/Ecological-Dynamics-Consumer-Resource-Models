# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 10:58:57 2025

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
from effective_LV_models import eLV_SL
from community_level_properties import max_le
    
sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/resource_diversity_stability(sl)")
from simulation_functions import pickle_dump

# %%

def eLV(CRM_community : Literal["SL_CRM"],
        sces : any = None):
    
    # initialise eLV (generate growth rates, interaction matrices, etc from the CRM)
    eLV_community = eLV_SL()
    
    if sces is not None: 
        
        cavity_phi_R = sces.loc[np.where((sces["mu_c"] == 
                                          np.round(CRM_community.mu_c * CRM_community.no_resources, 4)) & \
                                         (sces["M"] == CRM_community.no_resources)),
                                "phi_R"].to_numpy()
    
        eLV_community.elv_from_crm(CRM_community,
                                   cavity_phi_R)
    
    else:
        
        eLV_community.elv_from_crm(CRM_community)
        
    eLV_community.generate_elv_parameters()
    eLV_community.calculate_interaction_stats()
    
    # run simulations from randomly generated initial abundances
    eLV_community.simulate_community(t_end = 7000,
                                     no_init_cond = 1)
       
    # numerically estimate the max. lyapunov exponent
    eLV_community.max_lyapunov_exponent = max_le(eLV_community,
                                                 eLV_community.ODE_sols[0].y[:, -1],
                                                 T = 1000,
                                                 perturbation = 1e-6)
    
    return eLV_community
    
# %%

def eLV_M(eLV_directory : str,
          CRM_directory : str,
          all_resource_survive : bool = False):
    
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

    def read_call_eLV(full_CRM_directory : str,
                      full_eLV_directory : str,
                      filename : str,
                      sces : any = None):
        
        '''
        
        Read in consumer-resource models and generate effective-Lotka Volterra 
        models from them. Returns resource pool size vs stability

        Parameters
        ----------
        filename : str
            Consumer-resource model filenames (determined by mu_c).
        cavity_phi_R : float
            resource survival fraction from the cavity calculation.

        Returns
        -------
        dict
            resource pool size vs max. lyapunov exponents for all eLVs.

        '''
        
        # read in consumer-resource model (CRM) communities
        CRM_communities = pd.read_pickle(full_CRM_directory + "/" + filename)
        
        # generate eLV from CRM communities, run simulations
        eLV_communities = [eLV(CRM_community, sces) 
                            for CRM_community in 
                            tqdm(CRM_communities, leave = False, position = 0,
                                 total = len(CRM_communities))]
       
        # save eLV
        pickle_dump(full_eLV_directory + "/" + filename,
                    eLV_communities)
    
    ###################################################################################
    
    full_CRM_directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/" + \
                           CRM_directory
                           
    full_eLV_directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/" + \
                          eLV_directory
    
    # make file directory for eLVs
    if not os.path.exists(full_eLV_directory):
        
        os.makedirs(full_eLV_directory)
            
    # generate filenames based on mu_c
    filenames = os.listdir(full_CRM_directory)
        
    # if the resource survival fraction is used to parametrise the eLV
    if all_resource_survive is False:
    
        # get the resource survival fraction
        sces = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/self_consistency_equations/" + \
                               CRM_directory + ".pkl")
            
        # generate eLVs
        
        for filename in tqdm(filenames,
                             leave = True,
                             position = 1,
                             total = len(filenames)):
            
            read_call_eLV(full_CRM_directory,
                          full_eLV_directory,
                          filename, 
                          sces)
    
    # if we do not use the resource survival fraction (when we don't use any resource dynamics)
    elif all_resource_survive is True:
        
        # generate eLVs assuming phi_R = 1
        for filename in tqdm(filenames,
                             leave = True,
                             position = 1,
                             total = len(filenames)):
            
            read_call_eLV(full_CRM_directory,
                          full_eLV_directory,
                          filename)

# %%

eLV_M(CRM_directory = "M_vs_mu_c",
       eLV_directory = "eLV/M_vs_mu_c")

# %%

eLV_M(CRM_directory = "M_vs_mu_c",
       eLV_directory = "eLV/M_vs_mu_c(all_resource)",
       all_resource_survive = True)

