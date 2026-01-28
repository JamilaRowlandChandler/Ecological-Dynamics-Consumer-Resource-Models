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
from typing import Literal
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

def call_egLV(CRM_community : Literal["SL_CRM"], cavity_phi_R : float):
    
    # initialise eLV (generate growth rates, interaction matrices, etc from the CRM)
    gLV_community = eLV_SL(CRM_community, cavity_phi_R)
    
    # calculate mean interaction strength, self-inhibition etc
    gLV_community.interaction_statistics()
    
    # run simulations from randomly generated initial abundances
    gLV_community.simulation(t_end=7000,
                             initial_abundances=np.random.uniform(1e-8,
                                                                  2/gLV_community.no_species,
                                                                  gLV_community.no_species))
    
    # numerically estimate the max. lyapunov exponent
    gLV_community.max_lyapunov_exponent = max_le(gLV_community,
                                                 gLV_community.ODE_sol.y[:, -1],
                                                 T = 1000,
                                                 perturbation = 1e-6)
    
    return gLV_community
    
# %%

def egLV_sigma(sigma_name : Literal['sigma_c', 'sigma_y'],
               gLV_directory : str,
               CRM_directory : str,
               M : float = 225,
               all_resource_survive : bool = False,
               **kwargs):
    
    '''
    
    ...

    Parameters
    ----------
    sigma_name : Literal['sigma_c', 'sigma_y']
        parameter name
    gLV_directory : str
        File directory to save elVs in.
    M : float, optional
        The default is 225.
    all_resource_survive : bool, optional
        Do we assume all resources survive or not. The default is False.

    Returns
    -------
    dict
        sigmas vs max. lyapunov exponents for all eLVs.

    '''

    def read_call_egLV(filename : str, cavity_phi_R : float):
        
        '''
        
        Read in consumer-resource models and generate effective-Lotka Volterra 
        models from them. Returns sigma vs stability

        Parameters
        ----------
        filename : str
            Consumer-resource model filenames (determined by M).
        cavity_phi_R : float
            resource survival fraction from the cavity calculation.

        Returns
        -------
        dict
            resource pool size vs max. lyapunov exponents for all eLVs.

        '''
        
        # read in consumer-resource model (CRM) communities
        CRM_communities = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/" + \
                                          CRM_directory + "/simulations_" + filename + ".pkl")
        
        # generate eLV from CRM communities, run simulations
        egLV_communities = [call_egLV(CRM_community, cavity_phi_R) 
                            for CRM_community in 
                            tqdm(CRM_communities, leave = False, position = 0,
                                 total = len(CRM_communities))]
       
        # save eLV
        pickle_dump("C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/egLV/" + \
                    gLV_directory + "/simulations_" + filename + ".pkl",
                    egLV_communities)
        
        # return dict of resource pool size vs eLV stability (max. lyapunov exponents) 
        return dict(M = np.repeat(CRM_communities[0].no_resources,
                                  len(egLV_communities)),
                    max_le = [gLV_community.max_lyapunov_exponent 
                              for gLV_community in egLV_communities])

    def set_phi_R(sces,
                  M,
                  sigma_name):
        '''
        Determine phi_R (resource survival fraction) from the self-consistency equations
        for a given value of mu_c
        '''
        
        return sces.loc[sces['M'] == M, [sigma_name, 'phi_R']]
    
    ###################################################################################
    
    # make file directory for eLVs
    if not os.path.exists("C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/egLV/" + \
                          gLV_directory):
        
        os.makedirs("C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/egLV/" + \
                    gLV_directory)
            
    if sce_directory := kwargs.get('sce_directory', None):
            
        sces = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/self_consistency_equations/"
                              + sce_directory + ".pkl")
    
    else:
        
        sces = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/self_consistency_equations/"
                              + CRM_directory + ".pkl")
    
    sigmas = np.unique(sces[sigma_name])
    
    # generate filenames
    match sigma_name:
        
        case 'sigma_c':
    
            filenames = [str(M) + "_" + str(np.round(sigma/np.sqrt(M), 4)) 
                         for sigma in sigmas]
            
        case 'sigma_y':
            
            filenames = [str(M) + "_" + str(np.round(sigma, 4)) 
                         for sigma in sigmas]
    
    # if the resource survival fraction is used to parametrise the eLV
    if all_resource_survive is False:
    
        # get the resource survival fraction
        
        sigma_phi_Rs = set_phi_R(sces, M, sigma_name)
        ordered_phi_Rs = np.array([sigma_phi_Rs.loc[sigma_phi_Rs[sigma_name] == sigma,
                                                    'phi_R'].to_numpy()
                                   for sigma in sigmas])
        
        # generate eLVs
        egLV_sigma_vs_stability = [read_call_egLV(filename, cavity_phi_R) 
                                   for filename, cavity_phi_R in 
                                   tqdm(zip(filenames, ordered_phi_Rs),
                                        leave = True, position = 1,
                                        total = len(sigmas))]
    
    # if we do not use the resource survival fraction (when we don't use any resource dynamics)
    elif all_resource_survive is True:
        
        # generate eLVs assuming phi_R = 1
        egLV_sigma_vs_stability = [read_call_egLV(filename, 1) 
                                   for filename in 
                                   tqdm(filenames,
                                        leave = True, position = 1,
                                        total = len(sigmas))]
   
    return egLV_sigma_vs_stability

# %%

egLV_sigma(sigma_name = "sigma_c",
#           gLV_directory = "M_vs_sigma_c_repeat",
#           CRM_directory = "M_vs_sigma_c_repeat",
           gLV_directory = "M_vs_sigma_c_M_225",
           CRM_directory = "M_vs_sigma_c",
           sce_directory = "M_vs_sigma_c",
           all_resource_survive=True)

egLV_sigma(sigma_name = "sigma_c",
#           gLV_directory = "M_vs_sigma_c_repeat(phi_R)",
#           CRM_directory = "M_vs_sigma_c_repeat",
           gLV_directory = "M_vs_sigma_c_M_225(phi_R)",
           CRM_directory = "M_vs_sigma_c",
           sce_directory = "M_vs_sigma_c",
           all_resource_survive=False)

egLV_sigma(sigma_name = "sigma_y",
#           gLV_directory = "M_vs_sigma_y_repeat(phi_R)",
#           CRM_directory = "M_vs_sigma_y_repeat",
           gLV_directory = "M_vs_sigma_y_M_225",
           CRM_directory = "M_vs_sigma_y",
           sce_directory = "M_vs_sigma_y",
           all_resource_survive=True)

egLV_sigma(sigma_name = "sigma_y",
#           gLV_directory = "M_vs_sigma_y_repeat(phi_R)",
#           CRM_directory = "M_vs_sigma_y_repeat",
           gLV_directory = "M_vs_sigma_y_M_225(phi_R)",
           CRM_directory = "M_vs_sigma_y",
           sce_directory = "M_vs_sigma_y",
           all_resource_survive=True)

