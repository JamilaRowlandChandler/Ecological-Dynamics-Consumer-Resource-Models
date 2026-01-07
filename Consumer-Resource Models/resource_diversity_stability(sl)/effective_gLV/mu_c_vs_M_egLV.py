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
from typing import Union, Literal
import numpy.typing as npt

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Consumer-Resource Models/resource_diversity_stability(sl)/effective_gLV')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
                    "Ecological-Dynamics/Consumer-Resource Models/consumer_resource_modules")
from effective_LV_models import eLV_SL
from community_level_properties import max_le
    
sys.path.insert(0, "C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
                    "Ecological-Dynamics/Consumer-Resource Models/resource_diversity_stability(sl)")
from simulation_functions import pickle_dump

# %%

def call_egLV(CRM_community : Literal["SL_CRM"], cavity_phi_R : float):
    
    # initialise eLV (generate growth rates, interaction matrices, etc from the CRM)
    gLV_community = eLV_SL(CRM_community, cavity_phi_R)
    
    # calculate mean interaction strength, self-inhibition etc
    gLV_community.interaction_statistics()
    
    # run simulations from randomly generated initial abundances
    gLV_community.simulation(t_end = 7000,
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

def egLV_M(resource_pool_sizes : npt.NDArray = np.arange(50, 275, 25),
           mu_c : float = 145,
           gLV_directory : str = "egLV/M_vs_mu_c",
           all_resource_survive : bool = False):
    
    '''
    
    ...

    Parameters
    ----------
    resource_pool_sizes : npt.NDArray, optional
        The default is np.arange(50, 275, 25).
    mu_c : float, optional
        The default is 145.
    gLV_directory : str, optional
        File directory to save elVs in. The default is "egLV/M_vs_mu_c".
    all_resource_survive : bool, optional
        Do we assume all resources survive or not. The default is False.

    Returns
    -------
    dict
        resource pool size vs max. lyapunov exponents for all eLVs.

    '''

    def read_call_egLV(filename : str, cavity_phi_R : float):
        
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
        CRM_communities = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/resource_diversity_stability/simulations/M_vs_mu_c/" + \
                                          "simulations_" + filename + ".pkl")
        
        # generate eLV from CRM communities, run simulations
        egLV_communities = [call_egLV(CRM_community, cavity_phi_R) 
                            for CRM_community in 
                            tqdm(CRM_communities, leave = True, position = 0,
                                 total = len(CRM_communities))]
       
        # save eLV
        pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/resource_diversity_stability/simulations/" + \
                    gLV_directory + "/simulations_" + filename + ".pkl",
                    egLV_communities)
        
        # return dict of resource pool size vs eLV stability (max. lyapunov exponents) 
        return dict(M = np.repeat(CRM_communities[0].no_resources,
                                  len(egLV_communities)),
                    max_le = [gLV_community.max_lyapunov_exponent 
                              for gLV_community in egLV_communities])

    def set_phi_R(sces,
                  mu_c):
        '''
        Determine phi_R (resource survival fraction) from the self-consistency equations
        for a given value of mu_c
        '''
        
        return sces.loc[sces['mu_c'] == mu_c, ['M', 'phi_R']]
    
    ###################################################################################
    
    # make file directory for eLVs
    if not os.path.exists("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/resource_diversity_stability/simulations/" + \
                          gLV_directory):
        
        os.makedirs("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/resource_diversity_stability/simulations/" + \
                    gLV_directory)
     
    # generate filenames based on mu_c
    filenames = [str(M) + "_" + str(np.round(mu_c/M, 4)) 
                 for M in resource_pool_sizes]
    
    # if the resource survival fraction is used to parametrise the eLV
    if all_resource_survive is False:
    
        # get the resource survival fraction
        sces = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/resource_diversity_stability/self_consistency_equations/M_vs_mu_c.pkl")
        M_phi_Rs = set_phi_R(sces, mu_c)
        ordered_phi_Rs = np.array([M_phi_Rs.loc[M_phi_Rs['M'] == M,
                                                'phi_R'].to_numpy()
                                   for M in resource_pool_sizes])
        
        # generate eLVs
        egLV_M_vs_stability = [read_call_egLV(filename, cavity_phi_R) 
                               for filename, cavity_phi_R in 
                               tqdm(zip(filenames, ordered_phi_Rs),
                                    leave = True, position = 1,
                                    total = len(resource_pool_sizes))]
    
    # if we do not use the resource survival fraction (when we don't use any resource dynamics)
    elif all_resource_survive is True:
        
        # generate eLVs assuming phi_R = 1
        egLV_M_vs_stability = [read_call_egLV(filename, 1) 
                               for filename in 
                               tqdm(filenames,
                                    leave = True, position = 1,
                                    total = len(resource_pool_sizes))]
   
    return egLV_M_vs_stability

# %%

egLV_M(gLV_directory = "egLV/M_vs_mu_c_145", mu_c = 145)

egLV_M(gLV_directory = "egLV/M_vs_mu_c_190", mu_c = 190)

#egLV_M(gLV_directory = "egLV/M_vs_mu_c_175", mu_c = 175)

egLV_M(gLV_directory = "egLV/M_vs_mu_c_145(all_resource)", mu_c = 145,
       all_resource_survive = True)

egLV_M(gLV_directory = "egLV/M_vs_mu_c_190(all_resource)", mu_c = 190,
       all_resource_survive = True)

