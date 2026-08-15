# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 17:44:58 2026

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

def recalculate_interact_stats(eLV_community : Literal["eLV_SL"]):
    
    eLV_community.r = eLV_community.species_growth
    delattr(eLV_community, "species_growth")
    
    eLV_community.calculate_interaction_stats()
    
    return eLV_community
    
# %%

def recalculate_eLVs(eLV_directory : str):


    def read_upd_eLV(full_eLV_directory : str,
                      filename : str):
        
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
        eLV_communities = pd.read_pickle(full_eLV_directory + "/" + filename)
        
        # generate eLV from CRM communities, run simulations
        eLV_communities_upd = [recalculate_interact_stats(eLV_community)
                               for eLV_community in
                               tqdm(eLV_communities,
                                    leave = False,
                                    position = 0,
                                    total = len(eLV_communities))]
       
        # save eLV
        pickle_dump(full_eLV_directory + "/" + filename,
                    eLV_communities_upd)
    
    ###################################################################################
                           
    full_eLV_directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/" + \
                          eLV_directory
            
    filenames = os.listdir(full_eLV_directory)
        
    for filename in tqdm(filenames,
                         leave = True,
                         position = 1,
                         total = len(filenames)):
        
        read_upd_eLV(full_eLV_directory,
                     filename)

# %%

recalculate_eLVs(eLV_directory = "eLV/M_vs_mu_c")

# %%

recalculate_eLVs(eLV_directory = "eLV/M_vs_mu_c(all_resource)")

