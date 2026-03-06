# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 16:47:01 2026

@author: jamil
"""

# %%

import os
import pandas as pd
import numpy as np

abspath = os.path.abspath(__file__)
file_directory_name = os.path.dirname(abspath)
os.chdir(file_directory_name)

from simulation_functions import save_models

# %%

def rewrite_communities(old_directory : str,
                        new_directory : str):
    
    for file in os.listdir(old_directory):
    
        communities = pd.read_pickle(os.path.join(old_directory, file))
        
        save_models(communities, new_directory, file)
        
# %%

data_filepath = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations"

rewrite_communities(data_filepath + "/M_vs_mu_c",
                    data_filepath + "/M_vs_mu_c_converted")

# %%

print(np.sum(np.array([os.path.getsize(ele) 
                       for ele in os.scandir(data_filepath + "/M_vs_mu_c")])) - \
      
     np.sum(np.array([os.path.getsize(ele) 
                      for ele in os.scandir(data_filepath + "/M_vs_mu_c_converted")])))
    
# %%

print(np.sum(np.array([os.path.getsize(ele) 
                      for ele in os.scandir(data_filepath + "/M_vs_mu_c_converted")]))/1e9)