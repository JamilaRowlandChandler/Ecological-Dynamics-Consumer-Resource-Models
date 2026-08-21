# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 00:00:00 2026

@author: jamil

Backfill calculate_community_properties() (species_survival_fraction,
species_avg_abundance, species_abundance_fluctuations) onto already-pickled
eLV community objects - the base eLV_M() pipeline in all_mu_c_vs_M_egLV.py
never calls it, so these attributes are missing on existing eLV data (see
read_pickled_eLV_directory()/survival_fraction_pivot() in
plot_eLV_ablations.py, which needed this to compare survival fraction
against the ablations).

Cheap: only recomputes summary statistics from each community's existing
ODE_sols - no re-simulation is done.

"""

from __future__ import annotations

import os
import sys
from tqdm import tqdm
from typing import Literal

os.chdir('C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/resource_diversity_stability(sl)/effective_gLV')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/consumer_resource_modules")

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/resource_diversity_stability(sl)")
from simulation_functions import pickle_dump

# %%

def backfill_community_properties(eLV_community : Literal["eLV_SL"]):

    eLV_community.calculate_community_properties()

    return eLV_community

# %%

def backfill_eLV_directory(eLV_directory : str):

    '''

    For every file in eLV_directory (a pickled list of eLV community
    objects), call calculate_community_properties() on every community and
    re-save the file in place.

    Parameters
    ----------
    eLV_directory : str
        Directory (relative to
        C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/)
        of pickled eLV community lists, e.g. "eLV/M_vs_mu_c".

    Returns
    -------
    None.

    '''

    def read_upd_eLV(full_eLV_directory : str,
                     filename : str):

        import pandas as pd

        eLV_communities = pd.read_pickle(full_eLV_directory + "/" + filename)

        eLV_communities_upd = [backfill_community_properties(eLV_community)
                               for eLV_community in
                               tqdm(eLV_communities,
                                    leave = False,
                                    position = 0,
                                    total = len(eLV_communities))]

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

        read_upd_eLV(full_eLV_directory, filename)

# %%

# example usage (guarded so importing this module never runs it):

# if __name__ == "__main__":
#
#     backfill_eLV_directory("eLV/M_vs_mu_c")
