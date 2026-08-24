# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 00:00:00 2026

@author: jamil

Generate gLV communities from the *pooled* statistics of a whole batch of
eLV communities, rather than one gLV per eLV (c.f. mean_variance_test.py).

For each file of eLV communities:
    1. load all eLV communities in the file
    2. average their growth-rate/interaction statistics (mu_r, sigma_r,
       mu_Aij, sigma_Aij, mu_Aii, sigma_Aii, and any requested rhos) across
       every community in the file
    3. draw n new gLV communities from that single averaged distribution

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

os.chdir('C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/resource_diversity_stability(sl)/effective_gLV')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/consumer_resource_modules")
from effective_LV_models import gLV
from community_level_properties import max_le

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/resource_diversity_stability(sl)")
from complete_simulation_functions import eLV_gLV_df_from_communities

# %%

# statistics carried over from the eLV community onto every gLV community
# generated from it (c.f. gLV_dynamics in mean_variance_test.py)
CARRIED_ATTRIBUTES = ["no_resources", "mu_c", "sigma_c", "mu_g", "sigma_g"]

def average_eLV_statistics(eLV_communities : list,
                           include_rho : Union[list[Literal["rho_D",
                                                            "rho_R",
                                                            "rho_C",
                                                            "rho_1idx"]],
                                               None] =
                           ["rho_D", "rho_R", "rho_C", "rho_1idx"]):

    '''

    Average the growth rate and interaction statistics of a list of eLV
    communities (assumed to already have calculate_interaction_stats()
    called on them - true for any pickled eLV community, see all_mu_c_vs_M_egLV.py).

    Parameters
    ----------
    eLV_communities : list
        List of eLV community objects (eLV_SL/eLV_ES) loaded from a single file.
    include_rho : list of str, or None, optional
        Which interaction-matrix correlations to average and later use to
        generate correlated gLV interactions. The default is
        ["rho_D", "rho_R", "rho_C", "rho_1idx"].

    Returns
    -------
    dict
        Averaged statistics, keyed by attribute name.

    '''

    stat_keys = ["mu_r", "sigma_r", "mu_Aij", "sigma_Aij", "mu_Aii", "sigma_Aii"]

    if include_rho is not None:

        stat_keys += list(include_rho)

        # correlations between A_ij and r, and A_ij and A_ii
        stat_keys += ["rho_r_Aij", "rho_Aii_Aij"]

    averaged_stats = {key : np.mean([getattr(eLV_community, key)
                                     for eLV_community in eLV_communities])
                      for key in stat_keys}

    # carry over (averaged) resource/consumption statistics for bookkeeping
    for attribute in CARRIED_ATTRIBUTES:

        averaged_stats[attribute] = np.mean([getattr(eLV_community, attribute)
                                             for eLV_community in eLV_communities])

    return averaged_stats

# %%

def gLV_from_averaged_stats(averaged_stats : dict,
                            no_species : int,
                            include_rho : Union[list[Literal["rho_D",
                                                             "rho_R",
                                                             "rho_C",
                                                             "rho_1idx"]],
                                                None] =
                            ["rho_D", "rho_R", "rho_C", "rho_1idx"],
                            empirical : bool = True):

    '''

    Generate and simulate a single gLV community from a dict of averaged
    eLV statistics (c.f. gLV_dynamics in mean_variance_test.py).

    Parameters
    ----------
    averaged_stats : dict
        Output of average_eLV_statistics().
    no_species : int
        Number of species in the gLV community.
    include_rho : list of str, or None, optional
        Which interaction-matrix correlations to use to generate correlated
        gLV interactions. The default is ["rho_D", "rho_R", "rho_C", "rho_1idx"].
    empirical : bool, optional
        Only used when rho_r_Aij/rho_Aii_Aij are requested (i.e. include_rho
        is not None) - see gLV.__correlated_rates() in effective_LV_models.py
        for what this controls. The default is True.

    Returns
    -------
    gLV_community : gLV
        Simulated gLV community.

    '''

    if include_rho is not None:

        rhos = {rho_key : averaged_stats[rho_key] for rho_key in include_rho}

        # correlations between A_ij and r, and A_ij and A_ii
        rhos["rho_r_Aij"] = averaged_stats["rho_r_Aij"]
        rhos["rho_Aii_Aij"] = averaged_stats["rho_Aii_Aij"]

        interaction_args = dict(mu = averaged_stats["mu_Aij"],
                                sigma = averaged_stats["sigma_Aij"],
                                rhos = rhos)

    else:

        interaction_args = dict(mu = averaged_stats["mu_Aij"],
                                sigma = averaged_stats["sigma_Aij"])
        
    gLV_community = gLV(no_species)
    gLV_community.model_specific_rates(growth_method='normal',
                                       growth_args=dict(mu = averaged_stats["mu_r"],
                                                        sigma = averaged_stats["sigma_r"]),
                                       interaction_method='normal',
                                       interaction_args=interaction_args,
                                       self_inhibition_method='normal',
                                       self_inhibition_args=dict(mu = averaged_stats["mu_Aii"],
                                                                 sigma = averaged_stats["sigma_Aii"]),
                                       empirical=empirical)

    gLV_community.calculate_interaction_stats()
    
    #gLV_community.interaction_matrix = gLV_community.interaction_matrix/gLV_community.r
    #gLV_community.r = gLV_community.r/gLV_community.r

    # run simulations from randomly generated initial abundances
    gLV_community.simulate_community(t_end = 7000,
                                     no_init_cond = 1)
    
    gLV_community.calculate_community_properties()

    # numerically estimate the max. lyapunov exponent
    gLV_community.max_lyapunov_exponent = max_le(gLV_community,
                                                 gLV_community.ODE_sols[0].y[:, -1],
                                                 T = 1000,
                                                 perturbation = 1e-6)

    for attribute in CARRIED_ATTRIBUTES:

        setattr(gLV_community, attribute, averaged_stats[attribute])

    return gLV_community

# %%

def gLV_communities_from_averaged_eLV(eLV_communities : list,
                                      n : int,
                                      include_rho : Union[list[Literal["rho_D",
                                                                       "rho_R",
                                                                       "rho_C",
                                                                       "rho_1idx"]],
                                                          None] =
                                      ["rho_D", "rho_R", "rho_C", "rho_1idx"],
                                      empirical : bool = True):

    '''

    Pool the statistics of a list of eLV communities, then draw n gLV
    communities from that single averaged distribution.

    Parameters
    ----------
    eLV_communities : list
        List of eLV community objects loaded from a single file.
    n : int
        Number of gLV communities to generate from the averaged statistics.
    include_rho : list of str, or None, optional
        The default is ["rho_D", "rho_R", "rho_C", "rho_1idx"].
    empirical : bool, optional
        Only used when include_rho is not None - see gLV.__correlated_interactions()/
        __correlated_rates() in effective_LV_models.py for what this
        controls. The default is True.

    Returns
    -------
    list
        n simulated gLV communities.

    '''

    averaged_stats = average_eLV_statistics(eLV_communities, include_rho)

    # assumes every eLV community in the file has the same species pool size
    no_species = eLV_communities[0].no_species

    return [gLV_from_averaged_stats(averaged_stats, no_species, include_rho,
                                    empirical)
            for _ in tqdm(range(n), leave = False, position = 0, total = n)]

# %%

def gLV_M_averaged(gLV_directory : str,
                   eLV_directory : str,
                   n : int,
                   include_rho : Union[list[Literal["rho_D",
                                                    "rho_R",
                                                    "rho_C",
                                                    "rho_1idx"]],
                                       None] =
                   ["rho_D", "rho_R", "rho_C", "rho_1idx"],
                   empirical : bool = True):

    '''

    For every file of eLV communities in eLV_directory, pool the statistics
    of all eLV communities in that file, then generate n gLV communities
    from that single averaged distribution and save them to gLV_directory.

    Parameters
    ----------
    gLV_directory : str
        File directory to save gLVs in.
    eLV_directory : str
        File directory to load eLVs from.
    n : int
        Number of gLV communities to generate per file (per averaged eLV
        distribution).
    include_rho : list of str, or None, optional
        The default is ["rho_D", "rho_R", "rho_C", "rho_1idx"].
    empirical : bool, optional
        Only used when include_rho is not None - see gLV.__correlated_interactions()/
        __correlated_rates() in effective_LV_models.py for what this
        controls. The default is True.

    Returns
    -------
    None.

    '''

    def read_call_gLV(full_eLV_directory : str,
                      full_gLV_directory : str,
                      filename : str,
                      n : int,
                      include_rho : bool):

        # read in eLV communities
        eLV_communities = pd.read_pickle(full_eLV_directory + "/" + filename)

        # pool eLV statistics, generate n gLVs from the averaged distribution
        gLV_communities = gLV_communities_from_averaged_eLV(eLV_communities,
                                                             n,
                                                             include_rho,
                                                             empirical)

        # save gLV interaction statistics as a csv (v3 method), rather than
        # pickling the whole community objects
        gLV_df = eLV_gLV_df_from_communities(gLV_communities)

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

    # generate gLVs from the averaged eLV statistics for each file
    for filename in tqdm(filenames,
                         leave = True,
                         position = 1,
                         total = len(filenames)):

        read_call_gLV(full_eLV_directory,
                      full_gLV_directory,
                      filename,
                      n,
                      include_rho)

# %%

# example usage, mirroring the directory/rho sweeps in mean_variance_test.py.
# Guarded so importing this module (e.g. to reuse the functions above) never
# re-triggers the real generation sweep - only running this file directly does.

if __name__ == "__main__":

    gLV_directories = [#"M_vs_mu_c(averaged)",
                       #"M_vs_mu_c_drc(averaged)",
                       #"M_vs_mu_c_dr(averaged)",
                       #"M_vs_mu_c_d(averaged)",
                       "M_vs_mu_c_norho(averaged)"]

    incl_rhos = [#["rho_D", "rho_R", "rho_C", "rho_1idx"],
                 #["rho_D", "rho_R", "rho_C"],
                 #["rho_D", "rho_R"],
                 #["rho_D"],
                 None]

    for gLV_directory, include_rho in zip(gLV_directories, incl_rhos):

         gLV_M_averaged(eLV_directory="M_vs_mu_c",
                        gLV_directory=gLV_directory,
                        n=40,
                        include_rho=include_rho)

    gLV_directories_ar = ["M_vs_mu_c(averaged, all_resource)",
                          "M_vs_mu_c_drc(averaged, all_resource)",
                          "M_vs_mu_c_dr(averaged, all_resource)",
                          "M_vs_mu_c_d(averaged, all_resource)",
                          "M_vs_mu_c_norho(averaged, all_resource)"]

    for gLV_directory, include_rho in zip(gLV_directories_ar, incl_rhos):

         gLV_M_averaged(eLV_directory="M_vs_mu_c(all_resource)",
                        gLV_directory=gLV_directory,
                        n=40,
                        include_rho=include_rho)
         