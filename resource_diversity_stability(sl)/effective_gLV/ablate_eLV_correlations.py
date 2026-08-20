# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 00:00:00 2026

@author: jamil

Load a consumer-resource model (CRM) simulation, build its corresponding
eLV community, then use decompose_eLV()/reconstruct_ablated() (eLVMethods,
in effective_LV_models.py) to individually remove each of the five
correlations an eLV's interaction matrix carries - rho_R, rho_C, rho_D,
rho_r_Aij, rho_Aii_Aij - producing one ablated eLV community per
correlation, with every other correlation left intact.

"""

import numpy as np
import pandas as pd
import os
import sys
from copy import deepcopy
from typing import Union

os.chdir('C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/resource_diversity_stability(sl)/effective_gLV')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/consumer_resource_modules")
from community_level_properties import max_le

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/resource_diversity_stability(sl)")
from complete_simulation_functions import eLV_from_CRM_dynamics

# %%

_CRM_MODEL_NAMES = {"SL_CRM" : "Self-limiting resource supply",
                    "ES_CRM" : "Externally-supplied resources"}

# one entry per correlation, mapping its label to the single
# reconstruct_ablated() kwarg that removes it (and only it)
_ABLATIONS = {"rho_R"        : dict(keep_row=False),
             "rho_C"        : dict(keep_col=False),
             "rho_D"        : dict(keep_residual=False),
             "rho_r_Aij"    : dict(keep_r_corr=False),
             "rho_Aii_Aij"  : dict(keep_Aii_corr=False)}

# %%

def ablate_eLV_correlations(eLV_community,
                            t_end : float = 7000,
                            no_init_cond : int = 1,
                            seed : Union[int, None] = None):

    '''

    Decompose eLV_community's interaction matrix (decompose_eLV()), then
    build one ablated community per correlation (reconstruct_ablated()),
    each with exactly that correlation removed and every other correlation
    left intact. Every ablated community is re-simulated and re-analysed
    from scratch.

    Parameters
    ----------
    eLV_community : eLV_SL or eLV_ES
        Must already have interaction_matrix/r set (i.e.
        generate_elv_parameters() has been called).
    t_end : float, optional
        Simulation end time for each ablated community. The default is 7000.
    no_init_cond : int, optional
        Number of initial abundances dynamics are simulated from. The
        default is 1.
    seed : int, optional
        Seed for the row/diagonal shuffles reconstruct_ablated() uses when
        removing rho_r_Aij/rho_Aii_Aij, for reproducibility. The default is
        None.

    Returns
    -------
    ablated : dict
        {correlation label : ablated eLV community}, one entry for each of
        rho_R, rho_C, rho_D, rho_r_Aij, rho_Aii_Aij.

    '''

    if seed is not None:

        np.random.seed(seed)

    grand_mean, lam, kap, W = eLV_community.decompose_eLV()

    ablated = {}

    for label, kwargs in _ABLATIONS.items():

        community = deepcopy(eLV_community)

        A_new, r_new = eLV_community.reconstruct_ablated(grand_mean, lam, kap, W,
                                                          **kwargs)

        community.interaction_matrix = A_new
        community.r = r_new

        community.calculate_interaction_stats()

        # run simulations from randomly generated initial abundances
        community.simulate_community(t_end = t_end,
                                     no_init_cond = no_init_cond)

        community.calculate_community_properties()
        community.max_lyapunov_exponent = max_le(community,
                                                 community.ODE_sols[0].y[:, -1],
                                                 T = 1000,
                                                 perturbation = 1e-6)

        ablated[label] = community

    return ablated

# %%

def decompose_and_ablate_CRM(CRM_filepath : str,
                             CRM_index : int = 0,
                             cavity_phi_R : Union[float, None] = None,
                             t_end : float = 7000,
                             no_init_cond : int = 1,
                             seed : Union[int, None] = None):

    '''

    Load a consumer-resource model (CRM) simulation, build its corresponding
    eLV community (via elv_from_crm()), then generate one ablated eLV
    community per correlation (see ablate_eLV_correlations()).

    Parameters
    ----------
    CRM_filepath : str
        Full path to a pickled file of CRM community objects (SL_CRM or
        ES_CRM).
    CRM_index : int, optional
        Which community in the file to use. The default is 0.
    cavity_phi_R : float, optional
        Cavity-predicted resource survival fraction, used by eLV_SL to
        restrict interactions to surviving resources only (Self-limiting
        resource supply only). The default is None (all resources assumed
        to survive).
    t_end : float, optional
        Simulation end time for each ablated community. The default is 7000.
    no_init_cond : int, optional
        Number of initial abundances dynamics are simulated from. The
        default is 1.
    seed : int, optional
        Seed for the row/diagonal shuffles reconstruct_ablated() uses when
        removing rho_r_Aij/rho_Aii_Aij, for reproducibility. The default is
        None.

    Returns
    -------
    eLV_community : eLV_SL or eLV_ES
        The original (un-ablated) eLV community built from the CRM.
    ablated : dict
        {correlation label : ablated eLV community} - see
        ablate_eLV_correlations().

    '''

    # --- load the CRM simulation, build the eLV ---

    CRM_communities = pd.read_pickle(CRM_filepath)
    CRM_community = CRM_communities[CRM_index]

    model = _CRM_MODEL_NAMES[type(CRM_community).__name__]

    eLV_community = eLV_from_CRM_dynamics(model,
                                          [CRM_community],
                                          cavity_phi_R)[0]

    # --- ablate each correlation in turn ---

    ablated = ablate_eLV_correlations(eLV_community, t_end, no_init_cond, seed)

    return eLV_community, ablated

# %%

# example usage (guarded so importing this module never runs it):

# if __name__ == "__main__":
#
#     eLV_community, ablated = decompose_and_ablate_CRM(
#         "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/M_vs_mu_c/simulations_100_1.0.pkl",
#         seed=0)
#
#     print(f"original: species_survival_fraction={eLV_community.species_survival_fraction[0]:.4f}  "
#           f"max_lyapunov_exponent={eLV_community.max_lyapunov_exponent:.4f}")
#
#     for label, community in ablated.items():
#         print(f"no {label}: species_survival_fraction={community.species_survival_fraction[0]:.4f}  "
#               f"max_lyapunov_exponent={community.max_lyapunov_exponent:.4f}")
