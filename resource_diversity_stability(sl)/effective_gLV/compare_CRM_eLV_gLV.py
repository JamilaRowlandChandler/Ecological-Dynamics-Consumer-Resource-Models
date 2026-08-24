# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 00:00:00 2026

@author: jamil

Load a consumer-resource model (CRM) simulation, build its corresponding
eLV, then build a gLV from that eLV, comparing species_survival_fraction
across all three.

Note: this deliberately does NOT import mean_variance_test.py to reuse its
gLV_dynamics() - that file has an unguarded top-level sweep (unlike
averaged_stats_gLV.py, which is guarded behind
if __name__ == "__main__":) that would run for real on import. gLV_from_eLV()
below is a self-contained copy of the same single-eLV-to-gLV logic instead.

"""

import numpy as np
import pandas as pd
import os
import sys
from typing import Literal, Union
from matplotlib import pyplot as plt

os.chdir('C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/resource_diversity_stability(sl)/effective_gLV')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/consumer_resource_modules")
from effective_LV_models import gLV
from community_level_properties import max_le

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/resource_diversity_stability(sl)")
from complete_simulation_functions import eLV_from_CRM_dynamics

# %%

_CRM_MODEL_NAMES = {"SL_CRM" : "Self-limiting resource supply",
                    "ES_CRM" : "Externally-supplied resources"}

def gLV_from_eLV(eLV_community,
                 include_rho : Union[list[Literal["rho_D",
                                                  "rho_R",
                                                  "rho_C",
                                                  "rho_1idx"]],
                                     None] =
                 ["rho_D", "rho_R", "rho_C", "rho_1idx"],
                 empirical : bool = True,
                 t_end : float = 7000,
                 no_init_cond : int = 1):

    '''

    Build and simulate a gLV community matching a single eLV community's own
    statistics (c.f. gLV_dynamics in mean_variance_test.py, with
    calculate_community_properties() added so species_survival_fraction is
    available for comparison).

    Parameters
    ----------
    eLV_community : eLV_SL or eLV_ES
        Must already have calculate_interaction_stats() called on it (true
        of any community returned by eLV_from_CRM_dynamics()).
    include_rho : list of str, or None, optional
        Which interaction-matrix correlations to use to generate correlated
        gLV interactions. The default is ["rho_D", "rho_R", "rho_C", "rho_1idx"].
    empirical : bool, optional
        Only used when include_rho is not None - see
        gLV.__correlated_interactions()/__correlated_rates() in
        effective_LV_models.py for what this controls. The default is True.
    t_end : float, optional
        Simulation end time. The default is 7000.
    no_init_cond : int, optional
        Number of initial abundances dynamics are simulated from. The
        default is 1.

    Returns
    -------
    gLV_community : gLV
        Simulated gLV community.

    '''

    if include_rho is not None:

        rhos = {rho_key : getattr(eLV_community, rho_key)
                for rho_key in include_rho}

        # correlations between A_ij and r, and A_ij and A_ii
        rhos["rho_r_Aij"] = eLV_community.rho_r_Aij
        rhos["rho_Aii_Aij"] = eLV_community.rho_Aii_Aij
        print(rhos["rho_r_Aij"])

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
                                                                 sigma = eLV_community.sigma_Aii),
                                       empirical=empirical)

    gLV_community.calculate_interaction_stats()

    # run simulations from randomly generated initial abundances
    gLV_community.simulate_community(t_end = t_end,
                                     no_init_cond = no_init_cond)

    # community properties (species_survival_fraction etc.), and the
    # max. lyapunov exponent
    gLV_community.calculate_community_properties()
    gLV_community.max_lyapunov_exponent = max_le(gLV_community,
                                                 gLV_community.ODE_sols[0].y[:, -1],
                                                 T = 1000,
                                                 perturbation = 1e-6)

    for attribute in ["no_resources", "mu_c", "sigma_c", "mu_g", "sigma_g"]:

        setattr(gLV_community, attribute, getattr(eLV_community, attribute))

    return gLV_community

# %%

def compare_CRM_eLV_gLV(CRM_filepath : str,
                        CRM_index : int = 0,
                        cavity_phi_R : bool = False,
                        include_rho : Union[list[Literal["rho_D",
                                                         "rho_R",
                                                         "rho_C",
                                                         "rho_1idx"]],
                                            None] =
                        ["rho_D", "rho_R", "rho_C", "rho_1idx"],
                        empirical : bool = True):

    '''

    Load a consumer-resource model (CRM) simulation, build its corresponding
    eLV community (via elv_from_crm()), then build a gLV community matching
    that eLV's statistics. Compares species_survival_fraction across all
    three and returns the model objects.

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
    include_rho : list of str, or None, optional
        Which interaction-matrix correlations to use to generate correlated
        gLV interactions. The default is ["rho_D", "rho_R", "rho_C", "rho_1idx"].
    empirical : bool, optional
        Only used when include_rho is not None - see
        gLV.__correlated_interactions()/__correlated_rates() in
        effective_LV_models.py for what this controls. The default is True.

    Returns
    -------
    CRM_community : SL_CRM or ES_CRM
        The loaded consumer-resource model community.
    eLV_community : eLV_SL or eLV_ES
        The eLV community built from CRM_community.
    gLV_community : gLV
        The gLV community built from eLV_community's statistics.

    '''

    # --- load the CRM simulation ---
    
    complete_CRM_filepath = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/" + \
                        CRM_filepath

    CRM_communities = pd.read_pickle(complete_CRM_filepath)
    CRM_community = CRM_communities[CRM_index]

    if not hasattr(CRM_community, "species_survival_fraction"):

        CRM_community.calculate_community_properties()

    model = _CRM_MODEL_NAMES[type(CRM_community).__name__]

    # --- build the eLV from the CRM ---
    
    if cavity_phi_R is True:
        
        sces = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/self_consistency_equations/" + \
                              CRM_filepath.split("/")[0] + ".pkl")
        
        phiR = sces.loc[np.where((sces["mu_c"] == 
                                  np.round(CRM_community.mu_c * CRM_community.no_resources, 4)) & \
                                 (sces["M"] == CRM_community.no_resources)),
                        "phi_R"].to_numpy()
            
        eLV_community = eLV_from_CRM_dynamics(model,
                                              [CRM_community],
                                              phiR)[0]
    
    else:

        eLV_community = eLV_from_CRM_dynamics(model,
                                              [CRM_community])[0]

    # --- build the gLV from the eLV ---

    gLV_community = gLV_from_eLV(eLV_community,
                                 include_rho,
                                 empirical)

    # --- compare species_survival_fraction ---

    print("--- species_survival_fraction ---")
    print(f"  CRM: {CRM_community.species_survival_fraction[0]:.4f}")
    print(f"  eLV: {eLV_community.species_survival_fraction[0]:.4f}")
    print(f"  gLV: {gLV_community.species_survival_fraction[0]:.4f}")

    return CRM_community, eLV_community, gLV_community

# %%

# example usage (guarded so importing this module never runs it):

if __name__ == "__main__":

     CRM_community, eLV_community, gLV_community = compare_CRM_eLV_gLV(
         "M_vs_mu_c/simulations_250_1.0.pkl",
         cavity_phi_R=True,
         empirical=True)
     
# %%

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(eLV_community.ODE_sols[0].t,
         eLV_community.ODE_sols[0].y.T)

ax2.plot(gLV_community.ODE_sols[0].t,
         gLV_community.ODE_sols[0].y.T)