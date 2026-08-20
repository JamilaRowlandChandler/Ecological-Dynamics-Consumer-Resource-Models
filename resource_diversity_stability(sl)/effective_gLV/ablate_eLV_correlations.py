# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 00:00:00 2026

@author: jamil

Load a consumer-resource model (CRM) simulation, build its corresponding
eLV community, then use decompose_eLV()/reconstruct_ablated() and
ablate_moments() (eLVMethods, in effective_LV_models.py) to individually
remove each of the ten correlations/moments an eLV carries - rho_R, rho_C,
rho_D, rho_r_Aij, rho_Aii_Aij, mu_r, mu_Aij, sigma_r, sigma_Aij, sigma_Aii -
producing one ablated eLV community per component, with every other
component left intact.

Also provides ablate_eLV_directory(), which cycles through every file in a
directory of CRM or eLV communities, ablates every community in every file,
and saves each ablation type's results to its own subdirectory (v3 method,
eLV_gLV_df_from_communities() in complete_simulation_functions.py).

"""

import numpy as np
import pandas as pd
import os
import sys
from copy import deepcopy
from tqdm import tqdm
from typing import Literal, Union

os.chdir('C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/resource_diversity_stability(sl)/effective_gLV')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/consumer_resource_modules")
from community_level_properties import max_le

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/resource_diversity_stability(sl)")
from complete_simulation_functions import eLV_from_CRM_dynamics, eLV_gLV_df_from_communities

# %%

_CRM_MODEL_NAMES = {"SL_CRM" : "Self-limiting resource supply",
                    "ES_CRM" : "Externally-supplied resources"}

# one entry per correlation, mapping its label to the single
# reconstruct_ablated() kwarg that removes it (and only it)
_CORRELATION_ABLATIONS = {"rho_R"        : dict(keep_row=False),
                          "rho_C"        : dict(keep_col=False),
                          "rho_D"        : dict(keep_residual=False),
                          "rho_r_Aij"    : dict(keep_r_corr=False),
                          "rho_Aii_Aij"  : dict(keep_Aii_corr=False)}

# one entry per mean/variance, mapping its label to the single
# ablate_moments() kwarg that removes it (and only it)
_MOMENT_ABLATIONS = {"mu_r"       : dict(ablate_mu_r=True),
                     "mu_Aij"     : dict(ablate_mu_Aij=True),
                     "sigma_r"    : dict(ablate_sigma_r=True),
                     "sigma_Aij"  : dict(ablate_sigma_Aij=True),
                     "sigma_Aii"  : dict(ablate_sigma_Aii=True)}

# every ablation label this module can produce
_ALL_ABLATIONS = list(_CORRELATION_ABLATIONS) + list(_MOMENT_ABLATIONS)

# %%

def _ablated_community(eLV_community, A_new, r_new, t_end, no_init_cond):

    '''

    Build a fresh community from an ablated (A_new, r_new) pair - a deep
    copy of eLV_community with interaction_matrix/r replaced, then fully
    re-simulated and re-analysed.

    '''

    community = deepcopy(eLV_community)

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

    return community

# %%

def _ablate_single(eLV_community,
                   ablation : str,
                   decomposition : Union[tuple, None] = None,
                   t_end : float = 7000,
                   no_init_cond : int = 1):

    '''

    Perform exactly one ablation on eLV_community, returning the resulting
    (fully re-simulated/re-analysed) ablated community.

    Parameters
    ----------
    eLV_community : eLV_SL or eLV_ES
        Must already have interaction_matrix/r set.
    ablation : str
        One of _ALL_ABLATIONS (rho_R, rho_C, rho_D, rho_r_Aij, rho_Aii_Aij,
        mu_r, mu_Aij, sigma_r, sigma_Aij, sigma_Aii).
    decomposition : tuple, optional
        Precomputed (grand_mean, lam, kap, W) from
        eLV_community.decompose_eLV(), to avoid recomputing it when
        ablating several correlations from the same community in a loop
        (see ablate_eLV_correlations()). Only used for correlation
        ablations; ignored (and unnecessary) for moment ablations. If None
        and ablation is a correlation, decompose_eLV() is called here.
    t_end : float, optional
        Simulation end time. The default is 7000.
    no_init_cond : int, optional
        Number of initial abundances dynamics are simulated from. The
        default is 1.

    Returns
    -------
    community : eLV_SL or eLV_ES
        The ablated community.

    '''

    if ablation in _CORRELATION_ABLATIONS:

        if decomposition is None:

            decomposition = eLV_community.decompose_eLV()

        grand_mean, lam, kap, W = decomposition

        A_new, r_new = eLV_community.reconstruct_ablated(grand_mean, lam, kap, W,
                                                          **_CORRELATION_ABLATIONS[ablation])

    elif ablation in _MOMENT_ABLATIONS:

        A_new, r_new = eLV_community.ablate_moments(**_MOMENT_ABLATIONS[ablation])

    else:

        raise ValueError(f"Unknown ablation '{ablation}'. Must be one of {_ALL_ABLATIONS}.")

    return _ablated_community(eLV_community, A_new, r_new, t_end, no_init_cond)

# %%

def ablate_eLV_correlations(eLV_community,
                            t_end : float = 7000,
                            no_init_cond : int = 1,
                            seed : Union[int, None] = None):

    '''

    Individually remove each of the ten correlations/moments an eLV
    carries, each with every other component left intact. Every ablated
    community is re-simulated and re-analysed from scratch.

    Correlations (rho_R, rho_C, rho_D, rho_r_Aij, rho_Aii_Aij) are removed
    via decompose_eLV()/reconstruct_ablated(). Moments (mu_r, mu_Aij,
    sigma_r, sigma_Aij, sigma_Aii) are removed via ablate_moments() - see
    both for what "removed" means for each.

    For a single ablation type at a time (e.g. to parallelise across
    ablation types), use _ablate_single() instead.

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
        {label : ablated eLV community}, one entry per label in
        _ALL_ABLATIONS (rho_R, rho_C, rho_D, rho_r_Aij, rho_Aii_Aij, mu_r,
        mu_Aij, sigma_r, sigma_Aij, sigma_Aii).

    '''

    if seed is not None:

        np.random.seed(seed)

    decomposition = eLV_community.decompose_eLV()

    return {label : _ablate_single(eLV_community, label, decomposition,
                                   t_end, no_init_cond)
           for label in _ALL_ABLATIONS}

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

    complete_CRM_filepath = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/" + \
                        CRM_filepath

    CRM_communities = pd.read_pickle(complete_CRM_filepath)
    CRM_community = CRM_communities[CRM_index]

    sces = _load_sces(CRM_filepath) if cavity_phi_R is True else None

    eLV_community = _build_eLV_from_CRM(CRM_community, sces)

    # --- ablate each correlation in turn ---

    ablated = ablate_eLV_correlations(eLV_community, t_end, no_init_cond, seed)

    return eLV_community, ablated

# %%
###############################################################################
# Directory-cycling: read every file in a directory of CRM or eLV
# communities, ablate every community in every file, and save each ablation
# type's results (v3 method) to its own subdirectory.
###############################################################################

def _load_sces(CRM_filepath : str):

    '''

    Load the self-consistency-equation lookup table matching a CRM
    filepath's top-level directory (e.g. "M_vs_mu_c/simulations_..." ->
    "M_vs_mu_c.pkl") - same convention as decompose_and_ablate_CRM().

    '''

    return pd.read_pickle("C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/self_consistency_equations/" + \
                          CRM_filepath.split("/")[0] + ".pkl")

def _build_eLV_from_CRM(CRM_community, sces):

    '''

    Build a single eLV community from a single CRM community, optionally
    looking up its cavity-predicted resource survival fraction from sces
    (see decompose_and_ablate_CRM() for the same lookup, applied once
    there instead of per-community).

    '''

    model = _CRM_MODEL_NAMES[type(CRM_community).__name__]

    if sces is not None:

        phiR = sces.loc[np.where((sces["mu_c"] ==
                                  np.round(CRM_community.mu_c * CRM_community.no_resources, 4)) & \
                                 (sces["M"] == CRM_community.no_resources)),
                        "phi_R"].to_numpy()

        return eLV_from_CRM_dynamics(model, [CRM_community], phiR)[0]

    else:

        return eLV_from_CRM_dynamics(model, [CRM_community])[0]

# %%

def ablate_eLV_directory(source_directory : str,
                         output_directory : str,
                         ablation : str,
                         source_type : Literal['CRM', 'eLV'] = 'eLV',
                         cavity_phi_R : bool = False,
                         t_end : float = 7000,
                         no_init_cond : int = 1,
                         seed : Union[int, None] = None):

    '''

    For every file in source_directory (a pickled list of CRM or eLV
    community objects - one file per parameter combination, typically
    several replicate communities each), read in the communities (building
    the eLV first if source_type='CRM'), perform exactly ONE ablation
    (`ablation`) on every community, and save the results as an
    interaction-statistics csv (v3 method, eLV_gLV_df_from_communities() in
    complete_simulation_functions.py) to output_directory/<ablation>/,
    with one csv per source file (same filename, .csv extension).

    This runs a single ablation type across the whole directory - call it
    once per label in _ALL_ABLATIONS (in separate processes, if desired) to
    parallelise across ablation types, rather than looping over all ten
    within one call.

    Parameters
    ----------
    source_directory : str
        Full path to a directory of pickled CRM or eLV community lists.
    output_directory : str
        Full path to a directory to save the ablated results in. Results go
        in output_directory/<ablation>/, one csv per source file.
    ablation : str
        Which single ablation to perform on every community - one of
        _ALL_ABLATIONS (rho_R, rho_C, rho_D, rho_r_Aij, rho_Aii_Aij, mu_r,
        mu_Aij, sigma_r, sigma_Aij, sigma_Aii).
    source_type : str, optional
        'CRM' - build the eLV from each CRM community first (via
            eLV_from_CRM_dynamics()).
        'eLV' - the source files already contain eLV community objects.
        The default is 'eLV'.
    cavity_phi_R : bool, optional
        Only used when source_type = 'CRM'. If True, look up the
        cavity-predicted resource survival fraction for every community
        from the self-consistency-equation file matching
        source_directory's own top-level name (same convention as
        decompose_and_ablate_CRM()). The default is False (all resources
        assumed to survive).
    t_end : float, optional
        Simulation end time for each ablated community. The default is 7000.
    no_init_cond : int, optional
        Number of initial abundances dynamics are simulated from. The
        default is 1.
    seed : int, optional
        Seed for the row/diagonal shuffles reconstruct_ablated() uses when
        ablation is rho_r_Aij/rho_Aii_Aij, for reproducibility. Not used by
        any other ablation (they involve no randomness). The default is
        None.

    Returns
    -------
    None.

    '''

    if ablation not in _ALL_ABLATIONS:

        raise ValueError(f"Unknown ablation '{ablation}'. Must be one of {_ALL_ABLATIONS}.")

    ablation_directory = os.path.join(output_directory, ablation)

    if not os.path.exists(ablation_directory):

        os.makedirs(ablation_directory)

    if seed is not None:

        np.random.seed(seed)

    sces = _load_sces(source_directory) if (source_type == 'CRM' and cavity_phi_R is True) else None

    filenames = os.listdir(source_directory)

    for filename in tqdm(filenames,
                         leave = True,
                         position = 0,
                         total = len(filenames)):

        source_communities = pd.read_pickle(os.path.join(source_directory, filename))

        if source_type == 'CRM':

            eLV_communities = [_build_eLV_from_CRM(CRM_community, sces)
                               for CRM_community in source_communities]

        else:

            eLV_communities = source_communities

        # perform the one requested ablation on every community in this file
        ablated_communities = [_ablate_single(eLV_community, ablation,
                                              t_end = t_end, no_init_cond = no_init_cond)
                               for eLV_community in tqdm(eLV_communities,
                                                        leave = False,
                                                        position = 1,
                                                        total = len(eLV_communities))]

        # save the results (v3 method: interaction-stats csv)
        df = eLV_gLV_df_from_communities(ablated_communities)

        csv_filename = os.path.splitext(filename)[0] + ".csv"

        df.to_csv(os.path.join(ablation_directory, csv_filename),
                 index = False)

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
#
#     # one call per ablation label - run these in parallel (separate
#     # processes/jobs) if desired, since each is fully independent
#     for ablation in _ALL_ABLATIONS:
#
#         ablate_eLV_directory(
#             "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/eLV/M_vs_mu_c",
#             "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/eLV_ablations/M_vs_mu_c",
#             ablation,
#             source_type='eLV',
#             seed=0)
