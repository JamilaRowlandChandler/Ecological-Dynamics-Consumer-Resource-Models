# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 00:00:00 2026

@author: jamil

Complementary version of ts_eigenspectrum.py for eLV models derived from
self-limiting resource supply CRM communities (eLV_SL class). Unlike the CRM
communities in ts_eigenspectrum.py, eLV_SL communities have no separate
resource dynamics to timescale-separate (their resource dynamics have
already been integrated out into the interaction matrix, i.e. the
epsilon -> 0 limit) - so each CRM community is converted into a single eLV_SL
community, rather than being resimulated across a range of epsilons.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
from typing import Literal, Union
import numpy.typing as npt
from tqdm import tqdm
from scipy.linalg import schur
from matplotlib import pyplot as plt

os.chdir('C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/resource_diversity_stability(sl)/effective_gLV')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/consumer_resource_modules")
from effective_LV_models import eLV_SL
from community_level_properties import max_le, eigenspectrum

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/resource_diversity_stability(sl)")
from complete_simulation_functions import pickle_dump

# %%

def elv_from_CRM_community(CRM_community : Literal["SL_CRM"],
                           cavity_phi_R : Union[float, None] = None):

    eLV_community = eLV_SL(no_species = CRM_community.no_species,
                           no_resources = CRM_community.no_resources)

    eLV_community.elv_from_crm(CRM_community, cavity_phi_R)
    eLV_community.generate_elv_parameters()

    eLV_community.calculate_interaction_stats()

    # run simulations from randomly generated initial abundances
    eLV_community.simulate_community(t_end = 7000,
                                     no_init_cond = 1)

    eLV_community.calculate_community_properties()

    # numerically estimate the max. lyapunov exponent
    eLV_community.lyapunov_exponent = max_le(eLV_community,
                                             eLV_community.ODE_sols[0].y[:, -1],
                                             T = 1000,
                                             perturbation = 1e-6)

    eLV_community.eigenspec_stats = [eigenspectrum(eLV_community,
                                                    ode_sol.y[:, -1])
                                     for ode_sol in eLV_community.ODE_sols]

    return eLV_community

# %%

def eLV_eigenspec_from_CRM(CRM_directory : str,
                           resource_pool_sizes,
                           mu_c,
                           cavity_phi_R : Union[float, None] = None):


    def read_call_elv_from_crm(full_CRM_directory : str):

        # read in consumer-resource model (CRM) communities
        CRM_communities = pd.read_pickle(full_CRM_directory)

        eLV_communities = [elv_from_CRM_community(CRM_community, cavity_phi_R)
                          for CRM_community in tqdm(CRM_communities[7:9],
                                                    leave = True,
                                                    position = 1,
                                                    total = len(CRM_communities[7:9]))]

        return eLV_communities

    ###################################################################################

    full_CRM_directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/" + \
                           CRM_directory

    # generate filenames based on mu_c
    filenames = [full_CRM_directory + "/simulations_" + \
                 str(M) + "_" + str(np.round(mu_c/M, 4)) + ".pkl"
                 for M in resource_pool_sizes]

    eLV_eigenspec = {str(M) : read_call_elv_from_crm(filename)
                     for filename, M in zip(filenames, resource_pool_sizes)}

    return eLV_eigenspec

# %%

def eLV_eigenspec_from_existing(eLV_directory : str,
                                resource_pool_sizes,
                                mu_c,
                                community_indices : list[int] = [7, 8]):

    '''

    Load already-simulated eLV_SL communities from disk (e.g. as saved by
    eLV_M() in all_mu_c_vs_M_egLV.py, under
    .../simulations/eLV/M_vs_mu_c/) and compute their eigenspectra, rather
    than deriving/resimulating eLV_SL communities from CRM communities (see
    eLV_eigenspec_from_CRM() for that).

    Parameters
    ----------
    eLV_directory : str
        Directory (relative to
        C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/)
        containing the pickled eLV_SL communities, e.g. "eLV/M_vs_mu_c".
    resource_pool_sizes : array-like
        Resource pool sizes (M) to load, used to build filenames.
    mu_c : float
        Mean consumption rate (unscaled by M), used to build filenames -
        matches the naming convention of eLV_M()/CRMs_create_and_save().
    community_indices : list[int], optional
        Indices into each file's list of pre-simulated eLV_SL communities to
        use. The default is [7, 8], matching the community indices used by
        eLV_eigenspec_from_CRM()/timescale_separation_eigenspec().

    Returns
    -------
    eLV_eigenspec : dict
        {M : [community, ...]}.

    '''

    def read_existing_eLV(full_eLV_directory : str):

        # read in already-simulated eLV_SL communities
        eLV_communities = [pd.read_pickle(full_eLV_directory)[idx]
                          for idx in community_indices]

        for eLV_community in eLV_communities:

            # reuse the already-computed max. lyapunov exponent if present,
            # otherwise fall back to computing it - keeps the attribute name
            # consistent with elv_from_CRM_community()'s output
            eLV_community.lyapunov_exponent = \
                getattr(eLV_community, "max_lyapunov_exponent", None)

            if eLV_community.lyapunov_exponent is None:

                eLV_community.lyapunov_exponent = max_le(eLV_community,
                                                         eLV_community.ODE_sols[0].y[:, -1],
                                                         T = 1000,
                                                         perturbation = 1e-6)

            eLV_community.eigenspec_stats = [eigenspectrum(eLV_community,
                                                            ode_sol.y[:, -1])
                                             for ode_sol in eLV_community.ODE_sols]

        return eLV_communities

    ###################################################################################

    full_eLV_directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/" + \
                           eLV_directory

    # generate filenames based on mu_c
    filenames = [full_eLV_directory + "/simulations_" + \
                 str(M) + "_" + str(np.round(mu_c/M, 4)) + ".pkl"
                 for M in resource_pool_sizes]

    eLV_eigenspec = {str(M) : read_existing_eLV(filename)
                     for filename, M in zip(filenames, resource_pool_sizes)}

    return eLV_eigenspec

# %%

def absolute_eigenvec_contribution(eigenspecs):

    df = pd.DataFrame([absolute_eigenvec_contr_stats(community)
                       for communities in eigenspecs.values()
                       for community in communities])

    return df

def absolute_eigenvec_contr_stats(community):

    M = community.no_resources
    max_le = community.lyapunov_exponent

    leading_eigenvector = community.eigenspec_stats[0]["leading_vec"]

    # no separate species/resource pools to split a contribution across -
    # eLV_SL's state is species-only (resources are already integrated out
    # into the interaction matrix), so just report the leading eigenvector's
    # mean absolute magnitude across species
    vec_magnitude = np.mean(np.abs(leading_eigenvector))

    return dict(M = M,
               max_le = max_le,
               vec_magnitude = vec_magnitude)

# %%

def eigenspectrum_matrix(community):

    surviving_species = community.ODE_sols[0].y[:, -1] > 1e-4

    surviving_interaction_matrix = \
        community.interaction_matrix[np.ix_(surviving_species, surviving_species)]

    T, Z = schur(-surviving_interaction_matrix, output='complex')
    eigenvalues = np.diag(T)

    leading_val = eigenvalues[np.argmax(eigenvalues.real)]

    return {'eigenspectrum' : eigenvalues, 'leading_val' : leading_val}

# %%

def save_eigenspec_stats(CRM_ts_eLV : dict,
                         filepath : str) -> None:

    '''

    Save just the eigenspectrum stats (and max. lyapunov exponent) for every
    community in CRM_ts_eLV - a lightweight pickle that leaves out the much
    larger ODE_sols trajectories.

    Parameters
    ----------
    CRM_ts_eLV : dict
        {M : [community, ...]}, as returned by eLV_eigenspec_from_CRM().
    filepath : str
        Full filepath (including filename and extension) to save to.

    Returns
    -------
    None.

    '''

    eigenspec_stats = {M : [dict(eigenspec_stats = community.eigenspec_stats,
                                 lyapunov_exponent = community.lyapunov_exponent)
                            for community in communities]
                       for M, communities in CRM_ts_eLV.items()}

    pickle_dump(filepath, eigenspec_stats)

# %%

def save_example_trajectories(CRM_ts_eLV : dict,
                              community_indices : list[int],
                              filepath : str) -> None:

    '''

    Save (t, y) from ODE_sols[0] for a chosen subset of communities, as a
    pickle.

    Parameters
    ----------
    CRM_ts_eLV : dict
        {M : [community, ...]}, as returned by eLV_eigenspec_from_CRM().
    community_indices : list[int]
        Indices into each M's list of communities (i.e. into CRM_ts_eLV[M])
        to save trajectories for.
    filepath : str
        Full filepath (including filename and extension) to save to.

    Returns
    -------
    None.

    '''

    example_trajectories = {M : {idx : (communities[idx].ODE_sols[0].t, communities[idx].ODE_sols[0].y)
                                 for idx in community_indices}
                            for M, communities in CRM_ts_eLV.items()}

    pickle_dump(filepath, example_trajectories)

# %%

# source = 'CRM' derives/resimulates eLV_SL communities from CRM communities
# (see eLV_eigenspec_from_CRM()). source = 'existing' instead loads already-
# simulated eLV_SL communities from disk and just computes their eigenspectra
# (see eLV_eigenspec_from_existing()) - much faster if they're already there.
source = 'CRM'

match source:

    case 'CRM':

        CRM_ts_eLV = eLV_eigenspec_from_CRM(CRM_directory = "M_vs_mu_c",
                                            resource_pool_sizes = np.array([50, 250]),
                                            mu_c = 145)

    case 'existing':

        CRM_ts_eLV = eLV_eigenspec_from_existing(eLV_directory = "eLV/M_vs_mu_c",
                                                 resource_pool_sizes = np.array([50, 250]),
                                                 mu_c = 145)

# %%

GC_eigenspec_eLV = {M : [eigenspectrum_matrix(community) for community in communities]
                    for M, communities in CRM_ts_eLV.items()}

# %%

eigenspec_directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/CRM_TS/eigenspectra"

if not os.path.exists(eigenspec_directory):

    os.makedirs(eigenspec_directory)

save_eigenspec_stats(CRM_ts_eLV,
                     eigenspec_directory + "/M_vs_mu_c_eLV_eigenspec_stats.pkl")

save_example_trajectories(CRM_ts_eLV,
                          community_indices = [0],
                          filepath = eigenspec_directory + "/M_vs_mu_c_eLV_example_trajectories.pkl")
