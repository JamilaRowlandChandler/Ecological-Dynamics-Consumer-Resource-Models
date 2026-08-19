# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 00:00:00 2026

@author: jamil

Community-object version of diagnose_distribution_mismatch(): compares an
eLV community's r/interaction_matrix against a gLV community's, beyond just
means and pairwise correlations (which calculate_interaction_stats() already
covers) - distribution shape, sign, transitivity, spectrum, and fitness.

Also loads an eLV community from a pickled file and generates its
corresponding gLV community, following the same pooled-statistics approach
as averaged_stats_gLV.py, so the two can be diagnosed directly.

"""

import numpy as np
import pandas as pd
import os
import sys
from scipy.stats import skew, kurtosis
from typing import Literal, Union

os.chdir('C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/resource_diversity_stability(sl)/effective_gLV')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/consumer_resource_modules")

# needed to import the sibling averaged_stats_gLV module below when this
# file isn't the one being run directly (only the __main__ script's own
# directory is added to sys.path automatically)
sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/resource_diversity_stability(sl)/effective_gLV")

from averaged_stats_gLV import average_eLV_statistics, gLV_from_averaged_stats

# %%

def diagnose_distribution_mismatch(eLV_community, gLV_community):

    '''

    Compare the eLV and gLV parameter distributions beyond means and
    pairwise correlations.

    Parameters
    ----------
    eLV_community : eLV_SL or eLV_ES
        eLV community object (must already have r/interaction_matrix set,
        i.e. generate_elv_parameters() has been called).
    gLV_community : gLV
        gLV community object (must already have r/interaction_matrix set,
        i.e. model_specific_rates() has been called).

    Returns
    -------
    None.

    '''

    r_elv = eLV_community.r
    A_elv = eLV_community.interaction_matrix

    r_glv = gLV_community.r
    A_glv = gLV_community.interaction_matrix

    S = A_elv.shape[0]
    mask = ~np.eye(S, dtype=bool)

    elv_offdiag = A_elv[mask]
    glv_offdiag = A_glv[mask]

    # 1. Non-Gaussianity of interactions

    print("--- Distribution shape of off-diagonal A_ij ---")
    print(f"  eLV: skew={skew(elv_offdiag):.4f}  "
          f"kurtosis={kurtosis(elv_offdiag):.4f}  "
          f"min={elv_offdiag.min():.4f}  max={elv_offdiag.max():.4f}")
    print(f"  gLV: skew={skew(glv_offdiag):.4f}  "
          f"kurtosis={kurtosis(glv_offdiag):.4f}  "
          f"min={glv_offdiag.min():.4f}  max={glv_offdiag.max():.4f}")

    # 2. Fraction of negative interactions (facilitation)
    print(f"\n--- Negative entries (facilitation) ---")
    print(f"  eLV: {(elv_offdiag < 0).sum()} / {elv_offdiag.size} "
          f"({(elv_offdiag < 0).mean()*100:.1f}%)")
    print(f"  gLV: {(glv_offdiag < 0).sum()} / {glv_offdiag.size} "
          f"({(glv_offdiag < 0).mean()*100:.1f}%)")

    # 3. Transitivity: if A_ij and A_ik are both large,
    #    is A_jk also large? Measure via conditional correlation.
    B_elv = A_elv[mask].reshape(S, S - 1)
    B_glv = A_glv[mask].reshape(S, S - 1)

    # for each row, rank the entries and check whether
    # species that i competes with strongly also compete
    # with each other strongly
    print(f"\n--- Competitive transitivity ---")
    for label, A_mat in [("eLV", A_elv), ("gLV", A_glv)]:
        trans_scores = []
        for i in range(S):
            others = np.delete(np.arange(S), i)
            row_i = A_mat[i, others]
            # top quartile of i's competitors
            top = others[row_i > np.percentile(row_i, 75)]
            if len(top) < 2:
                continue
            # mean interaction among i's top competitors
            mutual = [A_mat[j, k] for j in top for k in top if j != k]
            # mean interaction among random pairs
            rand_pairs = np.random.choice(others, size=(len(mutual), 2))
            random_mutual = [A_mat[j, k] for j, k in rand_pairs if j != k]
            if mutual and random_mutual:
                trans_scores.append(np.mean(mutual) / (np.mean(random_mutual) + 1e-20))
        print(f"  {label}: mean transitivity ratio = {np.mean(trans_scores):.4f}  "
              f"(1.0 = no transitivity, >1 = transitive competition)")

    # 4. Eigenvalue comparison
    eigs_elv = np.linalg.eigvals(A_elv)
    eigs_glv = np.linalg.eigvals(A_glv)

    print(f"\n--- Eigenvalue spectrum ---")
    print(f"  eLV: max_real={np.max(eigs_elv.real):.4f}  "
          f"spectral_radius={np.max(np.abs(eigs_elv)):.4f}  "
          f"n_real={np.sum(np.abs(eigs_elv.imag) < 1e-6)}")
    print(f"  gLV: max_real={np.max(eigs_glv.real):.4f}  "
          f"spectral_radius={np.max(np.abs(eigs_glv)):.4f}  "
          f"n_real={np.sum(np.abs(eigs_glv.imag) < 1e-6)}")

    # 5. Fitness distribution at equal abundances
    f_elv = r_elv - A_elv.sum(axis=1)
    f_glv = r_glv - A_glv.sum(axis=1)

    print(f"\n--- Fitness at N=1 ---")
    print(f"  eLV: mean={f_elv.mean():.4f}  std={f_elv.std():.4f}  "
          f"cv={f_elv.std()/(abs(f_elv.mean())+1e-20):.4f}")
    print(f"  gLV: mean={f_glv.mean():.4f}  std={f_glv.std():.4f}  "
          f"cv={f_glv.std()/(abs(f_glv.mean())+1e-20):.4f}")

# %%

def load_eLV_and_generate_gLV(filepath : str,
                              include_rho : Union[list[Literal["rho_D",
                                                               "rho_R",
                                                               "rho_C",
                                                               "rho_1idx"]],
                                                  None] =
                              ["rho_D", "rho_R", "rho_C", "rho_1idx"],
                              empirical : bool = True):

    '''

    Load all eLV communities pickled in filepath, pool their statistics, and
    generate a single gLV community matching those pooled statistics - the
    same pooled-statistics approach as
    gLV_communities_from_averaged_eLV()/gLV_M_averaged() in
    averaged_stats_gLV.py, but returning a representative eLV community
    alongside the single generated gLV, for use with
    diagnose_distribution_mismatch().

    Parameters
    ----------
    filepath : str
        Full path to a pickled file of eLV community objects.
    include_rho : list of str, or None, optional
        Which interaction-matrix correlations to pool and use to generate
        correlated gLV interactions. The default is
        ["rho_D", "rho_R", "rho_C", "rho_1idx"].
    empirical : bool, optional
        Only used when rho_r_Aij/rho_Aii_Aij are requested (i.e. include_rho
        is not None) - see gLV.__correlated_rates() in effective_LV_models.py
        for what this controls. The default is True.

    Returns
    -------
    eLV_community : eLV_SL or eLV_ES
        A representative eLV community from filepath (the first one) -
        pooled statistics are drawn from the whole file, but a single eLV
        community is needed to compare against the single generated gLV.
    gLV_community : gLV
        gLV community generated from the pooled eLV statistics.

    '''

    eLV_communities = pd.read_pickle(filepath)

    averaged_stats = average_eLV_statistics(eLV_communities, include_rho)

    # assumes every eLV community in the file has the same species pool size
    no_species = eLV_communities[0].no_species

    gLV_community = gLV_from_averaged_stats(averaged_stats, no_species, include_rho,
                                            empirical)

    return eLV_communities[0], gLV_community

# %%

def diagnose_from_file(filepath : str,
                       include_rho : Union[list[Literal["rho_D",
                                                        "rho_R",
                                                        "rho_C",
                                                        "rho_1idx"]],
                                           None] =
                       ["rho_D", "rho_R", "rho_C", "rho_1idx"],
                       empirical : bool = True):

    '''

    Load an eLV community from filepath, generate its corresponding gLV
    community (c.f. load_eLV_and_generate_gLV()), and diagnose the
    distribution mismatch between the two.

    Parameters
    ----------
    filepath : str
        Full path to a pickled file of eLV community objects.
    include_rho : list of str, or None, optional
        The default is ["rho_D", "rho_R", "rho_C", "rho_1idx"].
    empirical : bool, optional
        Only used when rho_r_Aij/rho_Aii_Aij are requested (i.e. include_rho
        is not None) - see gLV.__correlated_rates() in effective_LV_models.py
        for what this controls. The default is True.

    Returns
    -------
    eLV_community : eLV_SL or eLV_ES
        The representative eLV community used in the comparison.
    gLV_community : gLV
        The gLV community generated from the pooled eLV statistics.

    '''

    eLV_community, gLV_community = load_eLV_and_generate_gLV(filepath, include_rho,
                                                              empirical)

    diagnose_distribution_mismatch(eLV_community, gLV_community)

    return eLV_community, gLV_community
