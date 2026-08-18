# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 00:00:00 2026

@author: jamil

Community-object version of diagnose_distribution_mismatch(): compares an
eLV community's r/interaction_matrix against a gLV community's, beyond just
means and pairwise correlations (which calculate_interaction_stats() already
covers) - distribution shape, sign, transitivity, spectrum, and fitness.

"""

import numpy as np
from scipy.stats import skew, kurtosis

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
