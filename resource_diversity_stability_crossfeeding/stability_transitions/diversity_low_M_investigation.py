# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 2026

@author: jamil

Investigate the humped o-vs-diversity relationship seen at low M (which
flips sign depending on single-source vs all-resources-supplied injection),
using growth_saturation=True MP_CRM with the timeout-protected simulation
runner (timeout_utils.py). M=[5,10,15,20], o=[0.1,0.3,0.5,0.7,0.9,1.1],
no_communities=20, p_s=min(5/M,1) (scaled with M), K_m=1e-4, d=0.1,
mu_C=40 (mu_c=mu_C/M), S=50, sigma_C=1.6 - all matching the earlier
coexistence investigations except the M/o ranges and community count.
"""

import numpy as np
import sys
import os
import pickle

abspath = os.path.abspath(__file__)
file_directory_name = os.path.dirname(abspath)
os.chdir(file_directory_name)

sys.path.insert(0, file_directory_name)
from timeout_utils import simulate_with_timeout

# %%

mu_C = 40
sigma_C = 1.6
d = 0.1
S = 50
t_end = 7000
no_communities = 20
timeout = 60

M_values = [5, 10, 15, 20]
o_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]

if __name__ == '__main__':

    results = {}

    for M in M_values:

        mu_c = mu_C / M
        sigma_c = sigma_C / np.sqrt(M)
        p_s = min(5 / M, 1)

        for condition in ['single', 'all']:

            for o_val in o_values:

                key = (M, condition, o_val)
                results[key] = []
                n_timed_out = 0

                for c in range(no_communities):

                    seed = 10000 * M + 1000 * int(round(o_val * 1000)) + c

                    result = simulate_with_timeout(M=M, S=S, mu_c=mu_c, sigma_c=sigma_c, d=d,
                                                   p_s=p_s, K_m=1e-4, condition=condition,
                                                   o_val=o_val, seed=seed, t_end=t_end,
                                                   timeout=timeout)

                    if result['timed_out']:
                        n_timed_out += 1

                    results[key].append(result)

                final_fracs = [r['survival_fraction'][-1] for r in results[key]
                              if not r['timed_out']]

                if len(final_fracs) > 0:
                    mean_sf = np.mean(final_fracs)
                    print(f"M={M}, {condition}, o={o_val}: mean final survival fraction = "
                          f"{mean_sf:.3f} (n_species = {mean_sf*S:.1f}), "
                          f"timed_out={n_timed_out}/{no_communities}", flush=True)
                else:
                    print(f"M={M}, {condition}, o={o_val}: ALL {no_communities} RUNS TIMED OUT",
                          flush=True)

    with open('diversity_low_M_investigation_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("Saved results to diversity_low_M_investigation_results.pkl")
