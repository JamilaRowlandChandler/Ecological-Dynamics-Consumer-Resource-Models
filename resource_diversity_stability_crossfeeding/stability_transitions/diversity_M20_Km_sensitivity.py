# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 2026

@author: jamil

Test whether the humped o-vs-diversity relationship found at M=20 (the
clearest example from diversity_low_M_investigation.py/_ps05.py) holds
across K_m in [1e-3, 1e-2, 1e-1], using growth_saturation=True MP_CRM with
the timeout-protected simulation runner (timeout_utils.py). M=20 fixed,
o=[0.1,0.3,0.5,0.7,0.9,1.1], no_communities=20, p_s=min(5/M,1)=0.25, d=0.1,
mu_C=40 (mu_c=mu_C/M), S=50, sigma_C=1.6 - matching
diversity_low_M_investigation.py's M=20 settings except for K_m.
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

M = 20
mu_c = mu_C / M
sigma_c = sigma_C / np.sqrt(M)
p_s = min(5 / M, 1)

K_m_values = [1e-3, 1e-2, 1e-1]
o_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]

if __name__ == '__main__':

    results = {}

    for K_m in K_m_values:

        for condition in ['single', 'all']:

            for o_val in o_values:

                key = (K_m, condition, o_val)
                results[key] = []
                n_timed_out = 0

                for c in range(no_communities):

                    seed = 10000 * M + 1000 * int(round(o_val * 1000)) + c

                    result = simulate_with_timeout(M=M, S=S, mu_c=mu_c, sigma_c=sigma_c, d=d,
                                                   p_s=p_s, K_m=K_m, condition=condition,
                                                   o_val=o_val, seed=seed, t_end=t_end,
                                                   timeout=timeout)

                    if result['timed_out']:
                        n_timed_out += 1

                    results[key].append(result)

                final_fracs = [r['survival_fraction'][-1] for r in results[key]
                              if not r['timed_out']]

                if len(final_fracs) > 0:
                    mean_sf = np.mean(final_fracs)
                    print(f"K_m={K_m}, {condition}, o={o_val}: mean final survival fraction = "
                          f"{mean_sf:.3f} (n_species = {mean_sf*S:.1f}), "
                          f"timed_out={n_timed_out}/{no_communities}", flush=True)
                else:
                    print(f"K_m={K_m}, {condition}, o={o_val}: ALL {no_communities} RUNS TIMED OUT",
                          flush=True)

    with open('diversity_M20_Km_sensitivity_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("Saved results to diversity_M20_Km_sensitivity_results.pkl")
