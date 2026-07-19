# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 2026

@author: jamil

Merge the per-M pickles produced by diversity_over_time_M_o.py (run
separately per M to avoid the accumulated-solve_ivp-calls slowdown) into a
single diversity_over_time_M_o_results.pkl.
"""

import pickle
import os

abspath = os.path.abspath(__file__)
os.chdir(os.path.dirname(abspath))

M_values = [10, 20, 50, 100]

results = {}
for M in M_values:
    with open(f'diversity_over_time_M_o_M{M}.pkl', 'rb') as f:
        results.update(pickle.load(f))

with open('diversity_over_time_M_o_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"Merged {len(results)} (M, condition, o) keys into diversity_over_time_M_o_results.pkl")
