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

DATA_DIR = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability_crossfeeding/influx_species_diversity"

M_values = [10, 20, 50, 100]

results = {}
for M in M_values:
    with open(os.path.join(DATA_DIR, f'diversity_over_time_M_o_M{M}.pkl'), 'rb') as f:
        results.update(pickle.load(f))

out_path = os.path.join(DATA_DIR, 'diversity_over_time_M_o_results.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(results, f)

print(f"Merged {len(results)} (M, condition, o) keys into {out_path}")
