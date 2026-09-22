# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 00:00:00 2026

@author: jamil

Rewrites the dataframes created by timescale_separation.py so that the
"timescalar" column is renamed to "timescalar_r", and a new "timescalar_s"
column (all entries = 1.0) is added.
"""

import os
import pandas as pd
from glob import glob

# %%

def rewrite_timescalar_columns(directory : str) -> None:

    csv_files = glob(os.path.join(directory, "*.csv"))

    for csv_file in csv_files:

        df = pd.read_csv(csv_file, index_col = 0)

        df.rename(columns = {'timescalar' : 'timescalar_r'}, inplace = True)
        df['timescalar_s'] = 1.0

        df.to_csv(csv_file)

# %%

base_directory = "C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/CRM_TS"

for subdirectory in ["M_vs_mu_c", "small_separation"]:

    rewrite_timescalar_columns(os.path.join(base_directory, subdirectory))
