# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:37:44 2026

@author: jamil
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:00:00 2026

@author: jamil
"""

import numpy as np
import sys
import os
from pathlib import Path
from copy import deepcopy

# %%

here = Path(__file__).resolve()

stability_dir = here.parents[0]            # stability_transitions
external_dir = here.parents[1]            # external_resource_stability
repo_root = here.parents[2]            # repo root

sys.path.insert(0, str(external_dir))
from complete_simulation_functions import CRM_across_parameter_space

sys.path.insert(0, str(repo_root / "cavity_method_functions"))
import self_consistency_equation_functions as sce

# %%

def rho_sigma(rho_range,
              sigma_range,
              fixed_parameters,
              subdirectory,
              task_id = 1,
              n_tasks = 1,
              **kwargs):
    
    parameters = generate_parameters(rho_range, sigma_range, fixed_parameters)
    
    parameters_chunk = parameters[task_id - 1::n_tasks]
    
    CRM_across_parameter_space(parameters_chunk,
                               subdirectory,
                               ['rho', 'sigma_M'],
                               model = 'MiCRM',
                               save_method = 'v3',
                               **kwargs)
                    
# %%

def generate_parameters(rho_range,
                        sigma_range,
                        fixed_parameters):
    
    rho_sigma_combos = np.unique(sce.parameter_combinations([rho_range,
                                                             sigma_range],
                                                            1),
                                    axis = 1)
    
    variable_parameters = np.vstack([rho_sigma_combos,
                                     rho_sigma_combos[1, :]/np.sqrt(fixed_parameters['M']),
                                     rho_sigma_combos[1, :]/np.sqrt(fixed_parameters['M'])])
    
    fixed_parameters_mod = deepcopy(fixed_parameters)
    
    fixed_parameters_mod['mu_c'] *= 1/fixed_parameters_mod['M']
    fixed_parameters_mod['mu_g'] *= 1/fixed_parameters_mod['M']

    # array of all parameter combinations
    parameters = sce.variable_fixed_parameters(variable_parameters,
                                               fixed_parameters_mod,
                                               ['rho', 'sigma_M',
                                                'sigma_c', 'sigma_g'])
    
    return parameters

# %%

if __name__ == "__main__":
    
    task_id = int(sys.argv[1])
    n_tasks = int(sys.argv[2])
    
    rhos = np.arange(0.1, 1.1, 0.1)
    sigmas = np.arange(2, 13, 1)
    
    fixed_parameters = dict(mu_c = 50,
                            mu_g = 50,
                            d = 1,
                            b = 3.0,
                            o = 0.05,
                            l = 0.7,
                            w = 1,
                            s = 0.05,
                            M = 150,
                            S = 150,
                            all_positive = True)
    
    rho_sigma(rhos,
              sigmas,
              fixed_parameters,
              "MiCRM_rho_vs_sigma_allpos",
              task_id=task_id,
              n_tasks=n_tasks,
              no_communities=20,
              t_end=1000,
              no_init_conds=1)

# %%
