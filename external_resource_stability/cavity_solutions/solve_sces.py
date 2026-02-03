# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 11:05:34 2026

@author: jamil
"""

import numpy as np
import pandas as pd
import numpy.typing as npt
from typing import Union, Literal, TypedDict
import os
import sys
from copy import copy 
from scipy.interpolate import BarycentricInterpolator

os.chdir('C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/external_resource_stability/cavity_solutions')

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/cavity_method_functions')
import self_consistency_equation_functions as sce

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models" + \
                    "/external_resource_stability")
from simulation_functions import pickle_dump, generate_simulation_df

# %%

def solve_sces(parameters : Union[list, dict],
                solved_quantities : list[str],
                bounds : Union[list[tuple[float], tuple[float]],
                               list[list[tuple[float], tuple[float]]]],
                x_init : Union[npt.NDArray, list[npt.NDArray]],
                solver_name : Literal['basin-hopping',
                                      'dual-annealing',
                                      'least-squares'],
                solver_kwargs : Union[dict, list] = {'xtol' : 1e-13,
                                                     'ftol' : 1e-13},
                other_kwargs : Union[dict, list] = {},
                include_multistability : bool = False):
    
    '''
    

    Parameters
    ----------
    parameters : Union[list, dict]
        DESCRIPTION.
    solved_quantities : list[str]
        DESCRIPTION.
    bounds : Union[list[tuple[float], tuple[float]], list[list[tuple[float], tuple[float]]]]
        DESCRIPTION.
    x_init : Union[npt.NDArray, list[npt.NDArray]]
        DESCRIPTION.
    solver_name : Literal['basin-hopping', 'least-squares']
        DESCRIPTION.
    solver_kwargs : Union[dict, list], optional
        DESCRIPTION. The default is {'xtol' : 1e-13, 'ftol' : 1e-13}.
    other_kwargs : Union[dict, list], optional
        DESCRIPTION. The default is {}.
    include_multistability : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    sol : TYPE
        DESCRIPTION.

    '''
    
    if isinstance(x_init[0], list):
        
        xscales = list(np.power(10, np.floor(np.log10(np.abs(x_init)))))
        solver_kwargs = [dict(list(solver_kwargs.items()) + [('x_scale', xscale)]) 
                         for xscale in xscales]
        
    else:
        
        solver_kwargs['x_scale'] = np.power(10, np.floor(np.log10(np.abs(x_init))))
    
    sol = sce.solve_self_consistency_equations(model = 'externally supplied',
                                               parameters = parameters,
                                               solved_quantities = solved_quantities,
                                               bounds = bounds,
                                               x_init = x_init,
                                               solver_name = solver_name,
                                               solver_kwargs = solver_kwargs,
                                               other_kwargs = other_kwargs,
                                               include_multistability = include_multistability)
     
    sol['Species packing'] = sol['phi_N'] / sol['gamma']
    
    sol['Infeasibility distance'] = sol['phi_N']/sol['gamma'] - 0.5
        
    return sol

# %%

def Global_Solve_SCEs(parameter : str,
                      parm_range : Union[tuple[float, float],
                                         npt.NDArray],
                      n : int, 
                      filename : str,
                      rhos : Union[tuple[float, float],
                                   npt.NDArray] = np.linspace(0.1, 1, 10),
                      any_fixed_parameters : TypedDict('any_fixed_parameters',
                                                       {'mu_c' : float, 'sigma_c' : float,
                                                        'mu_g' : float, 'sigma_g' : float,
                                                        'mu_d' : float, 'sigma_d' : float,
                                                        'mu_b' : float, 'sigma_b' : float,
                                                        'mu_o' : float, 'sigma_o' : float,
                                                        'gamma' : float})
                      = dict(mu_c = 160., sigma_c = 3.,
                             mu_g = 160., sigma_g = 3.,
                             mu_d = 1., sigma_d =  0.,
                             mu_b = 1., sigma_b = 0.,
                             mu_o = 1., sigma_o = 0., 
                             gamma = 1.)):
    
    '''
    

    Parameters
    ----------
    varying_parameter : str
        DESCRIPTION.
    p_range : tuple[float, float]
        DESCRIPTION.
    n : int
        DESCRIPTION.
    filename : str
        DESCRIPTION.
    resource_pool_sizes : npt.NDArray, optional
        DESCRIPTION. The default is np.arange(50, 275, 25).
    any_fixed_parameters : dict, optional
        DESCRIPTION. The default is dict(mu_c = 160,
                                         mu_g = 160,
                                         mu_d = 1, sigma_d =  0,
                                         mu_b = 1, sigma_b = 0,
                                         mu_o = 1, sigma_o = 0,
                                         gamma = 1).

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    def clean_bad_solves(sces, solved_quantities, bounds, x_init,
                         other_kwargs = {'niter' : 500}):
        
        bad_solves = sces.loc[sces['loss'] > -30, :]
        
        if bad_solves.empty is True:
            
            return sces
        
        else:
        
            parameters = bad_solves[['rho', 'gamma',
                                     'mu_c','sigma_c',
                                     'mu_g', 'sigma_g',
                                     'mu_d', 'sigma_d',
                                     'mu_b', 'sigma_b',
                                     'mu_o', 'sigma_o']].to_dict('records')
            
            cleaned_sces = solve_sces(parameters,
                                      solved_quantities,
                                      bounds,
                                      x_init,
                                      #solver_name = 'basin-hopping',
                                      solver_name = 'dual-annealing',
                                      other_kwargs = other_kwargs)
            final_sces = copy(sces)
            bad_solve_idx = final_sces.loc[final_sces['loss'] > -30, :].index.tolist()
            cleaned_sces.rename(index={old_idx : new_idx for old_idx, new_idx in
                                       zip(cleaned_sces.index.tolist(), bad_solve_idx)},
                                inplace = True)
            final_sces.update(cleaned_sces)
            
            return final_sces
        
    # create directory to save data 
    
    directory  = "C:/Users/jamil/Documents/PhD/Data/" + \
                    "external_resource_stability/self_consistency_equations"
                    
    if not os.path.exists(directory): 
        
        os.makedirs(directory) 
        
    variable_parameters = np.unique(sce.parameter_combinations([rhos,
                                                                parm_range],
                                                               n),
                                    axis = 1)
    
    if parameter == 'sigma' or 'mu':
        
        variable_parameters = np.vstack([variable_parameters,
                                         variable_parameters[1, :]])
    
        # array of all parameter combinations
        parameters = sce.variable_fixed_parameters(variable_parameters,
                                                   any_fixed_parameters,
                                                   ['rho',
                                                    parameter + '_c',
                                                    parameter + '_g']).tolist()
        
    else: 
        
        # array of all parameter combinations
        parameters = sce.variable_fixed_parameters(variable_parameters,
                                                   any_fixed_parameters,
                                                   ['rho', parameter]).tolist()
        
    if len(parameters) == 1:
        
        parameters = parameters[0]
        
    solved_quantities = ['phi_N', 'N_mean', 'q_N', 'v_N',
                         'R_mean', 'q_R', 'chi_R']
    
    #bounds = ([1e-10, 1e-10, 1e-10, -1e15, 1e-10, 1e-10, 1e-10],
     #         [0.5, 1e15, 1e15, 1e-10, 1e15, 1e15, 1e15])
     
    bounds = [(1e-10, 0.5),
              (1e-10, 1e15),
              (1e-10, 1e15),
              (-1e15, 1e-10), 
              (1e-10, 1e15),
              (1e-10, 1e15),
              (1e-10, 1e15)]
    
    x_init = np.array([0.1, 1, 30, -0.1, 0.01, 1e-4, 0.05])
    
    solved_sces = solve_sces(parameters, solved_quantities, bounds, x_init,
                             #'basin-hopping', other_kwargs = {'niter' : 200})
                             'dual-annealing', other_kwargs = {'maxiter' : 2000})
    
    final_sces = clean_bad_solves(solved_sces, solved_quantities, bounds, x_init)

    # save data
    
    pickle_dump(directory + "/" + filename + ".pkl", final_sces)
    
    return final_sces

# %%

##################### Solve for rho vs sigma #########################

# globally solved self consistency equations
sces = Global_Solve_SCEs("sigma",
                         np.linspace(2, 5, 11),
                         11,
                         "rho_vs_sigma")

# %%

sces = Global_Solve_SCEs("sigma",
                         (5.0, .0),
                         1,
                         "rho_vs_sigma",
                         rhos = (1, 1),
                         any_fixed_parameters = dict(mu_c = 3., sigma_c = 5.,
                                                     mu_g = 3., sigma_g = 5.,
                                                     mu_d = 1., sigma_d =  0.,
                                                     mu_b = 5., sigma_b = 0.,
                                                     mu_o = 1., sigma_o = 0., 
                                                     gamma = 1.))

# %%

simulations = generate_simulation_df("C:/Users/jamil/Documents/PhD/Data/external_resource_stability/simulations/rho_vs_sigma")