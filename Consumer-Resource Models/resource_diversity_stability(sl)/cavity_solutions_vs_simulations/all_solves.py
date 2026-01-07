# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 19:13:41 2025

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

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Consumer-Resource Models/resource_diversity_stability(sl)/cavity_solutions_vs_simulations')

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Consumer-Resource Models/cavity_method_functions')
import self_consistency_equation_functions as sce

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
                    "Ecological-Dynamics/Consumer-Resource Models/resource_diversity_stability(sl)")
from simulation_functions import pickle_dump

# %%

def solve_sces(parameters : Union[list, dict],
                    solved_quantities : list[str],
                    bounds : Union[list[tuple[float], tuple[float]],
                                   list[list[tuple[float], tuple[float]]]],
                    x_init : Union[npt.NDArray, list[npt.NDArray]],
                    solver_name : Literal['basin-hopping', 'least-squares'],
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
    
    sol = sce.solve_self_consistency_equations(model = 'self-limiting, yc c',
                                               parameters = parameters,
                                               solved_quantities = solved_quantities,
                                               bounds = bounds,
                                               x_init = x_init,
                                               solver_name = solver_name,
                                               solver_kwargs = solver_kwargs,
                                               other_kwargs = other_kwargs,
                                               include_multistability = include_multistability)
    
    sol['rho'] = np.sqrt(1 / (1 + \
                             ((sol['sigma_y']/sol['mu_y'])**2 * (1 + \
                                                               ((sol['mu_c']**2)/(sol['M'] * sol['sigma_c']**2))))))
    
    sol['Species packing'] = sol['phi_N']/(sol['phi_R'] * sol['gamma'])
    sol['Instability distance'] = sol['rho']**2 - sol['Species packing']
    
    sol['Infeasibility distance'] = sol['phi_R'] - sol['phi_N']/sol['gamma']
        
    return sol

# %%

def Global_Solve_SCEs(varying_parameter : str,
                      p_range : tuple[float, float],
                      n : int, 
                      filename : str,
                      resource_pool_sizes : npt.NDArray = np.arange(50, 275, 25),
                      any_fixed_parameters : TypedDict('any_fixed_parameters',
                                                       {'mu_c' : float, 'sigma_c' : float,
                                                        'mu_y' : float, 'sigma_c' : float,
                                                        'mu_d' : float, 'sigma_d' : float,
                                                        'mu_b' : float, 'sigma_b' : float,
                                                        'gamma' : float})
                      = dict(mu_c = 160, sigma_c = 1.6,
                             mu_y = 1, sigma_y = 0.130639,
                             mu_d = 1, sigma_d =  0,
                             mu_b = 1, sigma_b = 0, 
                             gamma = 1)):
    
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
        DESCRIPTION. The default is dict(mu_c = 160, sigma_c = 1.6,
                                         mu_y = 1, sigma_y = 0.130639,
                                         mu_d = 1, sigma_d =  0,
                                         mu_b = 1, sigma_b = 0, gamma = 1).

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
        
            parameters = bad_solves[['M', 'mu_c', 'mu_y','sigma_c', 
                                     'sigma_y', 'mu_b', 'sigma_b',
                                     'mu_d', 'sigma_d', 'gamma']].to_dict('records')
            
            cleaned_sces = solve_sces(parameters,
                                      solved_quantities,
                                      bounds,
                                      x_init,
                                      solver_name = 'basin-hopping',
                                      other_kwargs = other_kwargs)
            
            cleaned_sces['S'] = cleaned_sces['M']/cleaned_sces['gamma']
            
            final_sces = copy(sces)
            bad_solve_idx = final_sces.loc[final_sces['loss'] > -30, :].index.tolist()
            cleaned_sces.rename(index={old_idx : new_idx for old_idx, new_idx in
                                       zip(cleaned_sces.index.tolist(), bad_solve_idx)},
                                inplace = True)
            final_sces.update(cleaned_sces)
            
            return final_sces
        
    # create directory to save data 
    
    directory  = "C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" + \
                    "resource_diversity_stability/self_consistency_equations"
                    
    #directory = "self_consistency_equations"
        
    if not os.path.exists(directory): 
        
        os.makedirs(directory) 
        
    variable_parameters = np.unique(sce.parameter_combinations([resource_pool_sizes,
                                                                p_range],
                                                               n),
                                    axis = 1)
    
    fixed_parameters = pop_key(any_fixed_parameters, varying_parameter)

    # array of all parameter combinations
    parameters = sce.variable_fixed_parameters(variable_parameters,
                                               fixed_parameters,
                                               ['M', varying_parameter]).tolist()
    
    solved_quantities = ['phi_N', 'N_mean', 'q_N', 'v_N',
                         'phi_R', 'R_mean', 'q_R', 'chi_R']
    
    bounds = ([1e-10, 1e-10, 1e-10, -1e15, 1e-10, 1e-10,  1e-10, 1e-10],
              [1, 1e15, 1e15, 1e-10, 1, 1e15, 1e15, 1e15])
    
    x_init = np.array([0.5, 0.01, 0.01, -0.1, 0.7, 0.001, 0.01, 0.05])
    
    solved_sces = solve_sces(parameters, solved_quantities, bounds, x_init,
                             'basin-hopping', other_kwargs = {'niter' : 200})
    
    solved_sces['S'] = solved_sces['M']/solved_sces['gamma']
    
    final_sces = clean_bad_solves(solved_sces, solved_quantities, bounds, x_init)
    
    solved_sces['M'] = np.int32(solved_sces['M'])

    # save data
    
    pickle_dump(directory + filename + ".pkl", final_sces)
    
    return solved_sces

# %%

def Solve_Stability_Boundary(solved_sces : any,
                             varying_parameter : str,
                             interpolation_protocol : Literal['A', 'B'],
                             filename : str,
                             parameter_bounds : Union[None, list[float, float]] = None):
    
    '''
    

    Parameters
    ----------
    solved_sces : any
        DESCRIPTION.
    varying_parameter : str
        DESCRIPTION.
    quantity_bounds : list[float, float]
        DESCRIPTION.
    interpolation_protocol : Literal['mu', 'sigma']
        DESCRIPTION.

    Returns
    -------
    solved_phase : TYPE
        DESCRIPTION.

    '''
    
    #if not os.path.exists("self_consistency_equations"): 
        
    #    os.makedirs("self_consistency_equations") 
    
    directory = "C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                           + "resource_diversity_stability/self_consistency_equations/stability_bound"
                           
    if not os.path.exists(directory): 
        
        os.makedirs(directory)                      
        
    parm_names = ['mu_c', 'sigma_c', 'mu_y', 'sigma_y',
                  'mu_b', 'sigma_b', 'mu_d', 'sigma_d',
                  'gamma']
    
    solved_quantities = ['phi_N', 'N_mean', 'q_N', 'v_N',
                         'phi_R', 'R_mean', 'q_R', 'chi_R']
    
    
    # get good solves
    solved_sces = solved_sces[solved_sces['loss'] < -30]
    
    # extract parameter set and self-consistency equations where an instability 
    # quantity, dNde, is the largest
    
    solved_sces['dNde'] = np.log10(np.abs(solved_sces['dNde']))

    max_dnde_by_M = solved_sces.groupby('M')['dNde'].max().to_numpy()
    sorter = np.argsort(solved_sces['dNde'].to_numpy())
    max_dNde_df = solved_sces.iloc[sorter[np.searchsorted(solved_sces['dNde'].to_numpy(),
                                                           max_dnde_by_M,
                                                           sorter=sorter)]]
    
    # interpolate extracted variables to get initial values for the solver
    
    match interpolation_protocol:
        
        # I used this for mu_c
        case 'A' :
        
            interpolator_b = BarycentricInterpolator(max_dNde_df['M'],
                                                     max_dNde_df[parm_names + solved_quantities])
            
            interpolated_M = np.arange(np.min(max_dNde_df['M']) + 10,
                                       np.max(max_dNde_df['M']) - 10, 5)
            
            interpolated_matrix = interpolator_b(interpolated_M)
            
            # barycentric interpolated data is a bit wiggly - smooth it out
            interpolators = [np.poly1d(np.polyfit(interpolated_M, interpolated_y, 2))
                             for interpolated_y in interpolated_matrix.T]
            
        # I used this for sigma_c and sigma_y
        case 'B' :
        
            interpolated_M = np.arange(np.min(max_dNde_df['M']),
                                       np.max(max_dNde_df['M']), 5)
            
            # values are a bit dodge below M = 125, so I remove them 
            interpolators = [np.poly1d(np.polyfit(max_dNde_df.loc[max_dNde_df['M'] > 125,
                                                                  'M'],
                                                  max_dNde_df.loc[max_dNde_df['M'] > 125,
                                                                  col],
                                                  1))
                             for col in parm_names + solved_quantities]
            
    interpolated_data = pd.DataFrame([np.round(interpolator(interpolated_M), 7)
                                      for interpolator in interpolators],
                                     index = parm_names + solved_quantities).T
    
    # if these values are exactly at 0, the solver can break immediately 
    # (claims initial values are out of bounds). So, we set them to an arbitrarily
    # small number
    for col in parm_names + ['phi_N', 'N_mean', 'q_N', 'phi_R', 'R_mean', 'q_R']:
        
        interpolated_data.loc[interpolated_data[col] < 0, col] = 1e-10
     
    interpolated_data['M'] = interpolated_M
    
    # get new parameters 
    parm_names.append('M')
    parm_names.remove(varying_parameter)
    parameters = interpolated_data[parm_names].to_dict('records')
    
    # get new initial conditions (sces + parameter being solved for)
    solved_quantities.append(varying_parameter)
    x_init_dicts = interpolated_data[solved_quantities].to_dict('records')
    x_inits = [list(x_init.values()) for x_init in x_init_dicts]
    
    # constrained model bounds (so the solver doesn't wander off)
    
    if isinstance(parameter_bounds, list):
        
        bounds = ([1e-10, 1e-10, 1e-10, -1e15, 1e-10, 1e-10, 1e-10, 1e-10,
                   parameter_bounds[0]], 
                  [1, 1e15, 1e15, 1e-10, 1, 1e15, 1e15, 1e15,
                   parameter_bounds[1]])
    
    else:
        
        bounds = [([1e-10, 1e-10, 1e-10, -1e15, 1e-10, 1e-10, 1e-10, 1e-10,
                    0.85 * x_is[varying_parameter]], 
                   [1, 1e15, 1e15, 1e-10, 1, 1e15, 1e15, 1e15,
                    1.15 * x_is[varying_parameter]])
                  for x_is in x_init_dicts]
    

    # solve for the stability boundary
    solved_boundary = solve_sces(parameters, solved_quantities, bounds, x_inits,
                                 'least-squares', include_multistability = True)
    
    pickle_dump(directory + "/" + filename + ".pkl",
                solved_boundary)
    
    return solved_boundary

# %%

def pop_key(dictionary, key):
    
    dict_copy = dict(dictionary)
    
    dict_copy.pop(key)
    
    return dict_copy

# %%

##################### Solve for mu_c vs M (fig. 3) #########################

# globally solved self consistency equations
sces_mu_c = Global_Solve_SCEs("mu_c", (100, 250), 11, "M_vs_mu_c")

# %%

##################### Solve for sigma_c vs M (fig. 4) #########################

sces_sigma_c = Global_Solve_SCEs('sigma_c', (0.5, 2.5), 11, "M_vs_sigma_c")

# %%

##################### Solve for sigma_y vs M (fig. 4) #########################

sces_sigma_y = Global_Solve_SCEs('sigma_y', (0.05, 0.25), 9, "M_vs_sigma_y")

# %%

################### Load in data and solve for the stability condition #############

# mu_c 

sces_mu_c = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                           + "resource_diversity_stability/self_consistency_equations/" + \
                               "M_vs_mu_c.pkl")

# locally solve the stability boundary
stability_boundary_mu_c = Solve_Stability_Boundary(sces_mu_c,
                                                   "mu_c",
                                                   'A',
                                                   "M_vs_mu_c",
                                                   parameter_bounds = [80, 260])

# %%

# sigma_c

sces_sigma_c = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                              + "resource_diversity_stability/self_consistency_equations/" + \
                                  "M_vs_sigma_c.pkl")

# locally solve the stability boundary
stability_boundary_sigma_c = Solve_Stability_Boundary(sces_sigma_c,
                                                      'sigma_c',
                                                      'B',
                                                      "M_vs_sigma_c",
                                                      parameter_bounds = [0.1, 10])

# sigma_y

sces_sigma_y = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                      + "resource_diversity_stability/self_consistency_equations/" + \
                                          "M_sigma_y.pkl") 

# locally solve the stability boundary
stability_boundary_sigma_y = Solve_Stability_Boundary(sces_sigma_y,
                                                      'sigma_y',
                                                      'B',
                                                      "M_vs_sigma_y",
                                                      parameter_bounds = [0.001, 0.5])

# %%

##################### Solve for other parameters (in SI) #########################

# mean yield conversion factor 

sces_mu_y = Global_Solve_SCEs('mu_y', (0.25, 1), 7, "M_vs_mu_y",
                              fixed_parameters = dict(mu_c = 130, sigma_c = 1.6,
                                                      mu_y = 1, sigma_y = 0.130639,
                                                      mu_d = 1, sigma_d =  0,
                                                      mu_b = 1, sigma_b = 0, 
                                                      gamma = 1))

# death rates

sces_mu_d = Global_Solve_SCEs('mu_d', (0.5, 2.5), 9, "M_vs_mu_d",
                              fixed_parameters = dict(mu_c = 130, sigma_c = 1.6,
                                                      mu_y = 1, sigma_y = 0.130639,
                                                      mu_d = 1, sigma_d =  0,
                                                      mu_b = 1, sigma_b = 0, 
                                                      gamma = 1))

solved_sces_sigma_d = Global_Solve_SCEs('sigma_d', (0, 0.25), "M_vs_sigma_d",
                                        fixed_parameters = dict(mu_c = 130, sigma_c = 1.6,
                                                                mu_y = 1, sigma_y = 0.130639,
                                                                mu_d = 1, sigma_d =  0,
                                                                mu_b = 1, sigma_b = 0, 
                                                                gamma = 1))

# intrinsic resource growth rates 

solved_sces_mu_b = Global_Solve_SCEs('mu_b', (0.5, 2.5), 9, "M_vs_mu_b",
                                     fixed_parameters = dict(mu_c = 130, sigma_c = 1.6,
                                                             mu_y = 1, sigma_y = 0.130639,
                                                             mu_d = 1, sigma_d =  0,
                                                             mu_b = 1, sigma_b = 0, 
                                                             gamma = 1))

solved_sces_sigma_b = Global_Solve_SCEs('sigma_b', (0, 0.25), 9, "M_vs_sigma_b",
                                        fixed_parameters = dict(mu_c = 130, sigma_c = 1.6,
                                                                mu_y = 1, sigma_y = 0.130639,
                                                                mu_d = 1, sigma_d =  0,
                                                                mu_b = 1, sigma_b = 0, 
                                                                gamma = 1))
