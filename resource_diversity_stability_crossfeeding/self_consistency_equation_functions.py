# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:56:36 2025

@author: jamil

Trimmed copy of cavity_method_functions/self_consistency_equation_functions.py,
keeping only the two parameter-grid-building utilities used by the
simulation scripts in this folder (parameter_combinations,
variable_fixed_parameters). The rest of the original file - the
self-consistency-equation solving machinery (solve_self_consistency_equations,
solve_sces, the basin-hopping/dual-annealing/least-squares routines) - depends
on self_limiting_rho_equations, externally_supplied_equations,
self_limiting_gc_c_finite_equations, and self_limiting_g_cg_equations, none of
which are relevant here (this folder runs simulations, not analytical cavity
method solves), so it's been left out rather than dragging in four unrelated
modules just to satisfy unused imports.
"""

# %%

import numpy as np
import numpy.typing as npt
from typing import Union

# %%

def parameter_combinations(parameter_ranges : Union[list[tuple[float, float], tuple[float, float]],
                                                    list[tuple[float, float], npt.NDArray],
                                                    list[npt.NDArray, tuple[float, float]],
                                                    list[npt.NDArray, npt.NDArray]],
                           n : int):

    '''

    Generate all parameter combinations from 2 parameter sets
    Parameters
    ----------
    parameter_ranges : list of tuples or np.ndarray
        tuple = parameter range, np.ndarray =  pre-specified list of parameters.
    n : int
        Number of parameter values per set, if parameters are being generated
        from a range.

    Returns
    -------
    v_p_v_flattened : np.ndarray
        2D array of all parameter combinations.

    '''

    # Generate all parameter combinations from parameter ranges or
    #   pre-specified parameter sets
    variable_parameter_vals = np.meshgrid(*[np.linspace(*val_range, n)
                                            if isinstance(val_range, tuple)
                                            else val_range
                                            for val_range in parameter_ranges])

    # flatten meshgrid to get a 2D array of all parameter combinations
    #   1 row = 1 parameter
    v_p_v_flattened = np.array([v_p_v.flatten() for v_p_v in variable_parameter_vals])

    return v_p_v_flattened

# %%

def variable_fixed_parameters(variable_parameters : list,
                              fixed_parameters : dict,
                              v_names = None):

    '''

    Generate list of dictionaries of parameters

    Parameters
    ----------
    variable_parameters : list of lists, dicts or np.ndarrays
        Array of variable parameter combinations, usually generated using
        parameter_combinations().
    fixed_parameters : dict
        The fixed parameter values.
    v_names : list of str, optional
        Names of the variable parameters if they are a list or array.
        The default is None.

    Returns
    -------
    variable_list : list
        List of dictionaries of parameter sets.

    '''
    if isinstance(variable_parameters[0], (list, np.ndarray)):

        # convert array of variable parameters into a dictionary (with
        #   corresponding parameter names), then merge with the dict of fixed
        #   parameters
        def variable_dict(v_p, v_p_names, fixed_parameters):

            return fixed_parameters | dict(zip(v_p_names, v_p))

        # perform operation on all sets of variable parameters
        variable_list = np.apply_along_axis(variable_dict, 0, variable_parameters,
                                              v_p_names = v_names,
                                              fixed_parameters = fixed_parameters)

    # if variable parameters are already in a list of dicts, merge each dict
    #   with fixed parameters in a list comprehension
    elif isinstance(variable_parameters[0], dict):

        variable_list = [fixed_parameters | v_p for v_p in variable_parameters]

    return variable_list
