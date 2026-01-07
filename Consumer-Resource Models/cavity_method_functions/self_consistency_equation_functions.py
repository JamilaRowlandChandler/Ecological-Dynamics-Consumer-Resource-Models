# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:56:36 2025

@author: jamil
"""

# %%

########################

import numpy as np
import numpy.typing as npt
from typing import Union, Literal
import pandas as pd
import os
import pickle
from tqdm import tqdm
from inspect import getfullargspec

from scipy.optimize import least_squares
from scipy.optimize import basinhopping

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models/cavity_method_functions')

import self_limiting_rho_equations as slr
import externally_supplied_equations as es
import self_limiting_gc_c_finite_equations as slgcM
import self_limiting_g_cg_equations as slcg

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
            
            return dict(zip(v_p_names, v_p)) | fixed_parameters 
        
        # perform operation on all sets of variable parameters
        variable_list = np.apply_along_axis(variable_dict, 0, variable_parameters,
                                              v_p_names = v_names,
                                              fixed_parameters = fixed_parameters)
    
    # if variable parameters are already in a list of dicts, merge each dict
    #   with fixed parameters in a list comprehension
    elif isinstance(variable_parameters[0], dict):
        
        variable_list = [v_p | fixed_parameters for v_p in variable_parameters]
                         
    return variable_list

# %%

def solve_self_consistency_equations(model : Literal['self-limiting, rho',
                                                     'self-limiting, yc c',
                                                     'self-limiting, g cg',
                                                     'externally supplied'],
                                     parameters : Union[list, dict],
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
    
    Calls solve_sces for solving the system of self-consistency equations 
    (phi_N, N_mean, q_N, v_N, phi_R, R_mean, q_R, and chi_R) and any other model parameters.
    It then solves for the stability quations in eq. (62) and (63) of the SI (dNde and dRde)

    Parameters
    ----------
    model : str
        Name of the system of self-consistency equations you want to solve, aka
        the model you're solving.
        The options are 'self-limiting, rho' (Blumenthal et al., 2024),
        'self-limiting, yc c' (our model), 'self-limiting, g cg' (consumption
         is coupled to growth), 'externally supplied' (chemostat-style system
         with non-reciprocol growth and consumption rates)
    parameters : list of dicts or dict
        Values that are not being solved for. Usually the statistical properties 
        of the distributions of model parameters (e.g. mean consumption rate)
    solved_quantities : list of str
        The names of the quantities being solved for. Usually the self-consistency
        equations.
    bounds : list containing 2 tuples, or list of lists of tuples 
        The lower and upper bounds of the quantities being solved for.
    x_init : np.ndarray, or list of lists
        The initial values of the solved_quantities.
    solver_name : str
        The routine for solving the self consistency equations.
        Options: 'basin-hopping' (global optimisation for least-squares),
        'least-squares' (local optimisation). 
    solver_kwargs : dict or list of dicts, optional
        Options for the least-squares solver, which is called by both the 
        'basin-hopping' and 'least squares' routine.
        The default is {'xtol' : 1e-13, 'ftol' : 1e-13}  .                                                  
    other_kwargs : dict or list of dicts, optional
        Options for the basin-hopping solver. The default is {}.
    include_multistability : Bool, optional
        Whether or not the stability condition should be solved as well.
        Set as True if you are solving for the phase boundary, or False otherwise.
        The default is False.

    Returns
    -------
    sol : pd.DataFrame
        Dataframe of each parameter set + values of the solved self consistency
        equations + multistability equations.

    '''
    
    # Pick the solver
    match solver_name:
        
        case 'basin-hopping':
            
            solver = solve_equations_basinhopping
            
        case 'least-squares':
            
            solver = solve_equations_least_squares
    
    # This routine solves the self consistency equations
    sol = solve_sces(parameters, model,
                     solved_quantities = solved_quantities,
                     bounds = bounds,
                     x_init = x_init,
                     solver = solver,
                     solver_kwargs = solver_kwargs,
                     other_kwargs = other_kwargs,
                     include_multistability = include_multistability)
    
    # This routine solves the multistability equations for each row of the df
    #   (which contains each parameter set + solved self consistency equations)
    sol[['dNde', 'dRde', 'ms_loss']] = \
        pd.DataFrame(sol.apply(solve_for_multistability, axis = 1,
                               multistability_equation_func = model).to_list())
        
    return sol
    
# %%

def solve_sces(parameters : Union[list, dict],
               model : Literal['self-limiting, rho', 'self-limiting, yc c',
                               'self-limiting, g cg', 'externally supplied'],
               solved_quantities : list[str],
               bounds : Union[list[tuple[float], tuple[float]],
                              list[list[tuple[float], tuple[float]]]],
               x_init : Union[npt.NDArray, list[npt.NDArray]],
               solver,
               solver_kwargs : Union[dict, list] = {'xtol' : 1e-13,
                                                    'ftol' : 1e-13},
               other_kwargs : Union[dict, list] = {},
               include_multistability : bool = False):
    
    '''
    
    Solve the self-consistency equations

    Parameters
    ----------
    model : str
        Name of the system of self-consistency equations you want to solve, aka
        the model you're solving.
        The options are 'self-limiting, rho' (Blumenthal et al., 2024),
        'self-limiting, yc c' (our model), 'self-limiting, g cg' (consumption
        is coupled to growth), 'externally supplied' (chemostat-style system
        with non-reciprocol growth and consumption rates)
    parameters : list of dicts or dict
        Values that are not being solved for. Usually the statistical properties 
        of the distributions of model parameters (e.g. mean consumption rate)
    solved_quantities : list of str
        The names of the quantities being solved for. Usually the self-consistency
        equations.
    bounds : list containing 2 tuples, or list of lists of tuples 
        The lower and upper bounds of the quantities being solved for.
    x_init : list of np.ndarray, or list of lists
        The initial values of the solved_quantities.
    solver : function name
        The routine for solving the self consistency equations.
        Options: 'basin-hopping' (global optimisation for least-squares),
        'least-squares' (local optimisation). 
    solver_kwargs : dict or list of dicts, optional
        Options for the least-squares solver, which is called by both the 
        'basin-hopping' and 'least squares' routine.
        The default is {'xtol' : 1e-13, 'ftol' : 1e-13}  .                                                  
    other_kwargs : dict or list of dicts, optional
        Options for the basin-hopping solver. The default is {}.
    include_multistability : Bool, optional
        Whether or not the stability condition should be solved as well.
        Set as True if you are solving for the phase boundary, or False otherwise.
        The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    # Identify which set of self consistency equations to solve for (model
    #   dependent)
    match model:
        
        case 'self-limiting, rho':
            
            module = slr
            
        case 'self-limiting, yc c':
            
            module = slgcM
            
        case 'self-limiting, g cg':
            
            module = slcg
            
        case 'externally supplied':
            
            module = es
            
    function = module.self_consistency_equations
    
    # Identify whether to include the stability condition
    if include_multistability == True:
        
        ms_function = module.instability_condition
        
    else:
        
        ms_function = None
    
    def create_iterator(args_list):
        
        '''
        
        Creater iterator of solver routine arguments from the relevent args
        of solve_sces(). (Sorry, it's a bit spaghetti-fied.)

        Parameters
        ----------
        args_list : list
            List of all solver arguments. MUST be in the same order as 
            new_arg_names.

        Returns
        -------
        iterable_kwargs : list
            Iterator of solver routine arguments

        '''
        
        # names of the corresponding solver routine args to the relevent 
        #   solve_sce() args.
        new_arg_names = ['solved_quantities', 'bounds', 'x_init', 'ls_kwargs',
                         'solver_kwargs', 'other_kwargs']
       
        # if an arg is of these types, it'll need iterating through
        collection_types_1 = (list, np.ndarray)
        collection_types_2 = (list, tuple, dict, np.ndarray)
        
        # find the no. times/sets that need iterating through (n parameter sets
        #   = n iters)
        if isinstance(args_list[3], dict):
            
            iterable_args_dict = {arg_name : [arg]
                                  for arg_name, arg in zip(new_arg_names, args_list)}
            
        else: 
            
            max_iter = len(args_list[3]) # no of parameter sets
            
            # Create dictionary where the key is the solver arg name, and the value
            #   are the sets that need iterating through
            
            iterable_args_dict = {arg_name : (arg if any(arg) 
                                              and isinstance(arg, collection_types_1)
                                              and isinstance(arg[0], collection_types_2)
                                              else [arg for _ in range(max_iter)])
                                  for arg_name, arg in zip(new_arg_names, args_list)}
            
        # Convert dict of sets in list of dicts, where each dict value is one 
        #   arg val.
        iterable_kwargs = pd.DataFrame(iterable_args_dict).to_dict('records')
        
        return iterable_kwargs
    
    # create iterator of arguments for the solving routine (e.g. if there are
    #   different arguments for each parameter set)        
    iterable_kwargs = create_iterator([solved_quantities, bounds, x_init,
                                       parameters, solver_kwargs, other_kwargs])
    
    # Iterate through all parameter sets and solve the self-consistency equations
    fitted_values_final_loss = np.array([solver(equation_func = function,
                                                ms_function = ms_function,
                                                **i_kwargs)
                                          for i_kwargs in tqdm(iterable_kwargs,
                                                               position = 0,
                                                               leave = True)])
    
    # Convert array to dataframe. Each column solved quantity, each row is the 
    # solves sces for a given parameter set
    fitted_values_df = pd.DataFrame(fitted_values_final_loss, columns = solved_quantities + ['loss'])

    # combine parameter sets and solved equations into a single dataframe
    
    if isinstance(parameters, dict):
        
        df = pd.concat([pd.DataFrame([parameters]), fitted_values_df], axis = 1)
        
    else : 
    
        df = pd.concat([pd.DataFrame(parameters), fitted_values_df], axis = 1)
    
    return df
        

# %%

def solve_for_multistability(y, multistability_equation_func):
    
    '''
    
    Solve the stability-related simultaneous equations (<(dNde)^2> and <(dRde)^2>,
    not the stability condition)

    Parameters
    ----------
    y : dict
        Parameter set values + values for the solved self-consistency equations.
    multistability_equation_func : str
        Name of the system of self-consistency equations you want to solve, aka
        the model you're solving.
        The options are 'self-limiting, rho' (Blumenthal et al., 2024),
        'self-limiting, yc c' (our model), 'self-limiting, g cg' (consumption
        is coupled to growth), 'externally supplied' (chemostat-style system
        with non-reciprocol growth and consumption rates)
                                                      
    Returns
    -------
    sol : np.ndarray
        Solved values for <(dNde)^2> and <(dRde)^2>, and the loss function.

    '''
    
    # bounds and initial values for <(dNde)^2> and <(dRde)^2>
    bounds = ([-1e15, -1e15], [1e15, 1e15])
    x_init = [0, 0]
    
    # find the right set of equations for the model
    match multistability_equation_func:
        
        case 'self-limiting, rho':
            
            fun = slr.multistability_equations
            ls_kwarg_names = ['rho', 'gamma', 'sigma_c', 'sigma_g',
                              'phi_N', 'phi_R', 'v_N', 'chi_R']
            
        case 'self-limiting, yc c':
            
            fun = slgcM.multistability_equations
            ls_kwarg_names = ['M', 'gamma', 'sigma_c', 'sigma_y', 'mu_c', 'mu_y',
                              'phi_N', 'phi_R', 'v_N', 'chi_R']
            
        case 'self-limiting, g cg':
            
            fun = slcg.multistability_equations_inf
            ls_kwarg_names = ['gamma', 'sigma_c', 'sigma_g', 'mu_c', 'mu_g',
                              'phi_N', 'phi_R', 'chi_R', 'v_N']
            
        case 'externally supplied':
            
            fun = es.multistability_equations
            ls_kwarg_names = ['rho', 'gamma', 'mu_c', 'sigma_c', 'sigma_g', 'mu_K',
                              'mu_D', 'sigma_D', 'phi_N', 'N_mean', 'q_N', 'v_N',
                              'chi_R']
    
    # create kwargs (parameters) for the equation being solved for 
    ls_kwargs = {key : y[key] for key in ls_kwarg_names}
    
    # Solve the multistability equations
    sol = solve_equations_least_squares(fun, ['dNde', 'dRde'],  bounds, x_init,
                                        ls_kwargs)
    
    return sol

# %%

def solve_equations_basinhopping(equation_func, solved_quantities, bounds, x_init,
                                 ls_kwargs, solver_kwargs, other_kwargs,
                                 ms_function = None, return_all = False):
    
    '''
    
     Globally solve the self-consistency equations using
     scipy.optimize.basinhopping(). To locally solve the equations between each 
     "hop", the routine calls scipy.optimize.least_squares().

    Parameters
    ----------
    equation_func : function
        The system of equations being solved for.
    solved_quantities : list of str
        The names of the quantities being solved for. Usually the self-consistency
        equations.
    bounds : list containing 2 tuples
        The lower and upper bounds of the quantities being solved for.
    x_init : list of np.ndarray, or list of lists
        The initial values of the solved_quantities.
    ls_kwargs : dict
        Values that are not being solved for. Usually the statistical properties 
        of the distributions of model parameters (e.g. mean consumption rate)
    solver_kwargs : dict
        Options for the least-squares solver, which is called by both the 
        'basin-hopping' and 'least squares' routine.  .                                                  
    other_kwargs : dict
        Options for the basin-hopping solver.
    ms_function : function, optional
        The stability condition function. The default is None.
    return_all : Bool, optional
        Whether the all the attributes of the optimised results (True) or only 
        the solved values and loss function should be returned (False). 
        The default is False.

    Returns
    -------
    returned_values : np.ndarray
        The solved system of equations.

    '''
    
    def minimise_ls(fun, x0, *args, **kwargs):
        
        '''
        
        Customised routine using scipy.optimize.least_squares().
        The least_squares output is customised to a form that can be used by
        the basin-hopping routine (which is not possible in its native form).
        It also rejects solves that blow up.

        Parameters
        ----------
        fun : function
            The function to be minimised.
        x0 : TYPE
            Initial values for the solved quantities.
        *args : args
            Solver arguments.
        **kwargs : kwargs
            Solver kwargs that can potentially be used by the solver.

        Returns
        -------
        sol : OptimizeResult object + modifications
            Local solve of the function.

        '''
        
        # extract args for least_squares()
        least_squares_kwargs = filter_ls_kwargs(kwargs)
        
        # Calculate the loss function for the initial values
        
        # Call the solver for 1 iteration (so no solve occurs)
        sol_init = least_squares(fun, x0, max_nfev = 1, **least_squares_kwargs)
        # Calculate the loss function from the residuals (standard form of 
        #   least_squares() OptimizeResult.fun)
        sol_init.fun = np.sum(sol_init.fun**2)    
        
        # Locally solve the equations 
        
        sol = least_squares(fun, x0, **least_squares_kwargs)
        # Calculate loss + turn loss function into a form that basinhopping()
        #   can use
        sol.fun = np.sum(sol.fun**2)
        
        # Keep local solve if the loss function hasn't blown up, otherwise
        #   reject and keep initial values
        if sol.fun > 1e-2:
            
            return sol_init
        
        else:
        
            return sol

    # Construct the function to be minimised
    if ms_function:
        
        fun = lambda x : equation_func(**{key: val for key, val in 
                                          zip(solved_quantities, x)}, **ls_kwargs) \
                         + ms_function(**{key: val for key, val in 
                                          zip(solved_quantities, x)}, **ls_kwargs)
    
    else:

        fun = lambda x : equation_func(**{key: val for key, val in 
                                          zip(solved_quantities, x)}, **ls_kwargs)
    
    # Define the step constrainer and acceptance test for the basin-hopping routine 
    bounded_step = Take_Bounded_Step(bounds[0], bounds[1])
    accept_bounded_step = Accept_within_Bounds(bounds[1], bounds[0])
    
    # Globally minimise the function
    fitted_values = basinhopping(fun, x0 = x_init,
                                 minimizer_kwargs = {"method" : minimise_ls,
                                                     "bounds" : bounds,
                                                     "options" : solver_kwargs},
                                 take_step = bounded_step,
                                 accept_test = accept_bounded_step,
                                 callback = decent_solve,
                                 **other_kwargs)

    if return_all is True:
        
        # return all attributes of the OptimizeResult object
        returned_values = fitted_values
        
    else: 
        
        # returned solved values + loss function
        returned_values = np.append(fitted_values.x,
                                    np.log10(np.sum(fitted_values.fun**2)))
    
    return returned_values

class Take_Bounded_Step(object):
    
    '''
    
    Stop the basinhopping routine from taking too large a step (which could cause
    the loss to blow up).
    
    '''

    def __init__(self, xmin, xmax, stepsize=0.1):
        
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):
        
        #breakpoint()
        
        min_step = np.maximum(self.xmin - x, -self.stepsize)
        max_step = np.minimum(self.xmax - x, self.stepsize)

        random_step = np.random.uniform(low=min_step, high=max_step, size=x.shape)
        xnew = x + random_step

        return xnew

class Accept_within_Bounds:
    
    '''
    
    Only accept solved values within certain bounds (basin-hopping does not
    have a native bounds routine).
    
    '''
    
    def __init__(self, xmax, xmin):
        
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
        
    def __call__(self, **kwargs):
        
        #breakpoint()
        
        x = kwargs["x_new"]
        
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        
        return tmax and tmin
    
def decent_solve(x, f, accept):
    
    '''
    
    Stop the basinhopping routine when the solve is good enough (aka when the
    loss function is small enough)

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    f : float
        Value of the loss function.
    accept : TYPE
        DESCRIPTION.

    Returns
    -------
    bool
        Whether the routine should stop.

    '''
    
    # Stop solver if the loss function is less than 10^(-25)
    if np.log10(np.abs(f)) < -25:
        
        return True
    
def filter_ls_kwargs(kwargs):
    
    '''
    
    Extract the correct kwargs for the least_squares routine

    Parameters
    ----------
    kwargs : dict
        All kwargs.

    Returns
    -------
    dict
        Correct kwargs.

    '''
   
    valid_args = getfullargspec(least_squares)[0]
    collection_types = (list, dict, tuple, np.ndarray)
    
    return {arg_name : kwargs[arg_name]
            for arg_name in valid_args
            if arg_name in kwargs
            and isinstance(kwargs[arg_name], collection_types)
            and any(kwargs[arg_name])}
    
# %%

def solve_equations_least_squares(equation_func, solved_quantities, bounds, x_init,
                                  ls_kwargs, solver_kwargs = {}, other_kwargs = {},
                                  ms_function = None, return_all = False):
    
    '''
    
    Locally solve the self-consistency equations using 
    scipy.optimize.least_squares. 
    
    Parameters
    ----------
    See solve_equations_basinhopping().

    Returns
    -------
    returned_values : np.ndarray
        The solved system of equations.

    '''
    
    # Construct function to minimise
    if ms_function:
        
        fun = lambda x : equation_func(**{key: val for key, val in 
                                          zip(solved_quantities, x)}, **ls_kwargs) \
                         + ms_function(**{key: val for key, val in 
                                          zip(solved_quantities, x)}, **ls_kwargs)
    
    else:

        fun = lambda x : equation_func(**{key: val for key, val in 
                                          zip(solved_quantities, x)}, **ls_kwargs)
    
    # Solve equations
    fitted_values = least_squares(fun, x_init, bounds = bounds,
                                  max_nfev = 10000, **solver_kwargs)
    
    # Return the whole object or just the solved values + loss function
    if return_all is True:
        
        returned_values = fitted_values
        
    else: 
        
        returned_values = np.append(fitted_values.x, np.log10(np.sum(fitted_values.fun**2)))
    
    return returned_values
