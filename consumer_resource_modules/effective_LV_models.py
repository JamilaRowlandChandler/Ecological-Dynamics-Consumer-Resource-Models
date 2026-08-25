# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 14:13:06 2025

@author: jamil

"""

# %%

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp
from scipy.stats import pearsonr
from typing import Union, Literal, TypedDict

from models import SL_CRM, ES_CRM
from parameters import ParametersInterface
from differential_equations import DifferentialEquationsInterface_ELV
    
# %%

class eLVMethods(DifferentialEquationsInterface_ELV):
   
    def suriving_resources(self,
                           CRM_community : Union["SL_CRM", "ES_CRM"],
                           cavity_phi_R : float):
        
        #Identify which resources have survied
       #   (by 1. finding an extinction threshold where the fraction of surviving
       #   resources is closet to the cavity value of phi_R, then 2. selecting the 
       #   resources above this threshold)
       
       resource_abundances = CRM_community.ODE_sols[0].y[CRM_community.no_species :, -1]
       
       potential_ext_thresh = np.linspace(-8, -2, 200)
       
       extinct_thresh = \
       10**(potential_ext_thresh[np.abs(np.array([np.sum(resource_abundances > 10**(e_t))/self.no_resources
                                                  for e_t in potential_ext_thresh]) - cavity_phi_R).argmin()])
       
       self.phi_R = np.sum(resource_abundances > extinct_thresh)/self.no_resources
       
       self.resource_pa = np.int64(resource_abundances > extinct_thresh).astype(np.float64)
        
    def calculate_interaction_stats(self):
        
        '''
        
        Determine the statistics, such as the mean, variance and correlations,
        in species growth and interaction coefficients. 
        
        '''
        
        ### growth rates ###
        
        self.mu_r = np.mean(self.r)
        self.sigma_r = np.std(self.r)
        
        ### interactions ###
        
        self_inhibition = np.diagonal(self.interaction_matrix)
        
        inter_species_interactions = self.interaction_matrix[~np.eye(self.interaction_matrix.shape[0],
                                                                     dtype=bool)].reshape((self.no_species,
                                                                                           self.no_species - 1))
        
        self.mu_Aii = np.mean(self_inhibition)
        self.sigma_Aii = np.std(self_inhibition)
        self.mu_Aij = np.mean(inter_species_interactions)
        self.sigma_Aij = np.std(inter_species_interactions) 
        
        def diagonal_correlation(matrix):
            
            i_upper, j_upper = np.triu_indices_from(matrix, k=1)

            upper_elements = matrix[i_upper, j_upper]
            lower_elements = matrix[j_upper, i_upper]

            correlation = pearsonr(upper_elements, lower_elements)[0]
            
            return correlation
        
        def row_correlation(matrix):
            
            S = matrix.shape[0]
            
            mask = ~np.eye(S, dtype=bool)
            B = matrix[mask].reshape(S, S - 1)
            
            p, q = np.triu_indices(S - 1, k=1)
            
            correlation = pearsonr(B[:, p].ravel(), B[:, q].ravel())[0]
            
            return correlation
        
        def column_correlation(matrix):
            
            S = matrix.shape[0]
            
            mask = ~np.eye(S, dtype=bool)
            C = matrix.T[mask].reshape(S, S - 1)
            p, q = np.triu_indices(S - 1, k=1)
            
            correlation = pearsonr(C[:, p].ravel(), C[:, q].ravel())[0]
            
            return correlation
            
        def one_index_correlation(matrix):
            
            S = matrix.shape[0]
            
            mask = ~np.eye(S, dtype=bool)
            B = matrix[mask].reshape(S, S - 1)       # B[i, :] = A[i, j] for j != i  (row entries)
            C = matrix.T[mask].reshape(S, S - 1)
            
            offdiag_S1 = ~np.eye(S - 1, dtype=bool)
            
            cr1 = np.broadcast_to(B[:, :, None], (S, S - 1, S - 1))[:, offdiag_S1]
            cr2 = np.broadcast_to(C[:, None, :], (S, S - 1, S - 1))[:, offdiag_S1]
            
            correlation = pearsonr(cr1.ravel(), cr2.ravel())[0]
            
            return correlation
            
        self.rho_D = diagonal_correlation(self.interaction_matrix)
        self.rho_R = row_correlation(self.interaction_matrix)
        self.rho_C = column_correlation(self.interaction_matrix)
        self.rho_1idx = one_index_correlation(self.interaction_matrix)
         
        total_interact_per_species = np.sum(inter_species_interactions, axis = 1)
        
        self.interaction_statistics = dict(Aii = self_inhibition,
                                           sum_j_Aij = total_interact_per_species,
                                           mu_Aij_tot = self.mu_Aij * \
                                               np.float64(self.no_species),
                                           sigma_Aij_tot = self.sigma_Aij * \
                                               np.sqrt(np.float64(self.no_species)))
                                           
    #####################################################
                                          
    def simulation(self,
                   t_end : float,
                   initial_abundances : npt.NDArray,
                   assign : bool = True):
        
        def LV_dynamics(t, species, r, A):
            
            # change in consumer abundances over time
            dNdt = species * (r - np.sum(A * species, axis = 1))
        
            return dNdt + 1e-8
        
        def LV_jacobian(t, species, r, A):

            # growth term: r_i - sum_j A_ij * N_j  (shared with LV_dynamics)
            growth = r - np.sum(A * species, axis=1)
    
            # J = diag(growth) - diag(species) @ A
            J = np.diag(growth) - species[:, None] * A
    
            return J
            
        def unbounded_growth(t, var, *args):
            
            
            # if any species or resource abundances are greater than some threshold
            # or if any species abundances are less than or equal to 0  
            if np.any(np.log10(np.abs(var) + 1e-20) > 4) or np.isnan(np.log10(np.abs(var) + 1e-20)).any():
                
                return 0 # when the returned value of an event function is 0, the ode 
                            #solver terminates.
            
            else: 
                
                return 1 # the ode solver continues because the returned value is non-zero.
            
        return solve_ivp(LV_dynamics,
                         [0, t_end],
                         initial_abundances, 
                         args = (self.r, self.interaction_matrix),
                         method = 'LSODA',
                         rtol = 1e-7, atol = 1e-9,
                         t_eval = np.linspace(0, t_end, 200),
                         jac = LV_jacobian,
                         events = unbounded_growth)

# %%

class eLV_SL(eLVMethods, ParametersInterface):
    
    def __init__(self,
                 no_species : Union[float, None] = None,
                 no_resources : Union[float, None] = None):
        
        self.no_species = no_species
        self.no_resources = no_resources
        
        self.consumption = None
        self.growth = None
        self.d = None
        
        self.b = None
        
    def elv_from_crm(self,
                     CRM_community : "SL_CRM",
                     cavity_phi_R : Union[float, None] = None):
        
        
        self.no_species = CRM_community.no_species
        self.no_resources = CRM_community.no_resources
        
        self.growth_consumption_rates("user-supplied",
                                      mu_c = CRM_community.mu_c,
                                      sigma_c = CRM_community.sigma_c,
                                      mu_g = CRM_community.mu_g, 
                                      sigma_g = CRM_community.sigma_g,
                                      rho = getattr(CRM_community, "rho", None),
                                      consumption = CRM_community.consumption,
                                      growth = CRM_community.growth)
        
        self.model_specific_rates(death_method = 'user-supplied',
                                  death_args = dict(d = CRM_community.d),
                                  resource_growth_method = 'user-supplied',
                                  resource_growth_args = dict(b = CRM_community.b))
        
        # extract only surviving resources that species can interact through
        
        if cavity_phi_R and isinstance(CRM_community, SL_CRM):
            
            self.suriving_resources(CRM_community, cavity_phi_R) 
        
    def model_specific_rates(self,
                             death_method : 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             death_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'d' : float}),
                                                TypedDict('user-supplied', {'d' : npt.NDArray})]
                             = {'d' : 1},
                             resource_growth_method : 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             resource_growth_args : 
                                 Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                       TypedDict('constant', {'b' : float}),
                                       TypedDict('user-supplied', {'b' : npt.NDArray})]
                                 = {'b' : 1}):
        
        '''
        
        Generate parameters specific to the CRM with self-limiting resource 
        dynamics - consumer death rates and intrinsic resource growth rates


        Parameters
        ----------
        death_method : str
            Method used to generate death rates. Options are:
                'normal' : normally distributed parameters
                'constant' : death rates are fixed
                'user-supplied' : supply your own death rates
        death_args : dict
            Arguments for death_method.
            If 'normal', first argument is the mean, second is the stand deviation
            e.g., {'mu': mean, 'sigma' : standard deviation}
            If 'constant', the key is the parameter name, argument is the fixed value
            e.g., {'d' : val}
            If 'used-supplied', argument is the array of death rates 
            e.g., {'d' : array_of_vals}
        resource_growth_method : str
            Method used to generate intrinsic resource growth rates. Options are
            the same as death_method, but named 'b' rather than 'd'
        resource_growth_args : dict
            Arguments for resource_growth_method. Options are the same as 
            death_method args.

        Returns
        -------
        None.

        '''
        
        # labels used to assign parameters as object attributes
        p_labels = ['d', 'b']
        
        # dimensions for death rates and intrinsic growth rates
        dims_list = [(self.no_species, ), (self.no_resources, )]
        
        # generate parameters used the other_parameter_methods method
        for p_method, p_args, p_label, dims in \
            zip([death_method, resource_growth_method],
                [death_args, resource_growth_args],
                p_labels, dims_list):
                
                self.other_parameter_methods(p_method, p_args, p_label, dims)
    
    def generate_elv_parameters(self):
        
        if hasattr(self, "consumption") is True:
        
            if hasattr(self, "resource_pa") is True:
                
                self.r = np.sum((self.growth * self.b * self.resource_pa),
                                axis = 1) - self.d
                
                self.interaction_matrix = np.dot(self.growth,
                                                 (self.consumption.T * self.resource_pa).T)
            
            else:
                
                self.r = np.sum((self.growth * self.b), axis = 1) - self.d
                
                self.interaction_matrix = np.dot(self.growth,
                                                 (self.consumption.T).T)
                
        else:
            
            raise Exception("You need to generate the CRM parameters before " + \
                            "you can call this method. Please use elv_from_crm() " + \
                            "or growth_consumption_rates() + model_specific_rates() " + \
                            "do to this.")

# %%

class eLV_ES(eLVMethods, ParametersInterface):
    
    def __init__(self,
                 no_species : Union[float, None] = None,
                 no_resources : Union[float, None] = None):
        
        self.no_species = no_species
        self.no_resources = no_resources
        
        self.consumption = None
        self.growth = None
        self.d = None
        
        self.b = None
        self.o = None
    
    def elv_from_crm(self,
                     CRM_community : "ES_CRM"):
        
        self.no_species = CRM_community.no_species
        self.no_resources = CRM_community.no_resources
        
        self.growth_consumption_rates("user-supplied",
                                      mu_c = CRM_community.mu_c,
                                      sigma_c = CRM_community.sigma_c,
                                      mu_g = CRM_community.mu_g, 
                                      sigma_g = CRM_community.sigma_g,
                                      rho = getattr(CRM_community, "rho", None),
                                      consumption = CRM_community.consumption,
                                      growth = CRM_community.growth)
        
        self.model_specific_rates(death_method = 'user-supplied',
                                  death_args = dict(d = CRM_community.d),
                                  influx_method = 'user-supplied',
                                  influx_args = dict(b = CRM_community.b),
                                  outflux_method = 'user-supplied',
                                  outflux_args = dict(o = CRM_community.o))
        
    def model_specific_rates(self, 
                             death_method : 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             death_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'d' : float}),
                                                TypedDict('user-supplied', {'d' : npt.NDArray})]
                             = {'d' : 1},
                             influx_method: 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             influx_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'b' : float}),
                                                TypedDict('user-supplied', {'b' : npt.NDArray})]
                             = {'b' : 1},
                             outflux_method: 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             outflux_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'o' : float}),
                                                TypedDict('user-supplied', {'o' : npt.NDArray})]
                             = {'o' : 1}):
        
        '''
        
        Generate parameters specific to the CRM with self-limiting resource 
        dynamics - consumer death rates and intrinsic resource growth rates


        Parameters
        ----------
        death_method : str
            Method used to generate death rates. Options are:
                'normal' : normally distributed parameters
                'constant' : death rates are fixed
                'user-supplied' : supply your own death rates
        death_args : dict
            Arguments for death_method.
            If 'normal', first argument is the mean, second is the stand deviation
            e.g., {'mu': mean, 'sigma' : standard deviation}
            If 'constant', the key is the parameter name, argument is the fixed value
            e.g., {'d' : val}
            If 'used-supplied', argument is the array of death rates 
            e.g., {'d' : array_of_vals}
        influx_method : str
            Method used to generate intrinsic resource influx rates. Options are
            the same as death_method, but named 'b' rather than 'd'
        influx_args : dict
            Arguments for influx_method. Options are the same as 
            death_method args.
        outflux_method : str
            Method used to generate intrinsic resource outflux rates. Options are
            the same as death_method, but named 'o' rather than 'd'.
        outflux_args : dict
            Arguments for outflux_method. Options are the same as 
            death_method args.
        Returns
        -------
        None.

        '''
        
        # labels used to assign parameters as object attributes
        p_labels = ['d', 'b', 'o']
        
        # dimensions for death rates and intrinsic growth rates
        dims_list = [(self.no_species, ), (self.no_resources, ),
                     (self.no_resources, )]
        
        # generate parameters used the other_parameter_methods method
        for p_method, p_args, p_label, dims in \
            zip([death_method, influx_method, outflux_method],
                [death_args, influx_args, outflux_args],
                p_labels, dims_list):
                
                self.other_parameter_methods(p_method, p_args, p_label, dims)
        
    def generate_elv_parameters(self):
        
        if hasattr(self, "consumption") is True:
        
            self.r = \
                np.sum((self.growth * (self.b / self.o)), axis = 1) - self.d
        
            self.interaction_matrix = \
                np.dot(self.growth * (self.b / self.o**2), (self.consumption.T).T)
                
        else:
            
            raise Exception("You need to generate the CRM parameters before " + \
                            "you can call this method. Please use elv_from_crm() " + \
                            "or growth_consumption_rates() + model_specific_rates() " + \
                            "do to this.")
        
# %%

def Effective_LV_Model(model : Literal["Self-limiting resource supply",
                                       "Externally-supplied resources"],
                       **kwargs):
    
    '''
    
    Wrapper for different consumer resource model classes

    Parameters
    ----------
    model : str
        Type of consumer resource model. Options are:
            "Self-limiting resource supply" - resources grow logistically
            "Self-limiting resource supply, self-inhibition" - same as 
            "Self-limiting resource supply", but with direct consumer self-inhibition
            "Externally-supplied resources" - chemostat-style resource dynamics
            (constant influx + dilution)
    no_species : int
        species pool size
    no_resources : int
        resource pool size

    Raises
    ------
    Exception
        If a non-existent model is selected.

    Returns
    -------
    instance : object of some consumer-resource model class
        Instance of some consumer-resource model class.

    '''
    
    match model:
        
        case "Self-limiting resource supply":
            
            instance = eLV_SL(**kwargs)
            
        case "Externally-supplied resources":
            
            instance = eLV_ES(**kwargs)
            
        case _:
            
            raise Exception('You have not selected an exisiting model.\n' + \
                  'Please chose from either "Self-limiting resource supply"' + \
                      '"Self-limiting resource supply, self-inhibition"' + \
                      ' or "Externally-supplied resources"')
    return instance
        
# %%

class gLV(eLVMethods, ParametersInterface):
    
    def __init__(self,
                 no_species : float):
        
        self.no_species = no_species
        
    def model_specific_rates(self,
                             growth_method : 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             growth_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'r' : float}),
                                                TypedDict('user-supplied', {'r' : npt.NDArray})]
                             = {'r' : 1},
                             interaction_method : 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'normal',
                             interaction_args : Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                                TypedDict('constant', {'Aij' : float}),
                                                TypedDict('user-supplied', {'Aij' : npt.NDArray})]
                             = {'Aij' : {'mu' : 0, 'sigma' : 1}},
                             self_inhibition_method : 
                                 Literal['normal', 'constant', 'user-supplied'] 
                                 = 'constant',
                             self_inhibition_args : 
                                 Union[TypedDict('normal', {'mu' : float, 'sigma' : float}),
                                       TypedDict('constant', {'Aii' : float}),
                                       TypedDict('user-supplied', {'Aii' : npt.NDArray})]
                                 = {'Aii' : 1}):
        
        '''
        
        Generate parameters specific to the CRM with self-limiting resource 
        dynamics - consumer death rates and intrinsic resource growth rates

        Returns
        -------
        None.

        '''
        
        self.other_parameter_methods(growth_method,
                                     growth_args,
                                     'r',
                                     (self.no_species, ))
        
        
        
        self.other_parameter_methods(self_inhibition_method,
                                     self_inhibition_args,
                                     'Aii',
                                     (self.no_species, ))
        
        if interaction_args.get('rhos') is None:
            
            self.other_parameter_methods(interaction_method,
                                         interaction_args,
                                         'Aij',
                                         (self.no_species, self.no_species))
            
        else:
            
            self.__correlated_interactions(interaction_args['mu'],
                                           interaction_args['sigma'],
                                           **interaction_args['rhos'])
            
        self.interaction_matrix = self.Aij
        np.fill_diagonal(self.interaction_matrix, self.Aii)
        
    def __correlated_interactions(self,
                                  mu,
                                  sigma,
                                  rho_D=0.0,
                                  rho_R=0.0,
                                  rho_C=0.0,
                                  rho_1idx=0.0,
                                  rng=None):
        """
        Following Castedo, Holmes, Baron & Galla (arXiv:2409.12751), Sec. II.C.
        z_ij = lambda_i + kappa_j + w_ij, with (lambda_i, kappa_i) and (w_ij, w_ji)
        each drawn as correlated Gaussian pairs (unit variance), then rescaled by sigma.
        """
        
        rng = np.random.default_rng() if rng is None else rng
    
        self.__feasibility_reason(rho_R,
                                  rho_C,
                                  rho_1idx,
                                  rho_D)
        
        # --- species-level (lambda, kappa) pairs, unit variance, Eq. 4 ---
        cov_lk = np.array([[rho_R, rho_1idx],
                            [rho_1idx, rho_C]])
        L_lk = self.__psd_sqrt(cov_lk)
            
        lk = rng.standard_normal((self.no_species, 2)) @ L_lk.T
        lam, kap = lk[:, 0], lk[:, 1]
    
        # --- edge-level (w_ij, w_ji) pairs, unit variance, Eq. 5 ---
        resid_var = 1 - rho_R - rho_C
        resid_cov = rho_D - 2 * rho_1idx
        cov_w = np.array([[resid_var, resid_cov],
                           [resid_cov, resid_var]])
        L_w = self.__psd_sqrt(cov_w)
        
        iu, ju = np.triu_indices(self.no_species, k=1)
        wpairs = rng.standard_normal((iu.size, 2)) @ L_w.T
        W = np.zeros((self.no_species, self.no_species))
        W[iu, ju], W[ju, iu] = wpairs[:, 0], wpairs[:, 1]
    
        # --- assemble, unit-variance Z, Eq. 6 ---
        Z = lam[:, None] + kap[None, :] + W
        
        Aij = mu + sigma * Z
        
        #####
        
        self.Aij = Aij
        
        self.corr_violate = getattr(self, "corr_violate", False)
        
    def __feasibility_reason(self,
                             rho_R,
                             rho_C,
                             rho_1idx,
                             rho_D,
                             tol=0.05):
        """
        Checks the four analytic feasibility conditions in order and returns the
        message for the first one violated, or None if all are satisfied.
        """
        conditions = [
            (np.round(rho_R, 2) < 0 or np.round(rho_C, 2) < 0 + tol,
             "rho_row and rho_col must be >= 0."),
            (np.round(np.abs(rho_1idx), 2) + tol > np.round(np.sqrt(rho_R * rho_C), 2),
             "|rho_cross| must be <= sqrt(rho_row*rho_col)."),
            (np.round(rho_D - 2 * rho_1idx, 2) < 0 + tol,
             "rho_recip must be >= 2*rho_cross."),
            (np.round(rho_R + rho_C + (rho_D - 2 * rho_1idx), 2) > 1 + tol,
             "rho_row + rho_col + (rho_recip - 2*rho_cross) must be <= 1."),
        ]
        
        violation = next((msg for cond, msg in conditions if cond), None)
        
        if violation is not None:
            
            self.corr_violate = True
    
    def __psd_sqrt(self,
                   cov,
                   tol=1e-2):
        
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        if np.any(eigvals < -tol):
            
            self.corr_violate = True
            
        eigvals = np.clip(eigvals, 0, None)
        
        return eigvecs @ np.diag(np.sqrt(eigvals))
                     
       
