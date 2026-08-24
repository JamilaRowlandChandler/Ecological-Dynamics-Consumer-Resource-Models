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
from differential_equations import DifferentialEquationsInterface_LV
from community_level_properties import CommunityPropertiesInterface_LV
    
# %%

class eLVMethods(DifferentialEquationsInterface_LV,
                 CommunityPropertiesInterface_LV):
   
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

        S = self.no_species
        A = self.interaction_matrix

        ### growth rates ###

        self.mu_r = np.mean(self.r)
        self.sigma_r = np.std(self.r)

        ### interaction moments ###

        self_inhibition = np.diagonal(A)

        mask = ~np.eye(S, dtype=bool)

        inter_species_interactions = A[mask].reshape(S, S - 1)

        self.mu_Aii = np.mean(self_inhibition)
        self.sigma_Aii = np.std(self_inhibition)
        self.mu_Aij = np.mean(inter_species_interactions)
        self.sigma_Aij = np.std(inter_species_interactions)

        ### within-A correlations ###

        # shared arrays, computed once
        B = inter_species_interactions                  # B[i, :] = A[i, j] for j != i
        C = A.T[mask].reshape(S, S - 1)                # C[j, :] = A[i, j] for i != j
        p, q = np.triu_indices(S - 1, k=1)

        def diagonal_correlation(A):

            iu, ju = np.triu_indices(S, k=1)

            return pearsonr(A[iu, ju], A[ju, iu])[0]

        def row_correlation(B, p, q):

            return pearsonr(B[:, p].ravel(), B[:, q].ravel())[0]

        def column_correlation(C, p, q):

            return pearsonr(C[:, p].ravel(), C[:, q].ravel())[0]

        def one_index_correlation(B, C):

            offdiag_S1 = ~np.eye(S - 1, dtype=bool)

            cr1 = np.broadcast_to(B[:, :, None], (S, S - 1, S - 1))[:, offdiag_S1]
            cr2 = np.broadcast_to(C[:, None, :], (S, S - 1, S - 1))[:, offdiag_S1]

            return pearsonr(cr1.ravel(), cr2.ravel())[0]

        def growth_interaction_correlation(r, B):

            r_repeated = np.repeat(r, S - 1)

            return pearsonr(r_repeated, B.ravel())[0]

        def self_inhibition_interaction_correlation(Aii, B):

            Aii_repeated = np.repeat(Aii, S - 1)

            return pearsonr(Aii_repeated, B.ravel())[0]

        def growth_self_inhibition_correlation(r, Aii):

            return pearsonr(r, Aii)[0]

        self.rho_D = diagonal_correlation(A)
        self.rho_R = row_correlation(B, p, q)
        self.rho_C = column_correlation(C, p, q)
        self.rho_1idx = one_index_correlation(B, C)

        self.rho_r_Aij = growth_interaction_correlation(self.r, B)
        self.rho_Aii_Aij = self_inhibition_interaction_correlation(self_inhibition, B)
        self.rho_r_Aii = growth_self_inhibition_correlation(self.r, self_inhibition)

        ### summary statistics ###

        total_interact_per_species = np.sum(inter_species_interactions, axis=1)

        self.interaction_statistics = dict(Aii=self_inhibition,
                                           sum_j_Aij=total_interact_per_species,
                                           mu_Aij_tot=self.mu_Aij * \
                                               np.float64(self.no_species),
                                           sigma_Aij_tot=self.sigma_Aij * \
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

    def decompose_eLV(self):

        '''

        Decompose the off-diagonal entries of interaction_matrix into the
        additive components of the latent factor model:

            Z_ij = lam_i + kap_j + w_ij

        lam_i = mean of row i (row effect)
        kap_j = mean of column j after removing row effects (column effect)
        w_ij  = residual (edge-specific)

        Returns
        -------
        grand_mean : float
            Mean of all off-diagonal entries.
        lam : npt.NDArray
            Row effects, shape (no_species,).
        kap : npt.NDArray
            Column effects, shape (no_species,).
        W : npt.NDArray
            Edge-specific residuals, shape (no_species, no_species) (diagonal
            set to 0).

        '''

        A = self.interaction_matrix
        S = A.shape[0]
        mask = ~np.eye(S, dtype=bool)

        # work with off-diagonal entries only
        A_work = A.copy()
        grand_mean = A_work[mask].mean()

        # row effects: mean of each row's off-diagonal entries
        A_offdiag = A_work.copy()
        np.fill_diagonal(A_offdiag, np.nan)
        row_means = np.nanmean(A_offdiag, axis=1)
        lam = row_means - grand_mean

        # column effects: mean of each column after removing row effects
        residual_after_row = A_work - lam[:, None] - grand_mean
        np.fill_diagonal(residual_after_row, np.nan)
        col_means = np.nanmean(residual_after_row, axis=0)
        kap = col_means

        # edge-specific residual
        W = A_work - grand_mean - lam[:, None] - kap[None, :]
        np.fill_diagonal(W, 0)

        return grand_mean, lam, kap, W

    def reconstruct_ablated(self,
                            grand_mean, lam, kap, W,
                            keep_row=True,
                            keep_col=True,
                            keep_residual=True,
                            keep_r_corr=True,
                            keep_Aii_corr=True):

        '''

        Reconstruct an interaction matrix (and growth-rate vector) from the
        components returned by decompose_eLV(), selectively zeroing out
        specific structural features of this community's own
        interaction_matrix/r.

        Zeroing a component removes ONLY that component's contribution to
        the correlation structure, without collateral damage to others.

        Parameters
        ----------
        grand_mean, lam, kap, W : as returned by decompose_eLV().
        keep_row : bool, optional
            Keep the row effect (drives rho_R). The default is True.
        keep_col : bool, optional
            Keep the column effect (drives rho_C). The default is True.
        keep_residual : bool, optional
            Keep the edge-specific residual (carries the reciprocal
            correlation, rho_D, in its symmetric part). The default is True.
        keep_r_corr : bool, optional
            Keep r's correlation with the interaction matrix (if False,
            shuffle r so any such correlation is destroyed). The default is
            True.
        keep_Aii_corr : bool, optional
            Keep Aii's correlation with the interaction matrix (if False,
            shuffle the diagonal so any such correlation is destroyed). The
            default is True.

        Returns
        -------
        A_new : npt.NDArray
            Reconstructed interaction matrix.
        r_new : npt.NDArray
            Growth rates (this community's own r), shuffled if keep_r_corr
            is False.

        '''

        S = len(lam)

        # row effect
        row_component = lam[:, None] if keep_row else np.zeros((S, 1))

        # column effect
        col_component = kap[None, :] if keep_col else np.zeros((1, S))

        # residual (carries reciprocal correlation in its symmetric part)
        edge_component = W if keep_residual else np.zeros((S, S))

        # reconstruct off-diagonal
        A_new = grand_mean + row_component + col_component + edge_component

        # diagonal
        diag = np.diagonal(self.interaction_matrix)

        if keep_Aii_corr:
            np.fill_diagonal(A_new, diag)
        else:
            diag_shuffled = diag.copy()
            np.random.shuffle(diag_shuffled)
            np.fill_diagonal(A_new, diag_shuffled)

        # growth rates
        if keep_r_corr:
            r_new = self.r.copy()
        else:
            r_new = self.r.copy()
            np.random.shuffle(r_new)

        return A_new, r_new

    def ablate_moments(self,
                       ablate_mu_r=False,
                       ablate_mu_Aij=False,
                       ablate_sigma_r=False,
                       ablate_sigma_Aij=False,
                       ablate_sigma_Aii=False):

        '''

        Ablate the mean or variance of r/A_ij/A_ii, independent of the
        correlation ablations in reconstruct_ablated().

        Ablating a mean (ablate_mu_r/ablate_mu_Aij) shifts that parameter so
        its mean equals the community's own mean self-inhibition (mean of
        the diagonal of interaction_matrix), while leaving its variance and
        any correlation it carries with other parameters unchanged - this
        is a pure additive shift, and corr(x + c, y) = corr(x, y) for any
        constant c.

        Ablating a variance (ablate_sigma_r/ablate_sigma_Aij/
        ablate_sigma_Aii) collapses that parameter to a constant at its own
        mean, removing all of its variance and, necessarily, any
        correlation it carried with other parameters.

        Parameters
        ----------
        ablate_mu_r : bool, optional
            Shift r so mean(r) == mean(diag(interaction_matrix)). The
            default is False.
        ablate_mu_Aij : bool, optional
            Shift the off-diagonal entries of interaction_matrix so their
            mean == mean(diag(interaction_matrix)). The default is False.
        ablate_sigma_r : bool, optional
            Collapse r to a constant at its own mean. The default is False.
        ablate_sigma_Aij : bool, optional
            Collapse the off-diagonal entries of interaction_matrix to a
            constant at their own mean. The default is False.
        ablate_sigma_Aii : bool, optional
            Collapse the diagonal of interaction_matrix to a constant at
            its own mean. The default is False.

        Returns
        -------
        A_new : npt.NDArray
            Ablated interaction matrix.
        r_new : npt.NDArray
            Ablated growth rates.

        '''

        S = self.no_species
        mask = ~np.eye(S, dtype=bool)

        A_new = self.interaction_matrix.copy()
        r_new = self.r.copy()

        # reference point for mean ablations, taken before any ablation is
        # applied below
        mu_Aii = np.diagonal(A_new).mean()

        # --- mean ablations (additive shift; preserves variance/correlations) ---

        if ablate_mu_r:
            r_new = r_new - r_new.mean() + mu_Aii

        if ablate_mu_Aij:
            A_new[mask] = A_new[mask] - A_new[mask].mean() + mu_Aii

        # --- variance ablations (collapse to a constant, destroys all structure) ---

        if ablate_sigma_r:
            r_new = np.full(S, r_new.mean())

        if ablate_sigma_Aij:
            A_new[mask] = A_new[mask].mean()

        if ablate_sigma_Aii:
            np.fill_diagonal(A_new, np.diagonal(A_new).mean())

        return A_new, r_new

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
                                 = {'Aii' : 1},
                             empirical : bool = True):

        '''

        Generate parameters specific to the CRM with self-limiting resource
        dynamics - consumer death rates and intrinsic resource growth rates

        Parameters
        ----------
        empirical : bool, optional
            Only used if interaction_args['rhos'] requests correlated
            interactions. Standardises the interaction matrix (and, if
            cross-parameter correlations rho_r_Aij/rho_Aii_Aij are also
            requested, the row means used to build r/Aii) to their realised
            mean/variance per realisation, rather than their theoretical
            expected values - see __correlated_interactions() and
            __correlated_rates() for details. The default is True.

        Returns
        -------
        None.

        '''

        # check whether we have cross-parameter correlations to handle
        cross_rhos = None
        if interaction_args.get('rhos') is not None:
            cross_rhos = interaction_args['rhos'].pop('rho_r_Aij', None)
            # pull both out if either is present
            if cross_rhos is not None:
                cross_rhos = {
                    'rho_r_Aij': cross_rhos,
                    'rho_Aii_Aij': interaction_args['rhos'].pop('rho_Aii_Aij', 0.0),
                }

        # --- generate A_ij first (needed before r/Aii if cross-correlated) ---

        if interaction_args.get('rhos') is None:

            self.other_parameter_methods(interaction_method,
                                         interaction_args,
                                         'Aij',
                                         (self.no_species, self.no_species))
        else:

            self.__correlated_interactions(interaction_args['mu'],
                                           interaction_args['sigma'],
                                           **interaction_args['rhos'],
                                           empirical=empirical)

        # --- generate r and Aii ---

        if cross_rhos is not None:

            # cross-parameter correlations requested:
            # build r and Aii from row means of the already-generated Z
            self.__correlated_rates(growth_args,
                                    self_inhibition_args,
                                    **cross_rhos,
                                    empirical=empirical)
        else:

            # no cross-parameter correlations: generate independently as before
            self.other_parameter_methods(growth_method,
                                         growth_args,
                                         'r',
                                         (self.no_species, ))

            self.other_parameter_methods(self_inhibition_method,
                                         self_inhibition_args,
                                         'Aii',
                                         (self.no_species, ))

        self.interaction_matrix = self.Aij
        np.fill_diagonal(self.interaction_matrix, self.Aii)

    def __correlated_interactions(self,
                                  mu,
                                  sigma,
                                  rho_D=0.0,
                                  rho_R=0.0,
                                  rho_C=0.0,
                                  rho_1idx=0.0,
                                  empirical=True,
                                  rng=None):
        """
        Generate an S x S off-diagonal interaction matrix with four target
        correlations, following the latent-factor decomposition in
        Castedo, Holmes, Baron & Galla (arXiv:2409.12751), Sec. II.C.

        Each standardised entry is decomposed as:

            Z_ij = lambda_i + kappa_j + w_ij

        where:
            - (lambda_i, kappa_i) are species-level latent traits drawn as
              correlated pairs, generating row correlation (rho_R), column
              correlation (rho_C), and one-index correlation (rho_1idx).
            - (w_ij, w_ji) are edge-level latent pairs drawn with correlation
              that generates the reciprocal/diagonal correlation (rho_D).

        The final matrix is A_ij = mu + sigma * Z_ij.

        Stores self.Aij for use by model_specific_rates, and self._Z and
        self._rho_R for use by __correlated_rates (if cross-parameter
        correlations are requested).

        If any feasibility constraints are violated, self.corr_violate is
        set to True but generation proceeds anyway — eigenvalues are clipped
        to zero, producing the nearest valid covariance structure.

        Parameters
        ----------
        mu : float
            Mean of off-diagonal entries.
        sigma : float
            Standard deviation of off-diagonal entries.
        rho_D : float
            Reciprocal (cross-diagonal) correlation, corr(A_ij, A_ji).
        rho_R : float
            Row correlation, corr(A_ij, A_ik) for j != k.
        rho_C : float
            Column correlation, corr(A_ij, A_kj) for i != k.
        rho_1idx : float
            One-index correlation, corr(A_ij, A_ki).
        empirical : bool
            If True, standardise Z's off-diagonal entries to have exactly
            mean 0 and variance 1 before applying mu + sigma * Z. This
            guarantees the target sigma per realisation but makes the
            construction data-dependent. If False, use Z as-is (unit
            variance in expectation, can fluctuate per realisation).
        rng : numpy.random.Generator, optional
            Random number generator for reproducibility.
        """
        rng = np.random.default_rng() if rng is None else rng
        S = self.no_species

        # --- floor each rho at 0 ---

        rho_D = max(rho_D, 0.0)
        rho_R = max(rho_R, 0.0)
        rho_C = max(rho_C, 0.0)
        rho_1idx = max(rho_1idx, 0.0)

        # --- feasibility check (logs violation, does not stop) ---

        self.__feasibility_reason(rho_R, rho_C, rho_1idx, rho_D)

        # --- species-level latent traits (Eq. 4) ---

        cov_species = np.array([[rho_R, rho_1idx],
                                 [rho_1idx, rho_C]])
        L_species = self.__psd_sqrt(cov_species)

        species_traits = rng.standard_normal((S, 2)) @ L_species.T
        lam = species_traits[:, 0]
        kap = species_traits[:, 1]

        # --- edge-level latent pairs (Eq. 5) ---

        resid_var = 1 - rho_R - rho_C
        resid_cov = rho_D - 2 * rho_1idx

        cov_edge = np.array([[resid_var, resid_cov],
                              [resid_cov, resid_var]])
        L_edge = self.__psd_sqrt(cov_edge)

        iu, ju = np.triu_indices(S, k=1)
        edge_pairs = rng.standard_normal((iu.size, 2)) @ L_edge.T

        W = np.zeros((S, S))
        W[iu, ju] = edge_pairs[:, 0]
        W[ju, iu] = edge_pairs[:, 1]

        # --- assemble standardised matrix (Eq. 6) ---

        Z = lam[:, None] + kap[None, :] + W

        # --- empirical standardisation of Z ---
        # At finite S, the off-diagonal entries of Z can have
        # empirical variance != 1.0, because each latent component
        # (lam, kap, W) has only S or S(S-1)/2 independent draws.
        # When this variance overshoots, sigma * Z inflates the
        # effective interaction strength, widening the row-sum
        # distribution and reducing diversity.
        #
        # Rescaling all off-diagonal entries by a single scalar
        # preserves all four within-A correlations exactly (they
        # are ratios of covariances, and uniform scaling cancels).

        if empirical:
            print("empirical i")
            mask = ~np.eye(S, dtype=bool)
            Z_offdiag = Z[mask]
            Z_mean = Z_offdiag.mean()
            Z_std = Z_offdiag.std()
            Z = (Z - Z_mean) / (Z_std + 1e-20)

        # --- store for cross-parameter correlations ---

        self._Z = Z
        self._rho_R = rho_R

        # --- rescale to target moments ---

        self.Aij = mu + sigma * Z
        self.corr_violate = getattr(self, 'corr_violate', False)

    def __correlated_rates(self,
                           growth_args,
                           self_inhibition_args,
                           rho_r_Aij=0.0,
                           rho_Aii_Aij=0.0,
                           empirical=True,
                           rng=None):
        """
        Construct growth rates (r) and self-inhibition (Aii) that are
        each correlated with the off-diagonal interactions (Aij), by
        loading onto the row means of the standardised interaction
        matrix Z.

        The correlation between r and Aii is not set independently —
        it emerges from both loading onto the same row-mean signal:

            corr(r_i, A_ii) = beta_r * beta_Aii

        Two standardisation modes control how row means are normalised
        and how the loading coefficients are computed:

            empirical=True  (default): uses the actual variance and
                correlation of this realisation's row means. Guarantees
                Var(r) = sigma_r^2 exactly per realisation. Use for
                simulations comparing to specific eLV instances.

            empirical=False: uses the theoretical expected variance
                c = (1 + (S-2)*rho_R)/(S-1). Var(r) = sigma_r^2 on
                average across realisations, but can fluctuate for any
                individual one. Use for analytical predictions (cavity
                method, random matrix theory).

        Must be called AFTER __correlated_interactions, which stores
        self._Z and self._rho_R.

        Parameters
        ----------
        growth_args : dict
            Must contain 'mu' (or 'r') and 'sigma' for growth rate moments.
        self_inhibition_args : dict
            Must contain 'mu' (or 'Aii') and 'sigma' for self-inhibition moments.
        rho_r_Aij : float
            Target corr(r_i, A_ij) for j != i.
        rho_Aii_Aij : float
            Target corr(A_ii, A_ij) for j != i.
        empirical : bool
            If True, standardise row means using their realised variance.
            If False, use the theoretical expected variance.
        rng : numpy.random.Generator, optional
            Random number generator for reproducibility.
        """
        rng = np.random.default_rng() if rng is None else rng
        S = self.no_species
        tol = 0.05

        # --- extract moment targets ---

        r_mean = growth_args.get('r', growth_args.get('mu', 1.0))
        sigma_r = growth_args.get('sigma', 0.0)
        Aii_mean = self_inhibition_args.get('Aii', self_inhibition_args.get('mu', 1.0))
        sigma_Aii = self_inhibition_args.get('sigma', 0.0)

        # =====================================================
        # Shared signal: row means of standardised Z
        # =====================================================

        Z = self._Z
        rho_R = self._rho_R
        mask = ~np.eye(S, dtype=bool)

        Z_offdiag = Z.copy()
        np.fill_diagonal(Z_offdiag, 0)
        row_means = Z_offdiag.sum(axis=1) / (S - 1)

        # =====================================================
        # Standardisation and conversion factor
        # =====================================================
        # row_means_unit: row means scaled to unit variance.
        # corr_factor: corr(Zbar_i, Z_ij) — converts a loading
        #   on row means into the resulting corr with individual entries.
        #
        # Empirical: both computed from the realised Z.
        # Theoretical: both computed from rho_R and S alone.

        if empirical:

            rm_std = row_means.std()
            row_means_unit = ((row_means - row_means.mean())
                              / (rm_std + 1e-20))

            rm_repeated = np.repeat(row_means, S - 1)
            corr_factor = np.corrcoef(rm_repeated, Z[mask])[0, 1]
            
            print('empirical')

        else:

            c = (1 + (S - 2) * rho_R) / (S - 1)
            row_means_unit = row_means / np.sqrt(max(c, 1e-20))
            corr_factor = np.sqrt(max(c, 1e-20))
            
            print('theoretical')

        # =====================================================
        # Loading r onto row means
        # =====================================================
        # corr(r_i, A_ij) = beta_r * corr_factor
        # so beta_r = rho_r_Aij / corr_factor

        beta_r = rho_r_Aij / corr_factor if abs(corr_factor) > 1e-12 else 0.0

        if beta_r**2 > 1 + tol:
            self.corr_violate = True

        beta_r = np.clip(beta_r, -1, 1)
        noise_r_std = np.sqrt(max(1 - beta_r**2, 0))

        self.r = r_mean + sigma_r * (beta_r * row_means_unit
                                      + noise_r_std * rng.standard_normal(S))

        # =====================================================
        # Loading Aii onto row means
        # =====================================================
        # corr(A_ii, A_ij) = beta_Aii * corr_factor
        # so beta_Aii = rho_Aii_Aij / corr_factor

        beta_Aii = rho_Aii_Aij / corr_factor if abs(corr_factor) > 1e-12 else 0.0

        if beta_Aii**2 > 1 + tol:
            self.corr_violate = True

        beta_Aii = np.clip(beta_Aii, -1, 1)
        noise_Aii_std = np.sqrt(max(1 - beta_Aii**2, 0))

        self.Aii = Aii_mean + sigma_Aii * (beta_Aii * row_means_unit
                                           + noise_Aii_std * rng.standard_normal(S))

        # The implied corr(r, Aii) = beta_r * beta_Aii, which emerges
        # naturally from both loading on the same signal. No additional
        # correlation is imposed.

        # --- clean up temporary storage ---

        del self._Z, self._rho_R

    def __feasibility_reason(self,
                             rho_R,
                             rho_C,
                             rho_1idx,
                             rho_D,
                             tol=0.05):
        conditions = [
            (rho_R < -tol or rho_C < -tol,
             "rho_row and rho_col must be >= 0."),
            (abs(rho_1idx) - tol > np.sqrt(rho_R * rho_C),
             "|rho_cross| must be <= sqrt(rho_row*rho_col)."),
            (rho_D - 2 * rho_1idx < -tol,
             "rho_recip must be >= 2*rho_cross."),
            (rho_R + rho_C + (rho_D - 2 * rho_1idx) > 1 + tol,
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
                     
       
