# -*- coding: utf-8 -*-
"""
Created on Wed May  7 17:09:09 2025

@author: jamil
"""

import numpy as np
from scipy.special import erf
import os

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models/cavity_method_functions')

# %%

def self_consistency_equations(M, gamma,
                               mu_c, mu_y, sigma_c, sigma_y,
                               mu_d, sigma_d, mu_b, sigma_b,
                               phi_N, N_mean, q_N, v_N,
                               phi_R, R_mean, q_R, chi_R):
    
    '''
    
    Solve the self-consistency equations (sces) for the model
        dR_a/dt = R_a ( b_a - R_a - sum_{i=1}^S c_ia N_i ).
        dN_i/dt = N_i ( sum_{a=1}^M y_ia c_ia R_a - d_i )
        
    Inputs: some fixed quantities, can be the statistical properties of model 
    parameters or the statistical properties of consumer and resource distributions.
    
    Outputs: the solved quantities that satisfy the fixed parameters + sces.

    Parameters
    ----------
    M : float (but always an integer in value)
        Resource pool size (initial no. resources).
    gamma : float
        The ratio of the species pool size to resource pool size (M/S).
        (Therefore, S = M/gamma.)
    mu_c : float
        The average total resource consumption rate (per resource = mu_c/M).
    mu_y : float
        The average yield conversion factor.
    sigma_c : float
        The standard deviation in the total resource consumption rate (per 
        resource = sigma_c/root(M).
    sigma_y : float
        The standard deviation in the yield conversion factor.
    mu_d : float
        The average species/consumer death rate.
    sigma_d : float
        The standard deviation in death rate.
    mu_b : float
        The average intrinsic growth rate of a resource.
    sigma_b : float
        The standard deviation in the intrinsic growth rate of a resource.
    phi_N : float
        The species/consumer survival fraction.
    N_mean : float
        The average species/consumer abundance.
    q_N : float
        The second moment of the species/consumer abundance distribution.
    v_N : float
        Average species/consumer suseptibility.
    phi_R : float
        The resource survival fraction.
    R_mean : TYPE
        DESCRIPTION.
    q_R : TYPE
        DESCRIPTION.
    chi_R : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    
    # average species growth rate
    mu_gN = (mu_y * mu_c * R_mean) - mu_d
    
    # std. in species growth rate
    sigma_gN = np.sqrt(q_R * (((mu_c * sigma_y)**2)/M + (mu_y * sigma_c)**2 + \
                               (sigma_c * sigma_y)**2) \
                        + sigma_d**2)
    
    # delta g^(N)
    delta_gN = mu_gN/sigma_gN
    
    # species gaussian error function (survival fraction)
    erf_gN = 0.5*(1 + erf(delta_gN/np.sqrt(2)))
    
    # species exponential term
    exp_gN = np.exp(-(delta_gN**2)/2)/np.sqrt(2 * np.pi)
    
    A = mu_y * sigma_c**2 * chi_R
    
    ##### Species self consistency equations ####
    eq_phi_N = erf_gN
    eq_N_mean = (sigma_gN/A) * (exp_gN + delta_gN*erf_gN)
    eq_q_N = (sigma_gN/A)**2 * (delta_gN*exp_gN + (1 + delta_gN**2)*erf_gN)
    #eq_v_N = -phi_N/A
    eq_v_N = -phi_N / (mu_y * sigma_c**2 * (phi_R - phi_N/gamma))
    
    ##### Resource self consistency equations ####
    
    # average resource growth rate
    mu_gR = mu_b - (mu_c * N_mean)/gamma
    
    # std. in resource growth rate
    sigma_gR = np.sqrt((sigma_c**2 * q_N)/gamma + sigma_b**2)
    
    # delta g^(R)
    delta_gR = mu_gR/sigma_gR
    
    # resource gaussian error function (survival fraction)
    erf_gR = 0.5*(1 + erf(delta_gR/np.sqrt(2)))
    
    # resource exponential tern
    exp_gR = np.exp(-(delta_gR**2)/2)/np.sqrt(2 * np.pi)
    
    B = 1 - (mu_y * sigma_c**2 * v_N)/gamma
    
    eq_phi_R = erf_gR
    eq_R_mean = (sigma_gR/B) * (exp_gR + delta_gR*erf_gR)
    eq_q_R = (sigma_gR/B)**2 * (delta_gR*exp_gR + (1 + delta_gR**2)*erf_gR)
    eq_chi_R = phi_R - phi_N/gamma
    
    f_to_min = np.array([phi_N - eq_phi_N,
                         N_mean - eq_N_mean,
                         q_N - eq_q_N,
                         v_N - eq_v_N,
                         phi_R - eq_phi_R,
                         R_mean - eq_R_mean,
                         q_R - eq_q_R,
                         chi_R - eq_chi_R])
    
    return f_to_min

# %%

def multistability_equations(dNde, dRde, 
                             M, gamma, mu_c, mu_y, sigma_c, sigma_y,
                             phi_N, phi_R, v_N, chi_R):
    
    N_n = phi_R * (((mu_c*sigma_y)**2)/M + (mu_y*sigma_c)**2 + (sigma_c*sigma_y)**2)
    N_d = (mu_y * sigma_c**2 * chi_R)**2
          
    eq_dNde = (N_n/N_d) * (dRde + 1)
    
    R_n = sigma_c**2 * phi_N/gamma
    R_d = (1 - (mu_y * sigma_c**2 * v_N)/gamma)**2
    
    eq_dRde = (R_n/R_d) * (dNde + 1)
    
    f_to_min = np.array([dNde - eq_dNde, dRde - eq_dRde])
    
    return f_to_min

# %%

def instability_condition(M, gamma,
                          mu_c, mu_y, sigma_c, sigma_y,
                          mu_d, sigma_d, mu_b, sigma_b,
                          phi_N, N_mean, q_N, v_N, phi_R, R_mean, q_R, chi_R):
    
    rho_squared = 1 / (1 + ((sigma_y/mu_y)**2) * (1 + ((mu_c/sigma_c)**2)/M))
     
    return (rho_squared - phi_N/(gamma * phi_R))