# -*- coding: utf-8 -*-
"""
Created on Wed May  7 16:47:27 2025

@author: jamil
"""

import numpy as np
from scipy.special import erfc
from scipy.special import erf
import os

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models/cavity_method_functions')

# %%

def self_consistency_equations(gamma, mu_c, mu_g, sigma_c, sigma_g,
                               mu_m, sigma_m, mu_K, sigma_K,
                               phi_N, N_mean, q_N, v_N, phi_R, R_mean, q_R, chi_R):
    
    # average species growth rate
    kappa = (mu_g * R_mean) - mu_m
    
    # average resource growth rate
    omega = mu_K - (mu_c * mu_g * N_mean)/gamma
    
    # std. in species growth rate
    sigma_kappa = np.sqrt(sigma_m**2 + (q_R * sigma_g**2))
     
    # std. in resource growth rate
    sigma_omega = np.sqrt(sigma_K**2 + (q_N/gamma)*((mu_c*sigma_g)**2 + (sigma_c*sigma_g)**2))
    
    # delta kappa
    delta_kappa = kappa/sigma_kappa
    
    # delta omega
    delta_omega = omega/sigma_omega
    
    # species gaussian error function (survival fraction)
    erf_dk = 0.5*erfc(-delta_kappa/np.sqrt(2))
    
    # resource gaussian error function (survival fraction)
    erf_do = 0.5*erfc(-delta_omega/np.sqrt(2))
    
    # species exponential term
    exp_dk = np.exp(-(delta_kappa**2)/2)/np.sqrt(2 * np.pi)
    
    # resource exponential tern
    exp_do = np.exp(-(delta_omega**2)/2)/np.sqrt(2 * np.pi)
    
    A = mu_c * sigma_g**2 * chi_R
    B = 1 - ((mu_c * sigma_g**2 * v_N)/gamma)
    
    ##### Species self consistency equations ####
    eq_phi_N = erf_dk
    eq_N_mean = (sigma_kappa/A) * (exp_dk + delta_kappa*erf_dk)
    eq_q_N = (sigma_kappa/A)**2 * (delta_kappa*exp_dk + (1 + delta_kappa**2)*erf_dk)
    eq_v_N = -phi_N/A
    
    ##### Resource self consistency equations ####
    eq_phi_R = erf_do
    eq_R_mean = (sigma_omega/B) * (exp_do + delta_omega*erf_do)
    eq_q_R = (sigma_omega/B)**2 * (delta_omega*exp_do + (1 + delta_omega**2)*erf_do)
    eq_chi_R = phi_R/B
    
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

def multistability_equations_inf(dNde, dRde, 
                                         gamma, mu_c, mu_g, sigma_c, sigma_g,
                                         phi_N, phi_R, chi_R, v_N):
    
    N_n = phi_R * sigma_g**2
    N_d = (mu_g * sigma_c**2 * chi_R)**2
          
    eq_dNde = (N_n/N_d) * (dRde + 1)
    
    R_n = (phi_N/gamma) * ((mu_c * sigma_g)**2 + (sigma_c * sigma_g)**2)
    R_d = (1 - (mu_c * sigma_g**2 * v_N)/gamma)**2
    
    eq_dRde = (R_n/R_d) * (dNde + 1)
    
    f_to_min = np.array([dNde - eq_dNde, dRde - eq_dRde])
    
    return f_to_min
