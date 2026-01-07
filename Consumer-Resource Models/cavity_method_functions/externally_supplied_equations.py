# -*- coding: utf-8 -*-
"""
Created on Wed May  7 16:47:26 2025

@author: jamil

"""

import numpy as np
from scipy.special import erfc
from scipy.special import erf
import os

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models/cavity_method_functions')

# %%

def self_consistency_equations(rho, gamma, mu_c, mu_g, sigma_c, sigma_g, mu_m,
                               sigma_m, mu_K, sigma_K, mu_D, sigma_D,
                               phi_N, N_mean, q_N, v_N, R_mean, q_R, chi_R):
    
    #breakpoint()
    
    # average species growth rate
    kappa = (mu_g * R_mean) - mu_m
    
    # average resource growth rate
    omega = mu_D + (mu_c * N_mean)/gamma
    
    # std. in species growth rate
    sigma_kappa = np.sqrt((sigma_g**2 * q_R) + sigma_m**2)
    
    # std. in resource growth rate
    sigma_omega = np.sqrt((sigma_c**2 * q_N)/gamma + sigma_D**2)
    
    # delta kappa
    delta_kappa = kappa/sigma_kappa
    
    # species gaussian error function (survival fraction)
    erf_dk = 0.5*erfc(-delta_kappa/np.sqrt(2))
    
    # species exponential term
    exp_dk = np.exp(-(delta_kappa**2)/2)/np.sqrt(2 * np.pi)
    
    A = sigma_g * sigma_c * rho * chi_R
    B = (sigma_g * sigma_c * rho * v_N)/gamma
    
    ##### Species self consistency equations ####
    eq_phi_N = erf_dk
    
    eq_N_mean = (sigma_kappa/A) * (exp_dk + delta_kappa*erf_dk)
    
    eq_q_N = (sigma_kappa/A)**2 * (delta_kappa*exp_dk + (1 + delta_kappa**2)*erf_dk)
    
    eq_v_N = -phi_N/A
    
    ##### Resource self consistency equations ####
    eq_R_mean = (1/(2*B)) * (omega - np.sqrt(omega**2 - 4*B*mu_K) - \
                             (sigma_omega**2/(2 * np.sqrt(omega**2 - 4*B*mu_K))))
        
    eq_q_R = (1/(2*B))**2 * (2*omega**2 + 3*sigma_omega**2 - 4*B*mu_K - \
                             2*omega*np.sqrt(omega**2 - 4*B*mu_K) - \
                             (6*omega*sigma_omega**2)/np.sqrt(omega**2 - 4*B*mu_K) + \
                             (4*(omega*sigma_omega)**2 + 3*sigma_omega**4 + 16*(B*sigma_K)**2)/(omega**2 - 4*B*mu_K))
    #eq_q_R = R_mean**2 + (sigma_omega**4 + \
    #                      (8*(B*sigma_K)**2)/(8*B**2*(omega**2 - 4*B*mu_K)) + \
    #                      (sigma_omega**2*(np.sqrt(omega**2 - 4*B*mu_K) - omega**2))/(4*B**2*(omega**2 - 4*B*mu_K)))
    
    eq_chi_R = -(1/(2*B)) * (1 - omega/np.sqrt(omega**2 - 4*B*mu_K) + \
                            (omega*sigma_omega**2)/(2*(omega**2 - 4*B*mu_K)**(3/2)))
    #eq_chi_R = -(1/(2*B)) * (1 - omega/np.sqrt(omega**2 - 4*B*mu_K) + \
    #                        (3*omega*sigma_omega**2)/(2*(omega**2 - 4*B*mu_K)**(3/2)))
    
    f_to_min = np.array([phi_N - eq_phi_N,
                         N_mean - eq_N_mean,
                         q_N - eq_q_N,
                         v_N - eq_v_N,
                         R_mean - eq_R_mean,
                         q_R - eq_q_R,
                         chi_R - eq_chi_R])
    
    return f_to_min

# %%

def multistability_equations(dNde, dRde, 
                             rho, gamma, mu_c, sigma_c, sigma_g, mu_K, mu_D, sigma_D,
                             phi_N, N_mean, q_N, v_N, chi_R):
    
    #breakpoint()
     
    omega = mu_D + (mu_c * N_mean)/gamma
    B = (sigma_g * sigma_c * rho * v_N)/gamma
    x = (phi_N * sigma_c**2)/gamma
          
    eq_dNde = (dRde + 1) * (1/(sigma_c * rho * chi_R)**2) 
    
    eq_dRde = (dNde + 1) * (1/(2*B))**2 * (x*(1 - \
                                              (4*omega)/(np.sqrt(omega**2 - 4*B*mu_K)) + \
                                              4*(omega**2 + (sigma_c*sigma_D)**2)/(omega**2 - 4*B*mu_K)) + \
                                           (x**2)*(q_N/(omega**2 - 4*B*mu_K)))
    
    f_to_min = np.array([dNde - eq_dNde, dRde - eq_dRde])
    
    return f_to_min

# %%

def distance_from_multistability_threshold(y):
    
    eq_kwargs_names = ['mu_c', 'sigma_c', 'sigma_g', 'rho', 'gamma', 'mu_D',
                       'sigma_D', 'mu_K',
                       'chi_R', 'phi_N', 'N_mean', 'q_N']
    
    eq_kwargs = {key : y[key] for key in eq_kwargs_names}
    
    ms_condition = multistability_condition(**eq_kwargs)
    
    return ms_condition

# %%

def multistability_condition(mu_c, sigma_c, sigma_g, rho, gamma, mu_D, sigma_D, 
                             mu_K,
                             chi_R, phi_N, N_mean, q_N):
    
    omega = mu_D + (mu_c * N_mean)/gamma
    
    x = phi_N*(sigma_c**2)/gamma
    
    A = (sigma_c * rho * chi_R)**2
    
    frac_1B = ((chi_R**2)/(4/gamma)**2) * (x*(1 - (omega/np.sqrt(omega**2 + (4*mu_K)/(gamma*chi_R))) + \
                                              4*(omega**2 + (sigma_c*sigma_D)**2)/(omega**2 + (4*mu_K)/(gamma*chi_R))) + \
                                           (x**2)*(q_N/(omega**2 + (4*mu_K)/(gamma*chi_R))))
    B = 1/frac_1B
    
    return A*B - 1