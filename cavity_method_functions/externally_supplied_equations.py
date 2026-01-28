# -*- coding: utf-8 -*-
"""
Created on Wed May  7 16:47:26 2025

@author: jamil

"""

import numpy as np
from scipy.special import erfc
import os

os.chdir('C:/Users/jamil/Documents/PhD/Code Repositories/Ecological-Dynamics-Consumer-Resource-Models/cavity_method_functions')

# %%

def self_consistency_equations(gamma, 
                               rho,
                               mu_c, sigma_c,
                               mu_g, sigma_g,
                               mu_d, sigma_d,
                               mu_b, sigma_b, 
                               mu_o, sigma_o,
                               phi_N, N_mean, q_N, v_N,
                               R_mean, q_R, chi_R):
    
    #breakpoint()
    
    # average species growth rate
    mu_gN = (mu_g * R_mean) - mu_d
    
    # average resource growth rate
    mu_gR = mu_o + (mu_c * N_mean)/gamma
    
    # std. in species growth rate
    sigma_gN = np.sqrt((sigma_g**2 * q_R) + sigma_d**2)
    
    # std. in resource growth rate
    sigma_gR = np.sqrt((sigma_c**2 * q_N)/gamma + sigma_o**2)
    
    # delta gN
    delta_gN = mu_gN/sigma_gN
    
    # species gaussian error function (survival fraction)
    erf_dN = 0.5*erfc(-delta_gN/np.sqrt(2))
    
    # species exponential term
    exp_dN = np.exp(-(delta_gN**2)/2)/np.sqrt(2 * np.pi)
    
    X = sigma_g * sigma_c * rho * chi_R
    Y = (sigma_g * sigma_c * rho * v_N)/gamma
    
    ##### Species self consistency equations ####
    eq_phi_N = erf_dN
    
    eq_N_mean = (sigma_gN/X) * (exp_dN + delta_gN*erf_dN)
    
    eq_q_N = (sigma_gN/X)**2 * (delta_gN*exp_dN + (1 + delta_gN**2)*erf_dN)
    
    eq_v_N = -phi_N/X
    
    ##### Resource self consistency equations ####
    eq_R_mean = (1/(2*Y)) * (mu_gR \
                             - np.sqrt(mu_gR**2 - 4*Y*mu_b) \
                             - (sigma_gR**2/(2 * np.sqrt(mu_gR**2 - 4*Y*mu_b))))
        
    eq_q_R = (1/(4*Y**2)) * (2 * mu_gR**2 \
                             + 2 * sigma_gR**2 \
                             - 4 * Y * mu_b \
                            - 2 * mu_gR * np.sqrt(mu_gR**2 - 4 * Y * mu_b) \
                             - (3 * mu_gR * sigma_gR**2) / np.sqrt(mu_gR**2 - 4 * Y * mu_b) \
                             + (4 * (mu_gR * sigma_gR)**2 + 3 * sigma_gR**4 \
                                    + 16 * (Y * sigma_b)**2) / (mu_gR**2 - 4 * Y * mu_b))
    #eq_q_R = R_mean**2 + (sigma_gR**4 + \
    #                      (8*(Y*sigma_b)**2)/(2*Y**2*(mu_gR**2 - 4*Y*mu_b)) + \
    #                      (sigma_gR**2 * (np.sqrt(mu_gR**2 - 4*Y*mu_b) - mu_gR)**2)/(4*Y**2*(mu_gR**2 - 4*Y*mu_b)))
    
    eq_chi_R = -(1/(2*Y)) * (1 \
                             - mu_gR / np.sqrt(mu_gR**2 - 4 * Y * mu_b) \
                             + (mu_gR * sigma_gR**2)/(2*(mu_gR**2 - 4 * Y * mu_b)**(3/2)))
    #eq_chi_R = -(1/(2*Y)) * (1 - \
    #                         mu_gR / np.sqrt(mu_gR**2 - 4 * Y * mu_b) + \
    #                        (3 * mu_gR * sigma_gR**2)/(2*(mu_gR**2 - 4*Y*mu_b)**(3/2)))
    
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
                             gamma,
                             rho,
                             mu_c, sigma_c,
                             sigma_g,
                             mu_b,
                             mu_o, sigma_o,
                             phi_N, N_mean, q_N, v_N, chi_R):
    
    #breakpoint()
     
    mu_gR = mu_o + (mu_c * N_mean)/gamma
    Y = (sigma_g * rho * v_N)/gamma
          
    eq_dNde = (dRde + 1) * (1/(sigma_c * rho * chi_R)**2) 
    
    eq_dRde = (dNde + 1) * ((phi_N/gamma) / Y**2) * ( \
                         0.25 \
                         - mu_gR / np.sqrt(mu_gR**2 - 4 * Y * mu_b) \
                         + (mu_gR**2 + sigma_o**2 + 1.5*sigma_c**2) / (mu_gR**2 - 4 * Y * mu_b))
    
    f_to_min = np.array([dNde - eq_dNde, dRde - eq_dRde])
    
    return f_to_min

# %%

#def distance_from_multistability_threshold(y):
    
#    eq_kwargs_names = ['mu_c', 'sigma_c', 'sigma_g', 'rho', 'gamma', 'mu_D',
#                       'sigma_D', 'mu_K',
#                       'chi_R', 'phi_N', 'N_mean', 'q_N']
    
#    eq_kwargs = {key : y[key] for key in eq_kwargs_names}
    
#    ms_condition = multistability_condition(**eq_kwargs)
    
#    return ms_condition

# %%

def instability_condition(gamma, 
                          rho,
                          mu_c, sigma_c,
                          sigma_g,
                          mu_b,
                          mu_o, sigma_o,
                          phi_N, N_mean, q_N, v_N, chi_R):
    
    mu_gR = mu_o + (mu_c * N_mean)/gamma
    Y = (sigma_g * rho * v_N)/gamma
          
    frac_1A = (1/(sigma_c * rho * chi_R)**2) 
    
    frac_1B =  ((phi_N/gamma) / Y**2) * ( \
                         0.25 \
                         - mu_gR / np.sqrt(mu_gR**2 - 4 * Y * mu_b) \
                         + (mu_gR**2 + sigma_o**2 + 1.5*sigma_c**2) / (mu_gR**2 - 4 * Y * mu_b))
    
    A = 1 / frac_1A
    B = 1 / frac_1B
    
    return A*B - 1