# -*- coding: utf-8 -*-
"""
Smoke test for simulation_functions_unified.py - the merged/generalised
replacement for resource_diversity_stability(sl)/simulation_functions.py and
external_resource_stability/simulation_functions_new.py.

Covers: every supported model class, both 'constant' and 'normal'
model-specific rates, and the save/load (CRM_df) round trip.

Run with:
    python test_simulation_functions_unified.py
"""

import sys
import os
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

np.random.seed(0)

import simulation_functions_unified as sfu

M, S = 5, 6

# %% ---- model_specific_args + full pipeline for every non-trophic model, constant rates ----

constant_parm_sets = {
    "Self-limiting resource supply":
        dict(M=M, S=S, mu_c=1, sigma_c=0.1, mu_g=1, sigma_g=0.1, rho=0.5,
             d=0.5, b=1),
    "Self-limiting resource supply, self-inhibition":
        dict(M=M, S=S, mu_c=1, sigma_c=0.1, mu_g=1, sigma_g=0.1, rho=0.5,
             d=0.5, b=1, si=0.1),
    "Self-limiting resource supply, leached":
        dict(M=M, S=S, mu_c=1, sigma_c=0.1, mu_g=1, sigma_g=0.1, rho=0.5,
             d=0.5, b=1, A=0),
    "Externally-supplied resources":
        dict(M=M, S=S, mu_c=1, sigma_c=0.1, mu_g=1, sigma_g=0.1, rho=0.5,
             d=0.5, b=1, o=0.5),
    "Hybrid resource supply":
        dict(M=M, S=S, mu_c=1, sigma_c=0.1, mu_g=1, sigma_g=0.1, rho=0.5,
             d=0.5, b=1, o=0.1, a=0.05),
    "Metabolic pathways":
        dict(M=M, S=S, mu_c=1, sigma_c=0.1, mu_g=1, sigma_g=0.1, rho=1,
             d=0.5, o=1, b=0.1, A=0.05,
             mean_q=1, variance_q=1, p=0.3),
}

for model, parm_set in constant_parm_sets.items():

    parameter_sets = [parm_set]

    init_list, ms_args_list, extra_calls_list = sfu.model_specific_args(parameter_sets, model)
    assert init_list[0]['pool_sizes'] == [M, S]

    gc_args_list = sfu.growth_consumption_rates_args(parameter_sets, model)

    communities = sfu.consumer_resource_model_dynamics(init_list[0],
                                                        gc_args_list[0],
                                                        ms_args_list[0],
                                                        no_communities=1,
                                                        no_init_conds=1,
                                                        t_end=30,
                                                        extra_calls=extra_calls_list[0])

    assert len(communities) == 1
    community = communities[0]
    assert not np.any(np.isnan(community.ODE_sols[0].y))

    if model == "Metabolic pathways":
        assert hasattr(community, 'q') and hasattr(community, 'w') and hasattr(community, 'p')

    df = sfu.simulation_df_from_communities(communities, model, 'coupled by rho')
    assert len(df) == 1
    # death rate was 'constant' -> d_val column should exist
    assert 'd_val' in df.columns

    print(model, "(constant): OK -", list(df.columns))

# %% ---- normal-distributed death rate (SL_CRM) ----

normal_parm_set = dict(M=M, S=S, mu_c=1, sigma_c=0.1, mu_g=1, sigma_g=0.1, rho=0.5,
                       mu_d=0.5, sigma_d=0.05, b=1)

init_list, ms_args_list, extra_calls_list = sfu.model_specific_args([normal_parm_set],
                                                                    "Self-limiting resource supply")
assert ms_args_list[0]['death_method'] == 'normal'

gc_args_list = sfu.growth_consumption_rates_args([normal_parm_set], "Self-limiting resource supply")

communities = sfu.consumer_resource_model_dynamics(init_list[0], gc_args_list[0], ms_args_list[0],
                                                    no_communities=1, no_init_conds=1, t_end=30)
assert not np.any(np.isnan(communities[0].ODE_sols[0].y))
assert hasattr(communities[0], 'mu_d') and hasattr(communities[0], 'sigma_d')
assert not hasattr(communities[0], 'd_val')

df = sfu.simulation_df_from_communities(communities, "Self-limiting resource supply", 'coupled by rho')
assert 'mu_d' in df.columns and 'sigma_d' in df.columns and 'd_val' not in df.columns
print("SL_CRM (normal death rate): OK -", list(df.columns))

# %% ---- explicit method override ----

override_parm_set = dict(M=M, S=S, mu_c=1, sigma_c=0.1, mu_g=1, sigma_g=0.1, rho=0.5,
                         d=np.random.uniform(0.1, 1, S), b=1)
# even though 'd' is an array (which would auto-infer 'user-supplied'), force 'constant'
# via the override to prove the explicit-override tier of infer_rate_spec works
override_parm_set['d_method'] = 'user-supplied'

method, args = sfu.infer_rate_spec(override_parm_set, 'd')
assert method == 'user-supplied'
print("infer_rate_spec explicit override: OK")

# %% ---- multi-trophic model ----

trophic_parm_set = dict(pool_sizes=[M, S, 4], trophic_levels=3,
                        mu_c_2=1, sigma_c_2=0.1, mu_g_2=1, sigma_g_2=0.1, rho_2=0.5,
                        mu_c_3=1, sigma_c_3=0.1, mu_g_3=1, sigma_g_3=0.1, rho_3=0.5,
                        d_2=0.5, d_3=0.5, b=1, mu_A=0.1, sigma_A=0.05)

init_list, ms_args_list, extra_calls_list = sfu.model_specific_args([trophic_parm_set],
                                                                    sfu.TROPHIC_MODEL)
assert init_list[0]['pool_sizes'] == [M, S, 4]

gc_args_list = sfu.growth_consumption_rates_args([trophic_parm_set], sfu.TROPHIC_MODEL)

communities = sfu.consumer_resource_model_dynamics(init_list[0], gc_args_list[0], ms_args_list[0],
                                                    no_communities=1, no_init_conds=1, t_end=30)
assert not np.any(np.isnan(communities[0].ODE_sols[0].y))

df = sfu.simulation_df_from_communities(communities, sfu.TROPHIC_MODEL, 'coupled by rho')
assert len(df) == 1
print("SL_TL_CRM: OK -", list(df.columns))

# %% ---- save_models / CRM_df round trip (regression test for the missing
#          'method' argument bug in the original CRM_df) ----

scratch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_scratch_sfu")
if os.path.exists(scratch_dir):
    shutil.rmtree(scratch_dir)
os.makedirs(scratch_dir)

try:
    sl_parm_sets = [constant_parm_sets["Self-limiting resource supply"]]
    sl_init_list, sl_ms_args_list, _ = sfu.model_specific_args(sl_parm_sets,
                                                               "Self-limiting resource supply")
    sl_gc_args_list = sfu.growth_consumption_rates_args(sl_parm_sets, "Self-limiting resource supply")

    sl_communities = sfu.consumer_resource_model_dynamics(
        sl_init_list[0], sl_gc_args_list[0], sl_ms_args_list[0],
        no_communities=1, no_init_conds=1, t_end=30)

    sfu.save_models(sl_communities, scratch_dir, "roundtrip_test")

    reloaded_df = sfu.CRM_df(scratch_dir, ['no_resources', 'no_species', 'mu_c', 'sigma_c'])
    assert len(reloaded_df) == 1
    print("save_models -> CRM_df round trip: OK -", list(reloaded_df.columns))

finally:
    shutil.rmtree(scratch_dir)

print("\nAll simulation_functions_unified smoke tests passed.")
