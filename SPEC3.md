# Effect of resource supply on consumer coexistence

## Model set up

### Create a file "supply_consumer_diversity_2_consumers.py" in "C:\Users\jamil\Documents\PhD\Code Repositories\Ecological-Dynamics-Consumer-Resource-Models\resource_species_diversity\influx_diversity". 
### Set up a model using MP_CRM with two consumers (N_1 and N_2) and three resources (R_1, R_2, and R_3).
### R_1 has an energy (w_1) of 2, w_2 = 1, w_2 = 0.
### Both consumers can convert R_1 -> R_2 and R_2 -> R_3. Therefore, the metabolic network is gated, and R_1 cannot be converted directly into R_3.
### saturation_kinetics='reversible'.
### Both consumers have K_m = 1 and y_ia = 1 for all resources, d_i = 1. They should only differ in their consumption rates c_ia.
### For all resources, a_a = 0.
### R_2 and R_3 are not externally-supplied: o_2 and o_3 = 0.

## Investigate how changing the resource supply affects consumer coexistence

### Vary the supply rate of resource 1 (o_1) slowly (e.g., from np.linspace(0, 2, 0.025), but you may need to choose a suitable parameter range) and the dilution rate (b_a, which is the same for all resources) (e.g., from np.linspace(0, 1, 0.025), but you may need to decide this yourself).
### Consumption rates c_ia should be randomly sampled from a uniform distribution (Start with min=0.5, max=1, but vary e.g., if consumers never survive).
### Simulate community dynamics until they reach steady state.
### Assess the final resource and consumer diversity at the end of a simulation. Both properties can be calculated automatically using the calculate_community_properties() method. After calling this method on your model object, you can extract the resource_survival fraction and species_survival_fraction attribute. This is a list, where each entry is the resource/consumer survival fraction at the end of a simulation from a set of initial abundances.
### For each supply + dilution rate, simulate 30 communities. For each community, simulate from 5 initial conditions.
### Create a directory "C:\Users\jamil\Documents\PhD\Data\resource_species_diversity\influx_diversity" and save data on all varying model parameters (o_1, b_a, consumption rates c_11, c_12, c_21, c_22), the resource survival fraction and the consumer survival fraction in a csv file. (Do not save any data, .csv, .pkl files in the repo.)

## Plot data

# Make two heatmaps plotting (1) the average resource survival fraction and (2) the average consumer survival fraction. The columns should be the influx rate of resource 1 o_1 and the indexes should be b_a.  

