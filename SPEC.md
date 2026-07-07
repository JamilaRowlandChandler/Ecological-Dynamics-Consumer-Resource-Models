# \# Metabolic Pathway Model



Implement a new model in the consumer\_resource\_modules folder, alongside the GLV and other consumer resource models. This model will be a consumer-resource style model which incorporates structured metabolic pathways.



\## Model Structure



% Consumer dynamics



% $\\frac{dN\_i}{dt} = N\_i (\\sum\_\\alpha^M g\_{i, \\alpha} R\_\\alpha \\sum\_\\beta^M \\frac{q\_{i, \\alpha \\beta} (w\_\\alpha - w\_\\beta)}{\\sum\_\\gamma^M q\_{i, \\alpha \\gamma} + \\epsilon} - d\_i)$



% Resource dynamics



% $\\frac{dR\_\\alpha}{dt} = (o\_\\alpha - b\_\\alpha R\_\\alpha - A\_{\\alpha \\alpha} R\_\\alpha^2) - \\sum\_i^S c\_{i, \\alpha} R\_\\alpha N\_i \\sum\_\\beta^M \\frac{b\_{i, \\alpha \\beta}}{\\sum\_\\gamma^M b\_{i, \\alpha \\gamma} + \\epsilon} + \\sum\_i^S \\sum\_\\beta^M c\_{i, \\beta} p\_{i, \\beta} \\frac{q\_{i, \\beta \\alpha}}{\\sum\_\\gamma^M  q\_{i, \\beta \\gamma} + \\epsilon} R\_\\beta N\_i$



\## Parameter sampling



% c\_{i, \\alpha} is sampled from a normal distribution



% g\_{i, \\alpha} is sampled from a normal distribution that is coupled in some way to c\_{i, \\alpha}



% w\_\\alpha and w\_\\beta (all w's) are sampled from a uniform distribution with min = 0 and max = 1

% q\_{i, \\alpha \\beta} (all q's) is sampled from a Bernoulli distribution, so takes the values 0 or 1 with probability p. p is determined by a gamma distribution function f(x). For each pair of resources \\alpha and \\beta, for each consumer i, p = f(w\_\\alpha - w\_\\beta).



% d\_i can be sampled from a normal distribution, be constant, or supplied by the user



% o\_\\alpha can be sampled from a normal distribution, be constant, or supplied by the user



% b\_\\alpha can be sampled from a normal distribution, be constant, or supplied by the user



% A\_{\\alpha \\alpha} can be sampled from a normal distribution, be constant, or supplied by the user



% p\_{i, \\beta} can be sampled from a normal distribution, be constant, or supplied by the user



\## Code Structure



% The model class should be put in models.py, and should inherit from ParametersInterface, DifferentialEquationsInterface, and CommunityPropertiesInterface as the other consumer-resource models do. It should also be an option called by the Consumer\_Resource\_Models wrapper. 



% It should have the same init function as the other consumer-resource models in model.py, taking in the pool\_sizes argument



% The class does not need its on methods for generating g\_{i, \\alpha} and c\_{i, \\alpha}, this can be handled by the growth\_consumption\_rates method



% The model will need its own model\_specific\_rates methods, which should be defined with the class in models.py, for generating o\_\\alpha, b\_\\alpha and A\_{\\alpha \\alpha}. It can use the same version of the method as the Hybrid\_CRM class.



% You need to develop a new method in ParametersInterface for generating metabolic networks and resource production rates. 

% Sampling the w's: They should either be user-supplied, or sampled from a uniform distribution varying between 0 and 1, sample size = pool\_size\[0]. Then, the gamma distribution function should be generated and used as a likelihood function. 

% Sampling the b's: They should either be user-supplied, or sampled from a Bernoulli distribution with probability x, which will vary with each b. x is defined by a gamma function f(x), where the user should be able to specify some parameters relating to the function, like its mean and variance. To sample b for each resource pair (\\alpha and beta) and consumer, p = f(w\_\\alpha - w\_\\beta), and is then plugged into the Bernoulli distribution function. Make sure to construct the b's in the appropriate tensors. I think the sample size should be something like pool\_size\[0]\*\*2 \* pool\_size\[1]. Finally, generate the p's. They should be normally distributed, constant, or user-supplied. The parameters should then be assigned to the class in the same way as growth\_consumption\_rates method

% Therefore, the method structure should look something like this (doesn't have to be exactly this):


def metabolic\_network(energies : Union\[None, npt.NDArray] = None, # for the ws, if None generate, if array use these

&#x09;	      resource\_conversions: Union\[TypedDict('generate', {'mean' : float, 'variance' : float}), None] = {'mean' : 1, 'variance' : 1} # for generating the gamma function and then b's,

&#x09;	     production : Union\[TypedDict('normal', {'mu' : float, 'sigma' : float}), TypedDict('constant', {'p' : float}), TypedDict('user-supplied', {'p' : npt.NDArray})] = {'mu' : 1, 'sigma' : 0})



% The model will need its own simulation method containing the model function for simulating the ODE. This can follow the same structure as the simulation methods in the other model classes in models.py, but with the correct equations.



% The model does not need methods for generating initial conditions, saving simulation results, and calculating community properties

