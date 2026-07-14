# \# Resource pool size vs stability in the metabolic pathway model



% I want to investigate how increasing the resource pool size (no\_resources) affects stability in the metabolic pathway model (MP\_CRM).

% This will involve running many simulations across resource pool sizes and other parameter distributions, then estimating the stability of each community.

% Simulations should be run and saved using functions from "simulation\_functions\_unified.py".

% Please write the code that can achieve this, updating "simulation\_functions\_unified.py" as necessary.





\## Setting up a directory for running simulations



% Firstly, create a new folder in "/Ecological-Dynamics-Consumer-Resource-Models" called "/resource\_diversity\_stability\_crossfeeding". 

% Move "simulation\_functions\_unified.py" and "test\_simulation\_functions\_unified.py" from "/consumer\_resource\_modules" to this new folder.

% Also make a copy of "self\_consistency\_equation\_functions.py" and add it to this folder.

% Then, create a folder within "/resource\_diversity\_stability\_crossfeeding" called "stability\_transitions". 

% Within this folder, create a file called "mu\_c\_vs\_M.py". The structure of this file should look similar to "C:/Users/jamil/Documents/PhD/Code Repositories/CRM-Resource-diversity-vs-Stability/Simulation codes/mu\_c\_vs\_M.py" 



\## Running and saving the simulations

% As stated, the code should follow a similar structure to "C:/Users/jamil/Documents/PhD/Code Repositories/CRM-Resource-diversity-vs-Stability/Simulation codes/mu\_c\_vs\_M.py", tweaking the codes appropriately to make sure they're run in the correct directory.

% You may tweak it to remove inefficient or confusing code, but the code must call CRM\_across\_parameter\_space from "self\_consistency\_equation\_functions.py" to create and save communities.

% If necessary, tweak CRM\_across\_parameter\_space so that it can call MP\_CRM using different methods.

% The subdirectory argument in CRM\_across\_parameter\_space should begin with "resource\_diversity\_stability\_crossfeeding/mu\_c\_vs\_M"

\# Use save\_method = 'v3'



\## Model arguments and methods for generating parameters



\# resource\_pool\_sizes (which will eventually be supplied as pool\_sizes to MP\_CRM) = np.arange(50, 275, 25)

\# sigma\_c = 1.6

\# mu\_y = 1,

\# sigma\_y = 0.13,

\# d = 1,

\# b = 1,

\# For the metabolic\_network method, please set network\_method = 'step' and p\_s = 1.

\# You will need to vary mu\_c to find an appropriate range where you observe a stability transition (where the community.lyapunov\_exponent changes sign or where the community.ODE\_sols\[0].t\[-1] changes from equalling t\_end (supplied to solve\_ivp) to terminating early, or visa versa). As a constraint, mu\_c >> sigma\_c Do some tests and find an appropriate range.



\# The code from "C:/Users/jamil/Documents/PhD/Code Repositories/CRM-Resource-diversity-vs-Stability/Simulation codes/mu\_c\_vs\_M.py" already has functions that scale appropriate parameters by the resource pool size. Make sure these are not removed.

