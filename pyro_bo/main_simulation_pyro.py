# main simulation based on pyro
import pickle
import torch
import numpy as np

from bo_using_pyro import run_bo_pyro


def f_target(x):
    """
    the target target function that we are optimizing
    """
    return torch.norm((x -0.5), dim=1)


dim = 2
np.random.seed(42)
X = np.random.random(size=(7, dim))
X = torch.tensor(X, dtype=torch.float)
y = f_target(X)

params_data = {
    'X' : X,
    'y' : y,
    'dim' : dim,
    'noise' : 0.01, 
    'f_target' : f_target
    }

params_bo_mc = {
    'sampling_type' : 'MC', 
    'sample_size' : 20,
    'num_candidates' : 20
}

params_bo_rqmc = {
    'sampling_type' : 'RQMC', 
    'sample_size' : 20,
    'num_candidates' : 20
}

Mrep = 20
res_dict = {'MC': [], 'RQMC' : []}

for m_rep in range(Mrep):
    __, mc_dict = run_bo_pyro(params_bo_mc, params_data)
    __, rqmc_dict = run_bo_pyro(params_bo_rqmc, params_data)
    res_dict['MC'].append(mc_dict)
    res_dict['RQMC'].append(rqmc_dict)
    with open('pyro_bo_test.pkl', 'wb') as file:
        pickle.dump(res_dict, file, protocol=2)


