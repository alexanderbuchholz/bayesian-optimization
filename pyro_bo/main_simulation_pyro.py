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
sample_sizes_list = [5, 10, 20, 50, 100]

params_data = {
    'X' : X,
    'y' : y,
    'dim' : dim,
    'noise' : 0.01, 
    'f_target' : f_target
    }

params_bo_mc = {
    'sampling_type' : 'MC', 
    'sample_size' : sample_sizes_list[0],
    'num_candidates' : 20
}

params_bo_rqmc = {
    'sampling_type' : 'RQMC', 
    'sample_size' : sample_sizes_list[0],
    'num_candidates' : 20
}

Mrep = 40
outer_loop_steps = 40
res_dict = {'MC': {str(sample_size): [] for sample_size in sample_sizes_list}, 
            'RQMC' : {str(sample_size): [] for sample_size in sample_sizes_list}
            }
for sample_size in sample_sizes_list:
    params_bo_mc['sample_size'] = sample_size
    params_bo_rqmc['sample_size'] = sample_size
    for m_rep in range(Mrep):
        __, mc_dict = run_bo_pyro(params_bo_mc, params_data, outer_loop_steps=outer_loop_steps)
        __, rqmc_dict = run_bo_pyro(params_bo_rqmc, params_data, outer_loop_steps=outer_loop_steps)
        res_dict['MC'][str(sample_size)].append(mc_dict)
        res_dict['RQMC'][str(sample_size)].append(rqmc_dict)
        with open('pyro_bo_mrep_%s_v2.pkl'%Mrep, 'wb') as file:
            pickle.dump(res_dict, file, protocol=2)


