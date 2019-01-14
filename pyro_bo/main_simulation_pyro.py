# main simulation based on pyro
import pickle
import torch
import numpy as np

from bo_using_pyro import run_bo_pyro


def f_norm(x):
    """
    the target target function that we are optimizing
    """
    return torch.norm((x -0.5), dim=1)
    

def f_hart6(x,
          alpha=np.asarray([1.0, 1.2, 3.0, 3.2]),
          P=10**-4 * np.asarray([[1312, 1696, 5569, 124, 8283, 5886],
                                 [2329, 4135, 8307, 3736, 1004, 9991],
                                 [2348, 1451, 3522, 2883, 3047, 6650],
                                 [4047, 8828, 8732, 5743, 1091, 381]]),
          A=np.asarray([[10, 3, 17, 3.50, 1.7, 8],
                        [0.05, 10, 17, 0.1, 8, 14],
                        [3, 3.5, 1.7, 10, 17, 8],
                        [17, 8, 0.05, 10, 0.1, 14]])):
    """The six dimensional Hartmann function is defined on the unit hypercube.

    It has six local minima and one global minimum f(x*) = -3.32237 at
    x* = (0.20169, 0.15001, 0.476874, 0.275332, 0.311652, 0.6573).

    More details: <http://www.sfu.ca/~ssurjano/hart6.html>
    """
    return -np.sum(alpha * np.exp(-np.sum(A * (np.array(x) - P)**2, axis=1)))


def f_branin(x, a=1, b=5.1 / (4 * np.pi**2), c=5. / np.pi,
           r=6, s=10, t=1. / (8 * np.pi)):
    """Branin-Hoo function is defined on the square x1 ∈ [-5, 10], x2 ∈ [0, 15].

    It has three minima with f(x*) = 0.397887 at x* = (-pi, 12.275),
    (+pi, 2.275), and (9.42478, 2.475).

    More details: <http://www.sfu.ca/~ssurjano/branin.html>
    """
    # transform the samples, since we restrict the space to [0,1]
    x0 = (x[:,0]*15)-5 # x1 ∈ [-5, 10]
    x1 = x[:,1]*15 # x2 ∈ [0, 15]
    return (a * (x1 - b * x0 ** 2 + c * x0 - r) ** 2 +
            s * (1 - t) * np.cos(x0) + s)

f_target = f_branin
dim = 2
np.random.seed(42)
X = np.random.random(size=(7, dim))
X = torch.tensor(X, dtype=torch.float)
#import ipdb; ipdb.set_trace()
y = f_target(X)
sample_sizes_list = [5, 10, 20, 50]#, 100]

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
outer_loop_steps = 30
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
        with open('pyro_bo_mrep_%s_%s.pkl'%(Mrep, f_target.__name__), 'wb') as file:
            pickle.dump(res_dict, file, protocol=2)


