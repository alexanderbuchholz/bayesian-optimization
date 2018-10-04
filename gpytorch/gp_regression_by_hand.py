# gp regression, coded by hand

import math
import torch
import gpytorch
from matplotlib import pyplot as plt


from gpytorch.kernels.matern_kernel import MaternKernel

#import ipdb; ipdb.set_trace()
mkernel = MaternKernel()

X_to_evaluate = torch.ones(5, 1, requires_grad=True)
X_ones = torch.rand(5, 1, requires_grad=False)
x_test = torch.linspace(0,4,20, requires_grad=False).unsqueeze(1)
y_test = torch.sin(x_test)

def mean_pred(X_to_evaluate, y_test, x_test):
    #mu_zeros = torch.zeros(y_test.shape)
    mu_prior = torch.zeros(y_test.shape)
    K_test = mkernel.forward(x_test, x_test).squeeze(0)
    K_new = mkernel.forward(X_to_evaluate, x_test).squeeze(0)
    mu_predict = torch.matmul(K_new, torch.inverse(K_test)).mm(y_test-mu_prior)
    return mu_predict

def covar_pred(X_to_evaluate, y_test, x_test):
    #mu_zeros = torch.zeros(y_test.shape)
    
    K_test = mkernel.forward(x_test, x_test).squeeze(0)
    K_new = mkernel.forward(X_to_evaluate, x_test).squeeze(0)
    K_pred = mkernel.forward(X_to_evaluate, X_to_evaluate).squeeze(0)
    sigma_predict = K_pred - torch.matmul(K_new, torch.inverse(K_test)).mm(K_new.t())
    return sigma_predict

out_covar = covar_pred(X_to_evaluate, y_test, x_test)
#out_mean = mean_pred(X_to_evaluate, y_test, x_test)
for t in range(1000):
    out_mean = mean_pred(X_to_evaluate, y_test, x_test)
    out_mean.backward(X_ones)
    with torch.no_grad():
        X_to_evaluate += 0.001 * X_to_evaluate.grad
        X_to_evaluate.grad.zero_()

import ipdb; ipdb.set_trace()