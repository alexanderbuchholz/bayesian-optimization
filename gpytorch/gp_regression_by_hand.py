# gp regression, coded by hand

import math
import torch
import gpytorch
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


from gpytorch.kernels.matern_kernel import MaternKernel

#import ipdb; ipdb.set_trace()
x_end = 50
sigma_prior = 0.1
mkernel = MaternKernel()

X_to_evaluate = torch.ones(5, 1, requires_grad=True)
X_rand = torch.rand(5, 1, requires_grad=False)
#x_train = torch.linspace(0,4,20, requires_grad=False).unsqueeze(1)

x_train = x_end*torch.rand(20, 1, requires_grad=False)#.unsqueeze(1)
y_train = torch.sin(x_train)#+torch.normal(torch.ones(x_train.shape))

def mean_pred(X_to_evaluate, y_train, x_train):
    #mu_zeros = torch.zeros(y_train.shape)
    mu_prior = torch.zeros(y_train.shape)
    K_test = mkernel.forward(x_train, x_train).squeeze(0)+sigma_prior*torch.eye(x_train.shape[0])
    K_new = mkernel.forward(X_to_evaluate, x_train).squeeze(0)
    mu_predict = torch.matmul(K_new, torch.inverse(K_test)).mm(y_train-mu_prior)
    return mu_predict

def covar_pred(X_to_evaluate, y_train, x_train):
    #mu_zeros = torch.zeros(y_train.shape)
    
    K_test = mkernel.forward(x_train, x_train).squeeze(0)+sigma_prior*torch.eye(x_train.shape[0])
    K_new = mkernel.forward(X_to_evaluate, x_train).squeeze(0)
    K_pred = mkernel.forward(X_to_evaluate, X_to_evaluate).squeeze(0)
    sigma_predict = K_pred - torch.matmul(K_new, torch.inverse(K_test)).mm(K_new.t())
    return sigma_predict

def one_expected_improvement(X_to_evaluate, y_train, x_train, sample_size=100):
    f_max = y_train.max()
    out_covar = covar_pred(X_to_evaluate, y_train, x_train)
    out_mean = mean_pred(X_to_evaluate, y_train, x_train)
    L_x = torch.potrf(out_covar, upper=False)
    Z = torch.normal(torch.ones(X_to_evaluate.shape[0], sample_size))
    inner_term = torch.max(f_max - (out_mean - L_x.mm(Z)), torch.zeros(out_mean.shape))
    #import ipdb; ipdb.set_trace()
    return inner_term.mean(1)

X_one = torch.ones(1, 1, requires_grad=True)

mean_expected_improvement = one_expected_improvement(X_one, y_train, x_train)

#out_covar = covar_pred(X_to_evaluate, y_train, x_train)
#out_mean = mean_pred(X_to_evaluate, y_train, x_train)
if False: 
    for t in range(1000):
        out_mean = mean_pred(X_to_evaluate, y_train, x_train)
        out_mean.backward(X_rand)
        with torch.no_grad():
            X_to_evaluate += 0.001 * X_to_evaluate.grad
            X_to_evaluate.grad.zero_()
# play around and make some predictions
if True: 
    x_support = torch.linspace(0, x_end, 100, requires_grad=False).unsqueeze(1)
    confidence_bound = covar_pred(x_support, y_train, x_train).diag()
    predicted_y = mean_pred(x_support, y_train, x_train).squeeze()
    plt.scatter(x_train.data.numpy(), y_train.data.numpy(), label='true points')
    plt.plot(x_support.data.numpy(), predicted_y.data.numpy(), label='predicted_points')
    plt.plot(x_support.data.numpy(), (predicted_y+2*confidence_bound).data.numpy(), label='upper bound')
    plt.plot(x_support.data.numpy(), (predicted_y-2*confidence_bound).data.numpy(), label='lower bound')
    plt.plot(x_support.data.numpy(), one_expected_improvement(x_support, y_train, x_train).data.numpy(), label='expected improvement')
    plt.legend()
    plt.show()



import ipdb; ipdb.set_trace()