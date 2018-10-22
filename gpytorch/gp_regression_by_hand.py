# gp regression, coded by hand

import math
import torch
import gpytorch
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


from gpytorch.kernels.matern_kernel import MaternKernel

#import ipdb; ipdb.set_trace()
x_end = 20
sigma_prior = 0.1
dim_x = 2
mkernel = MaternKernel(ard_num_dims=dim_x) # this arguments tell the kernel that our input is 2d

X_to_evaluate = torch.ones(5, dim_x, requires_grad=True)
X_rand = torch.rand(5, dim_x, requires_grad=False)
#x_train = torch.linspace(0,4,20, requires_grad=False).unsqueeze(1)

x_train = x_end*torch.rand(20, 2, requires_grad=False)#.unsqueeze(1)
y_train = torch.sin(torch.norm(x_train, dim=1))#+torch.normal(torch.ones(x_train.shape))
#import ipdb; ipdb.set_trace()

def mean_pred(X_to_evaluate, y_train, x_train):
    assert X_to_evaluate.shape[1] == x_train.shape[1]
    #mu_zeros = torch.zeros(y_train.shape)
    mu_prior = torch.zeros(y_train.shape)
    #import ipdb; ipdb.set_trace()
    K_test = mkernel.forward(x_train, x_train).squeeze(0)+sigma_prior*torch.eye(x_train.shape[0])
    K_new = mkernel.forward(X_to_evaluate, x_train).squeeze(0)
    #import ipdb; ipdb.set_trace()
    mu_predict = torch.matmul(K_new, torch.inverse(K_test)).mm((y_train-mu_prior).unsqueeze(1))
    return mu_predict

def covar_pred(X_to_evaluate, y_train, x_train):
    assert X_to_evaluate.shape[1] == x_train.shape[1]
    #mu_zeros = torch.zeros(y_train.shape)
    
    K_test = mkernel.forward(x_train, x_train).squeeze(0)+sigma_prior*torch.eye(x_train.shape[0])
    K_new = mkernel.forward(X_to_evaluate, x_train).squeeze(0)
    K_pred = mkernel.forward(X_to_evaluate, X_to_evaluate).squeeze(0)
    sigma_predict = K_pred - torch.matmul(K_new, torch.inverse(K_test)).mm(K_new.t())
    return sigma_predict

def one_expected_improvement(X_to_evaluate, y_train, x_train, sample_size=100):
    assert X_to_evaluate.shape[1] == x_train.shape[1]
    #import ipdb; ipdb.set_trace()
    f_max = y_train.max()
    out_covar = covar_pred(X_to_evaluate, y_train, x_train)
    #import ipdb; ipdb.set_trace()
    out_mean = mean_pred(X_to_evaluate, y_train, x_train)
    L_x = torch.potrf(out_covar, upper=False)
    Z = torch.normal(torch.ones(X_to_evaluate.shape[0], sample_size))
    min_value, __ = torch.min(out_mean + L_x.mm(Z), dim=0)
    inner_term = torch.max((f_max - min_value), torch.zeros(sample_size))
    #import ipdb; ipdb.set_trace()
    return inner_term.mean()

#X_one = torch.ones(1, 1, requires_grad=True)

#mean_expected_improvement = one_expected_improvement(X_one, y_train, x_train)

#out_covar = covar_pred(X_to_evaluate, y_train, x_train)
#out_mean = mean_pred(X_to_evaluate, y_train, x_train)
if False: 
    for t in range(1000):
        import ipdb; ipdb.set_trace()
        out_mean = mean_pred(X_to_evaluate, y_train, x_train)
        out_mean.backward(y_train)
        with torch.no_grad():
            X_to_evaluate += 0.001 * X_to_evaluate.grad
            X_to_evaluate.grad.zero_()
# play around and make some predictions
m_multistart = 10
import copy
import numpy as np
X_test = (torch.rand((m_multistart, dim_x), requires_grad=False)*2*x_end)-x_end
X_test_copy = copy.deepcopy(X_test)
#import ipdb; ipdb.set_trace()
#X_test = torch.tensor([9.], requires_grad=True, dtype=torch.float)
multistart_loss = []
multistart_x_test = []
all_points_list = []
all_loss_list = []
for i_multistart in range(m_multistart):
    X_test_inter = X_test[i_multistart,:].unsqueeze(0)
    X_test_inter.requires_grad = True
    #import ipdb; ipdb.set_trace()
    for t in range(1000):
        import ipdb; ipdb.set_trace()
        one_expected_improvement(X_test, y_train, x_train, 1000)
        loss = one_expected_improvement(X_test_inter, y_train, x_train, 1000)
        all_loss_list.append(loss.data.numpy())
        all_points_list.append(copy.deepcopy(X_test_inter).data.numpy())
        #loss = mean_pred(X_test_inter, y_train, x_train)
        loss.backward(retain_graph=True)
        with torch.no_grad():
            X_test_inter += 0.01 * X_test_inter.grad
            #print(X_test_inter.grad)
            #store_grad = X_test_inter.grad.data.numpy()
            X_test_inter.grad.zero_()
    print(X_test_inter, loss)
    multistart_loss.append(loss.data.numpy())
    multistart_x_test.append(X_test_inter.data.numpy())
#import ipdb; ipdb.set_trace()
if True: 
    x_support = torch.linspace(-x_end, x_end, 200, requires_grad=False).unsqueeze(1)
    confidence_bound = covar_pred(x_support, y_train, x_train).diag()
    predicted_y = mean_pred(x_support, y_train, x_train).squeeze()
    #plt.scatter(x_train.data.numpy(), y_train.data.numpy(), label='true points')
    plt.scatter(np.array(all_points_list), np.array(all_loss_list), label='trajectories')
    plt.scatter(np.array(multistart_x_test), np.array(multistart_loss), label='optimized point')
    plt.scatter(np.array(X_test_copy.data.numpy()), one_expected_improvement(X_test_copy, y_train, x_train, 1000).squeeze().data.numpy(), label='starting point')
    #plt.plot(x_support.data.numpy(), predicted_y.data.numpy(), label='predicted_points')
    #plt.plot(x_support.data.numpy(), (predicted_y+2*confidence_bound).data.numpy(), label='upper bound')
    #plt.plot(x_support.data.numpy(), (predicted_y-2*confidence_bound).data.numpy(), label='lower bound')
    plt.plot(x_support.data.numpy(), one_expected_improvement(x_support, y_train, x_train, 1000).data.numpy(), label='expected improvement')
    plt.legend()
    plt.show()



import ipdb; ipdb.set_trace()