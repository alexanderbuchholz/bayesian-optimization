# fit the gp and test it
from __future__ import print_function
from __future__ import division


import math
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
#import seaborn as sns
import numpy as np


import torch
import gpytorch


# Training data is 11 points in [0,1] inclusive regularly spaced
dim = 1
#train_x = torch.randn((20, dim))
train_x = torch.linspace(0, 1, 20).unsqueeze(1)
# True function is sin(2*pi*x) with Gaussian noise
def target_y(train_x):
    train_y = torch.sin(train_x.sum(1)**2 * (2 * math.pi)) + torch.randn(train_x.size()[0]) * 0.2
    return(train_y)
train_y = target_y(train_x)

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.random_variables.GaussianRandomVariable(mean_x, covar_x)

class TrainedGPModel(ExactGPModel):
    def __init__(self, train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood()):
        super(TrainedGPModel, self).__init__(train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood())
        self.train_x = train_x
        self.train_y = train_y

    def train_model(self):
        # initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(self.train_x, self.train_y, self.likelihood)

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        training_iter = 101
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            loss.backward()
            if i % 50 == 0:
                print('Iter %d/%d - Loss: %.3f   log_lengthscale: %.3f   log_noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    self.model.covar_module.base_kernel.log_lengthscale.item(),
                    self.model.likelihood.log_noise.item()
                ))
            optimizer.step()
    
    def update_gp(self, new_x, new_y):
        # update the gp with new points and retrains the model
        self.train_x = torch.cat([self.train_x, new_x])
        self.train_y = torch.cat([self.train_y, new_y])
        self.train_model()

    def pred_model(self, test_x):
        self.model.eval()
        self.likelihood.eval()

        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.fast_pred_var():
            observed_pred = self.likelihood(self.model(test_x))
            f_preds = self.model(test_x)
            self.f_mean = f_preds.mean()
            self.f_var = f_preds.var()
            self.f_covar = f_preds.covar()
            self.f_samples = f_preds.sample(1000)
        return self.f_mean, self.f_covar, observed_pred
        

    def prepare_grad(self, x_for_g):
        self.x_for_g = x_for_g

    def pred_model_grad(self):
        self.model.eval()
        self.likelihood.eval()
        #observed_pred = self.likelihood(self.model(test_x))
        f_preds = self.model(self.x_for_g)
        f_mean = f_preds.mean()
        return(f_mean)


#import ipdb; ipdb.set_trace()


test_x = torch.linspace(0, 1, 100).unsqueeze(1)
test_y = target_y(test_x)


trained_gp = TrainedGPModel(train_x, train_y)
trained_gp.train_model()
#trained_gp.update_gp(test_x, test_y)




x_one = torch.ones((1,dim), requires_grad=True)
trained_gp.prepare_grad(x_one)
optimizer = torch.optim.Adam([trained_gp.x_for_g], lr=0.1)

for t in range(100):
    optimizer.zero_grad()
    loss = trained_gp.pred_model_grad()
    loss.backward()
    optimizer.step()

#import ipdb; ipdb.set_trace()
# Get into evaluation (predictive posterior) mode

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
f_mean, f_covar, observed_pred = trained_gp.pred_model(trained_gp.train_x)


with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(trained_gp.train_x.numpy(), trained_gp.train_y.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(trained_gp.train_x.numpy(), observed_pred.mean().numpy(), 'b')
    ax.plot(trained_gp.x_for_g.detach().numpy(), trained_gp.pred_model_grad().detach().numpy(), 'k*', color="green")
    # Shade between the lower and upper confidence bounds
    ax.fill_between(trained_gp.train_x.numpy().flatten(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    plt.show()

import ipdb; ipdb.set_trace()