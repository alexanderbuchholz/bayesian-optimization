# function BO
import torch
import gpytorch
from gpytorch.kernels.matern_kernel import MaternKernel

sigma_prior = 0.1


class GPregression(torch.nn.Module):
    """
    inherits from the GP regression
    """
    def __init__(self, parameters_model, parameters_data):
        #super(OneExpectedImprovement, self).__init__(parameters_model, parameters_data)
        super(GPregression, self).__init__()
        self.dim_x = torch.tensor(parameters_data['dim_x']) # dimension of the problem
        self.y_train = torch.tensor(parameters_data['y_train'], dtype=torch.float32, requires_grad=False)
        self.x_train = torch.tensor(parameters_data['x_train'], dtype=torch.float32, requires_grad=False)
        self.sigma_prior = torch.tensor(parameters_model['sigma_prior'], dtype=torch.float32)
        mkernel = MaternKernel(ard_num_dims=self.dim_x)
        

        #import ipdb; ipdb.set_trace()
        self.mkernel = mkernel
        self.my_param = torch.nn.Parameter(parameters_data['x_to_evaluate'], requires_grad=True)
        #self.my_param
        #ipdb.set_trace()
        #The parameters we adjust during training.

    def fit_gp(self):
        """
        fits the gp
        """
        assert self.my_param.shape[1] == self.x_train.shape[1]
        #mu_zeros = torch.zeros(y_train.shape)
        self.mu_prior = torch.zeros(self.y_train.shape)
        #import ipdb; ipdb.set_trace()
        self.K_test = self.mkernel.forward(self.x_train, self.x_train).squeeze(0)+self.sigma_prior*torch.eye(self.x_train.shape[0])
        self.K_test_inv = torch.inverse(self.K_test)

    def update_gp(self, y_train_new, x_train_new):
        """
        add elements to the list of points and refit the gp
        """
        self.y_train = torch.cat((self.y_train, y_train_new))
        self.x_train = torch.cat((self.x_train, x_train_new))
        self.fit_gp()

    def mean_pred(self):
        """
        mean of the gp
        """
        K_new = self.mkernel.forward(self.my_param, self.x_train).squeeze(0)
        #import ipdb; ipdb.set_trace()
        mu_predict = torch.matmul(K_new, self.K_test_inv).mm((self.y_train-self.mu_prior).unsqueeze(1))
        return mu_predict

    def covar_pred(self):
        """
        covariance of the gp
        """
        assert self.my_param.shape[1] == self.x_train.shape[1]
        #mu_zeros = torch.zeros(y_train.shape)
        
        K_new = self.mkernel.forward(self.my_param, self.x_train).squeeze(0)
        K_pred = self.mkernel.forward(self.my_param, self.my_param).squeeze(0)
        sigma_predict = K_pred - torch.matmul(K_new, self.K_test_inv).mm(K_new.t())
        return sigma_predict



class OneExpectedImprovement(GPregression):
    def __init__(self, parameters_model, parameters_data):
        #super(OneExpectedImprovement, self).__init__(parameters_model, parameters_data)
        super(OneExpectedImprovement, self).__init__(parameters_model, parameters_data)
        self.mkernel.log_lengthscale.requires_grad = False

    def one_expected_improvement(self, sample_size=10):
        assert self.my_param.shape[1] == self.x_train.shape[1]
        assert self.my_param.shape[0] == 1
        #import ipdb; ipdb.set_trace()
        f_max = self.y_train.max()
        out_covar = self.covar_pred()
        #import ipdb; ipdb.set_trace()
        out_mean = self.mean_pred()
        L_x = torch.potrf(out_covar, upper=False)
        Z = torch.normal(torch.ones(self.my_param.shape[0], sample_size))
        min_value, __ = torch.min(out_mean + L_x.mm(Z), dim=0)
        inner_term = torch.max((f_max - min_value), torch.zeros(sample_size))
        #import ipdb; ipdb.set_trace()
        return inner_term.mean()


def maximize_one_ei(class_ei, training_iter = 500):
    optimizer = torch.optim.RMSprop([
    {'params': filter(lambda p: p.requires_grad, class_ei.parameters())},  # Includes GaussianLikelihood parameters
    ], lr=0.1)
    print(class_ei.one_expected_improvement())
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        # Calc loss and backprop gradients
        loss = class_ei.one_expected_improvement()
        loss.backward(retain_graph = True)
        optimizer.step()
    optimized_value = class_ei.one_expected_improvement()
    print(optimized_value)
    for name, param in class_ei.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    return(class_ei.my_param, optimized_value)
