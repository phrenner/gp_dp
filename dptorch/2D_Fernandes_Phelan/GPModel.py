import torch
import gpytorch
from gpytorch.means import Mean
from .Model import V_INFINITY_


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, d,p, train_x, train_y, likelihood, cfg, batch_shape=torch.Size([])):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.d = d
        self.p = p
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=batch_shape)
        self.batch_shape = batch_shape
        if p == 0:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.PiecewisePolynomialKernel(
                    q = 0,
                    ard_num_dims = train_x.shape[-1],
                    eps = 1e-7,
                )) + gpytorch.kernels.ConstantKernel()
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.PiecewisePolynomialKernel(
                    q = 0,
                    ard_num_dims = train_x.shape[-1],
                    eps = 1e-7,
                )) + gpytorch.kernels.ConstantKernel()

        gp_offset = cfg["model"]["params"]["GP_offset"]
        lower_V = cfg["model"]["params"]["lower_V"]
        beta = cfg["model"]["params"]["beta"]
        self.cfg = cfg

        if p == 0:
            mask_feas = train_y + gp_offset > lower_V
        else:
            mask_feas = torch.ones_like(train_y).bool()
        self.min_feas_train_y = torch.min(train_y[mask_feas])
        self.mean_x = torch.mean(train_x[mask_feas,:], 0)
        self.std_x = torch.std(train_x[mask_feas,:], 0)
        self.mean_train_y = torch.mean(train_y[mask_feas])
        self.std_train_y = torch.std(train_y[mask_feas])

        self.scale_x = 1
        self.offset_x = 0
        self.scale_y = 1
        self.offset_y = 0

    def forward(self, x):
        x = (x - self.offset_x)/self.scale_x
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        if not self.batch_shape:
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        else:
            return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            )

    def predict_mean(self, x):
        observed_pred = self(x)
        mean = observed_pred.mean
        return self.scale_y * mean + self.offset_y
    
    def predict_var(self, x):
        observed_pred = self(x)
        var = observed_pred.variance
        return var

