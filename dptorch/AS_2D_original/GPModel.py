import torch
import gpytorch
from gpytorch.means import Mean


class CustomMean(Mean):
    def __init__(self, value=0. ,batch_shape=torch.Size(), **kwargs):
        super(CustomMean, self).__init__()
        self.batch_shape = batch_shape
        self.value = value

    def forward(self, input):
        mean = self.value * torch.ones(*self.batch_shape, 1, dtype=input.dtype, device=input.device)
        if input.shape[:-2] == self.batch_shape:
            return mean.expand(input.shape[:-1])
        else:
            return mean.expand(torch.broadcast_shapes(input.shape[:-1], mean.shape))

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, d,p, train_x, train_y, likelihood, cfg, batch_shape=torch.Size([])):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PiecewisePolynomialKernel(
                q = 0,
                ard_num_dims = train_x.shape[-1],
                eps = 1e-7,
            ))
        self.batch_shape = batch_shape
        self.mean = torch.mean(train_x, 0)
        self.var = torch.var(train_x, 0)

    def forward(self, x):
        # x = (x - self.mean)/torch.sqrt(self.var + 1e-5)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        if not self.batch_shape:
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        else:
            return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            )


