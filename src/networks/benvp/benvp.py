import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior_mean, prior_var):
        super(RealNVP, self).__init__()
        
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(self.mask))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(self.mask))])

    @property
    def prior(self):
        return MultivariateNormal(
            self.prior_mean.to(self.mask.device),
            self.prior_var.to(self.mask.device)
        )  

    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J
    
    def log_prob(self,x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp, z
        
    def sample(self, z): 
        # logp = self.prior.log_prob(z)
        x = self.g(z)
        return x