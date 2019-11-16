#%%
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import torch.distributions.constraints as constraints
import torch.distributions.constraints as constraints
from pyro import plate
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import matplotlib.pyplot as plt

# %% Data
# We know start and stop path:

#%%
class LatentMovement:
    def __init__(self):
        self.dim = 2
        self.T = 100
        self.z0 = torch.zeros((self.dim,))
        self.zT = torch.ones((self.dim,))

        self.zh = torch.ones((self.dim,))
        self.Th = 50
        self.zh[0] = 0.2
    def model(self):
        sigma = torch.ones(self.dim)*0.05
        # Prior
        z = torch.zeros((self.T+1, self.dim))
        z[0] = self.z0
        for t in range(1,self.T+1):
            if t == self.Th:
                z[self.Th] = pyro.sample("obs_{self.Th}", dist.Normal(z[self.Th-1], sigma), obs = self.zh)
            elif t == self.T:
                z[self.T] = pyro.sample("obs_{self.T}", dist.Normal(z[self.T-1], sigma), obs = self.zT)
            
            z[t] = pyro.sample(f"z_{t}", dist.Normal(z[t-1,], sigma))

        # data
        
        return z

    def guide(self):
        z_mu = pyro.param("z-mu", torch.zeros(self.T, self.dim)+0.5)
        z_scale = pyro.param("z-scale", torch.ones(self.T, self.dim)+0.5, constraint = constraints.positive)

        z = torch.zeros((self.T+1, self.dim))
        
        for t in range(1,self.T):
            z[t] = pyro.sample(f"z_{t}", dist.Normal(z_mu[t], z_scale[t]))

        z[0] = self.z0
        z[self.T] = self.zT
        return z

    def init_opt(self):
        adam_params = {"lr": 0.1, "betas": (0.90, 0.999)}
        optimizer = Adam(adam_params)

        self.svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())

pyro.clear_param_store()
system = LatentMovement()
#%%
#system.model()
#system.guide()
#self = system
#t = 1
#%%
system.init_opt()
L = [system.svi.step() for i in range(3000)]
plt.plot(L)
plt.yscale("log")
#%%
traces = [system.guide().detach().unsqueeze(0) for _ in range(100)]
traces = torch.cat(traces)
avgtrace = traces.mean(0)
for i in range(len(traces)):
    plt.plot(traces[i, :,0], traces[i, :,1], alpha = 0.05)
plt.plot(avgtrace[:,0], avgtrace[:,1], color="red")
#plt.plot(pyro.param("z-mu").detach())

# %%
pyro.param("z-scale").mean()

# %% MCMC

from pyro.infer import MCMC, NUTS


nuts_kernel = NUTS(system.model)
#%%
mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=50)

mcmc.run()

# %% VISUALIZE TRACES

hmc_samples = [v.detach().unsqueeze(1) for k, v in mcmc.get_samples().items()]
traces = torch.cat(hmc_samples, dim = 1)
avgtrace = traces.mean(0)
for i in range(len(traces)):
    plt.plot(traces[i, :,0], traces[i, :,1], alpha = 0.05)
plt.plot(avgtrace[:,0], avgtrace[:,1], color="red", alpha = 0.9)

# %%
