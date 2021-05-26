
#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import pyro
import pyro.distributions as dist
import torch.distributions.constraints as constraints
from pyro import plate
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
# %% States are random walks
with torch.no_grad():
    num_data = 20
    T = 12
    d_latent = 2
    d_obs = 2
    num_actions = 5
    noise_obs = 0.00
    noise_action = 0.05
    # nothing, left, right, up, down:
    a_vec_space = torch.tensor([[0,0], [-1,0], [1,0], [0,1], [0,-1]])

    ### Generate data ###
    actions = torch.randint(num_actions, size = (num_data,T))
    epsilon = torch.randn((num_data, T, d_latent))*noise_action
    action_vec = a_vec_space[actions] + epsilon

    st = torch.zeros((num_data, T, d_obs))
    st[:,1:] = action_vec[:,:-1].cumsum(1)
    st[0,:]
    plt.plot(st[0,:,0],st[0,:,1])

    state_to_obs = nn.Linear(d_latent, d_obs, bias=False)
    X_clean = st #state_to_obs(st).detach()

    X = X_clean + torch.randn_like(X_clean)*noise_obs

    plt.plot(X[0,:,0], X[0,:,1])
#%%
class Network(nn.Module):
    def __init__(self, d_latent, d_obs):
        super(Network, self).__init__()

        self.linear = nn.Linear(d_obs, d_latent, bias=False)

        self.a_emb = nn.Embedding(
            num_embeddings = num_actions,
            embedding_dim = d_latent)
    
    def forward(self, x, a):
        
        action_emb = self.a_emb(a)
        #print(x.size(), a.size(), action_emb.size())
        return self.linear(x) + action_emb

class System:
    def __init__(self, d_latent, d_obs):
        self.net = Network(d_latent, d_obs)

    def model(self, X, actions):
        # PRIOR
        prior = {}
        for name, par in self.net.named_parameters():
            prior[name] = dist.Uniform(-2*torch.ones_like(par), 2*torch.ones_like(par)) #dist.Normal(torch.zeros_like(par), 2*torch.ones_like(par))

        # LIKELIHOOD
        sampled_model = pyro.random_module("network", self.net, prior = prior)()

        st_next = sampled_model(X, actions)
        diff = st_next[:,1:] - st_next[:,:-1]
        diff_dist = dist.Normal(diff, 0.001*torch.ones_like(st_nextz))

        if observe_state:
            st_next = st_next[:,:-1] # skip latest prediction as we dont have data
            st_next_dist = dist.Normal(st_next, 0.001*torch.ones_like(st_next))
            pyro.sample("data", st_next_dist, obs = st[:,1:])

    def guide(self, X, actions):
        posterior = {}
        for name, par in self.net.named_parameters():
            mean = pyro.param(f"{name}-mu", (torch.rand_like(par)-0.5)*0.1)
            scale = pyro.param(f"{name}-scale", torch.rand_like(par)*0.1, constraint = constraints.positive)
            posterior[name] = dist.Normal(mean, scale)

        net = pyro.random_module("network", self.net, posterior)()
        return net

    def init_opt(self):
        adam_params = {"lr": 0.01, "betas": (0.90, 0.999)}
        optimizer = Adam(adam_params)

        self.svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
#%%
pyro.clear_param_store()

system = System(d_latent, d_obs)
system.init_opt()
#self = system
#%%
num_steps = 200
L = torch.zeros((num_steps,))
for i in range(num_steps):
    L[i] = system.svi.step(X,actions)

plt.plot(L)
plt.yscale("log")
#%%
print((pyro.param('a_emb.weight-mu')*10).round()/10), print(a_vec_space)
#%%
pyro.param("linear.weight-mu")
#%%
from matplotlib.collections import LineCollection
net = system.guide(X, actions)
s_pred = system.guide(X, actions)(X,actions).detach()[:,:-1]
s_pred = net(X,actions).detach()[:,:-1]
s_truth = st[:,1:]

s_pred.size()
n = 1
plt.plot(s_truth[n,:,0], s_truth[n,:,1])
plt.plot(s_pred[n,:,0], s_pred[n,:,1])
plt.xlim((-2,2))
plt.show()
#%%
for i in range(d_latent):
    plt.plot(s_truth[n,:,i], color = "green")
    plt.ylim((-2,2))
    for _ in range(50):
        s_pred = (system.guide(X, actions)(X,actions)).detach()[:,:-1]
        
        plt.plot(s_pred[n,:,i], alpha = 0.2)
    plt.show()

#%%

#%% MCMC
from pyro.infer import NUTS, MCMC, EmpiricalMarginal
nuts_kernel = NUTS(system.model, adapt_step_size=True)
mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=30)
mcmc_run = mcmc.run(X, actions)
#%%
hmc_samples = {k : v.detach().unsqueeze(1) for k, v in mcmc.get_samples().items()}
for key,val in hmc_samples.items():
    print(key)
#%%
#hmc_samples["network$$$a_emb.weight"][0]
hmc_samples["network$$$a_emb.weight"].mean(0)
#%%
for a in range(num_actions):
    plt.plot(hmc_samples["network$$$a_emb.weight"][:,0,a,:])

plt.plot(hmc_samples["network$$$a_emb.weight"][:,0,2,1])
#%%

plt.plot(st[0,:,0],st[0,:,1])
plt.plot(st_pred[0,:,0],st_pred[0,:,1])

plt.plot(st_pred[])
# %%
