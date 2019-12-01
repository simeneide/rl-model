#%%
import gym
import torch
import torch.nn as nn
import utils

#%%
obs_len = 4
env_name = "CartPole-v1"
# [position of cart, velocity of cart, angle of pole, rotation rate of pole]
data = utils.ReplayBufferTorch(capacity = 1000, obs_dim = (obs_len,))
dl = torch.utils.data.DataLoader(data, batch_size = 50)

#%% Play Game
class Game:
    def __init__(self, device = "cpu"):
        self.env = gym.make(env_name)
        self.num_max_steps = 1000
        self.device = device
        self.obs_dim = (4,)
        self.action_dim = (1,)
    def play_game(self, render=False):
        obs = self.env.reset()
        episode_data = {
            'obs' :         torch.zeros( (self.num_max_steps,) + self.obs_dim).to(self.device),
            'obs_next' :    torch.zeros( (self.num_max_steps,) + self.obs_dim).to(self.device),
            'action' :      torch.zeros( (self.num_max_steps,) + self.action_dim).to(self.device),
            'reward' :      torch.zeros( (self.num_max_steps,1) ).to(self.device),
            'done' :        torch.zeros( (self.num_max_steps,1) ).to(self.device)
        }
        for t in range(self.num_max_steps):
            if render:
                self.env.render()
            action = self.env.action_space.sample()
            obs_next, reward, done, info = self.env.step(action)
            dat = {'obs' : obs, 'action' :  action, 'obs_next' :  obs_next, 'reward' : reward, 'done' : done}
            for key, val in episode_data.items():
                episode_data[key][t,] = torch.tensor(dat.get(key))

            obs = obs_next
            if done:
                #print(f"Episode finished after {format(t+1)} timesteps")
                break
        
        # Prune tensors to be as long as number of timesteps played:
        for key, val in episode_data.items():
            episode_data[key] =episode_data[key][:(t+1),]

        # Save data to dataset:
        dl.dataset.push(episode_data)
        return episode_data

    def play_many_games(self):
        num_episode = 100

        for i_episode in range(num_episode):
            self.play_game()
game = Game()
game.play_many_games()


len(dl.dataset)
#episode_data = game.play_game(render=True)
#%%
def state_cost(episode_data):
    T = len(episode_data['obs'])
    pos_cost = (episode_data['obs'][:,0]/2.4)**2
    angle_cost = (episode_data['obs'][:,2]/0.15)**2
    return pos_cost + angle_cost

import matplotlib.pyplot as plt
state_cost(dl.dataset.data).mean()
plt.plot(state_cost(dl.dataset.data))

#%%
import torch.distributions.constraints as constraints
from pyro import plate
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.nn import PyroSample, PyroModule
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal, init_to_feasible
from pyro.infer import Predictive
def set_noninform_prior(mod, scale = 1.0):
    for key, par in list(mod.named_parameters()):
        setattr(mod, key, PyroSample(dist.Normal(torch.zeros_like(par), scale*torch.ones_like(par)).independent() ))

class Network(PyroModule):
    def __init__(self, obs_dim, action_dim = (1,), seed = 1, device="cpu"):
        super(Network, self).__init__()
        self.num_data = 1000
        h_num = 3
        self.actionemb = PyroModule[nn.Embedding](3, h_num)
        set_noninform_prior(self.actionemb)

        self.enc_obs = PyroModule[nn.Linear](obs_dim[0], h_num)
        set_noninform_prior(self.enc_obs)

        self.head_mean = PyroModule[nn.Linear](h_num, obs_dim[0])
        set_noninform_prior(self.head_mean)

        self.guide = None
    def forward(self, batch):
        obs_encoded = torch.relu(self.enc_obs(batch['obs']))
        action_encoded = self.actionemb(batch['action'].long()).squeeze()

        x = obs_encoded + action_encoded
        mu = self.head_mean(x)
        return mu

    def model(self, batch):
        mu_hat = self.forward(batch)
        with plate("data", size = self.num_data, subsample = batch['obs']):
            obsdist = dist.Normal(mu_hat, 0.0001*torch.ones_like(mu_hat)).to_event(1)
            pyro.sample("data", obsdist, obs = batch['obs_next']) # _next #"###"
        return mu_hat

    def build_guide(self):
        self.guide = AutoDiagonalNormal(self.model)

    def init_opt(self):
        if self.guide is None:
            self.build_guide()
        adam_params = {"lr": 0.001, "betas": (0.90, 0.999)}
        optimizer = Adam(adam_params)

        self.svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())

    def train_epoch(self, dataloader):
        if self.svi is None:
            self.init_opt()
        
        totloss = 0
        for i, batch in enumerate(dataloader):
            loss = self.svi.step(batch)
            totloss+= loss
        return totloss/i
#%%
pyro.clear_param_store()
pyro.enable_validation(True)
system = Network(obs_dim = (4,))
system.init_opt()


#%%
batch = next(iter(dl))
self = system
#%% TRAIN DYNAMICS

L = []
for _ in range(100):
    loss = system.train_epoch(dl)
    L.append(loss)

import matplotlib.pyplot as plt
plt.plot(L)
plt.yscale("log")
plt.show()
# %% QUANTILES OF PARAMETERS
#for key, val in system.guide.quantiles([0.25,0.5,0.75]).items():
#    for v in val:
#        print(f"{key}: \t {v.detach().numpy()}")

#%% Prediction
predictive = Predictive(model=system.model, guide = system.guide, num_samples=50, return_sites=("_RETURN",))

dl.dataset.data
pred_samples = predictive(dl.dataset.data)['_RETURN'].detach()

pred_samples.size()
len(pred_samples)
#%%
import matplotlib.pyplot as plt
for i in range(obs_len):
    plt.plot(dl.dataset.data['obs_next'][:,i], color = "green")
    for s in range(len(pred_samples)):
        plt.plot((pred_samples[s,:,i]), alpha = 0.05)
    plt.show()

#%% PREDICT FROM START
num_traces = 50
ep_data = game.play_game(render=False)
num_timesteps = len(ep_data['obs'])
pred = predictive(ep_data)['_RETURN'].detach()

pred_samples.size()
num_timesteps
# traces = [system.guide(None) for _ in range(num_traces)]

pred_states = torch.zeros((num_traces, num_timesteps, obs_len))
obs_states = torch.zeros((1, num_timesteps, obs_len))


for s in range(num_traces):
    for t in range(num_timesteps-1):
        if (t == 0) | (t==100):
            pred_states[s,t,:] = ep_data['obs'][t,].unsqueeze(0).float()
        obs_states[0,t,:] = ep_data['obs'][t,].unsqueeze(0).float()
        pred_states[s,t+1,:] = pred_samples[s,t,:]

# PLOT TRAJECTORIES
for i in range(obs_len):
    plt.plot(obs_states[0,:,i], color = "green")
    plt.plot(pred_states.detach()[:,:,i].mean(0), color = "red", alpha = 0.5)
    for k in range(num_traces):
        plt.plot(pred_states.detach()[k,:,i], alpha = 5/num_traces)
    
    plt.ylim(obs_states[0,:,i].min()*2,obs_states[0,:,i].max()*2)
    plt.show()
# %% LEARN STATE DYNAMICS
