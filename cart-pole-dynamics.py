#%%
import gym
import torch
import torch.nn as nn
import utils

#%%
obs_len = 2

data = utils.ReplayBufferTorch(capacity = 1000, obs_dim = (obs_len,))
dl = torch.utils.data.DataLoader(data, batch_size = 50)

#%% play game


class Game:
    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.num_max_steps = 1000
    def play_game(self, render=False):
        obs = self.env.reset()
        episode_data = []
        for t in range(self.num_max_steps):
            if render:
                self.env.render()
            action = self.env.action_space.sample()
            obs_next, reward, done, info = self.env.step(action)
            data = {'obs' : obs, 'action' :  action, 'obs_next' :  obs_next, 'reward' : reward, 'done' : done}
            episode_data.append(data)
            dl.dataset.push(data)
            obs = obs_next
            if done:
                #print(f"Episode finished after {format(t+1)} timesteps")
                break
        return episode_data

    def play_many_games(self):
        num_episode = 100

        for i_episode in range(num_episode):
            self.play_game()
game = Game()
game.play_many_games()


#%%
import torch.distributions.constraints as constraints
from pyro import plate
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


class Network(nn.Module):
    def __init__(self, obs_dim, action_dim = (1,), seed = 1, device="cpu"):
        super(Network, self).__init__()
        h_num = 3
        self.actionemb = nn.Embedding(3, h_num)
        self.enc_obs = nn.Linear(obs_dim[0], h_num)
        self.enc_action = nn.Linear(1,h_num)
        self.h1 = nn.Linear(h_num, h_num)
        self.head_mean = nn.Linear(h_num, obs_dim[0])
        #self.head_scale = nn.Linear(obs_dim[0] + action_dim[0], obs_dim[0])

    def forward(self, batch):
        obs_encoded = torch.relu(self.enc_obs(batch['obs']))
        action_encoded = self.actionemb(batch['action'].long())

        x = obs_encoded + action_encoded
        #x = torch.relu(self.h1(x))
        mu = self.head_mean(x)
        #scale = self.head_scale(x)
        return mu

class StateTransitionModel:
    def __init__(self, obs_dim, action_dim = (1,), seed = 1, device="cpu"):
        super(StateTransitionModel, self).__init__()
        self.data_size = 1000
        self.net = Network(obs_dim, action_dim, seed, device)
        self.svi = self.init_opt()

    def model(self, batch):
        prior = {}
        for name, par in self.net.named_parameters():
            prior[name] = dist.Normal(torch.zeros_like(par), torch.ones_like(par))

        net = pyro.random_module("network", self.net, prior)()

        mu_hat = net(batch) 
        with plate("data", size = self.data_size, subsample = batch['obs']):
            obsdist = dist.Normal(mu_hat, 0.0001*torch.ones_like(mu_hat)).to_event(1)
            pyro.sample("data", obsdist, obs = batch['obs_next']) # _next #"###"

    def guide(self, batch):
        posterior = {}
        for name, par in self.net.named_parameters():
            mean = pyro.param(f"{name}-mu", (torch.rand_like(par)-0.5)*0.01)
            scale = pyro.param(f"{name}-scale", torch.rand_like(par)*0.1, constraint = constraints.positive)
            posterior[name] = dist.Normal(mean, scale)

        net = pyro.random_module("network", self.net, posterior)()
        return net

    def init_opt(self):
        adam_params = {"lr": 0.0001, "betas": (0.90, 0.999)}
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

system = StateTransitionModel(obs_dim = (obs_len,))
pyro.clear_param_store()

#%% TRAIN DYNAMICS

L = []
for _ in range(50):
    loss = system.train_epoch(dl)
    L.append(loss)

import matplotlib.pyplot as plt
plt.plot(L)
plt.yscale("log")

#%%
import matplotlib.pyplot as plt
for i in range(obs_len):
    plt.plot(dl.dataset.data['obs_next'][:,i], color = "green")
    for _ in range(50):
        pred_next_obs = system.guide(None)({'obs' : dl.dataset.data['obs'], 'action' : dl.dataset.data['action'].squeeze()})
        plt.plot((pred_next_obs.detach()[:,i]), alpha = 0.05)
    plt.show()

#%% PREDICT FROM START
num_traces = 50
episode_data = game.play_game(render=False)
num_timesteps = len(episode_data)
traces = [system.guide(None) for _ in range(num_traces)]
pred_states = torch.zeros((num_traces, num_timesteps, obs_len))
obs_states = torch.zeros((1, num_timesteps, obs_len))

for k, net in enumerate(traces):
    for t, dat in enumerate(episode_data[:-1]):
        if (t == 0) | (t==100):
            pred_states[k,t,:] = torch.tensor(dat['obs']).unsqueeze(0).float()
        obs_states[0,t,:] = torch.tensor(dat['obs']).unsqueeze(0).float()
        pred_states[k,t+1,:] = net(
            {'obs' : pred_states[k,t,:], 'action' : torch.tensor(dat['action']).float().unsqueeze(0)}
            )

# PLOT TRAJECTORIES
for i in range(obs_len):
    plt.plot(obs_states[0,:,i], color = "green")
    plt.plot(pred_states.detach()[:,:,i].mean(0), color = "red", alpha = 0.5)
    for k in range(num_traces):
        plt.plot(pred_states.detach()[k,:,i], alpha = 5/num_traces)
    
    plt.ylim(obs_states[0,:,i].min()*2,obs_states[0,:,i].max()*2)
    plt.show()

# %% LEARN STATE DYNAMICS
