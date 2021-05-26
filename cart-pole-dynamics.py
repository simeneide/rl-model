#%%
import gym
import torch
import torch.nn as nn
import utils
from pyro import poutine
from pyro.poutine.util import prune_subsample_sites
import warnings
#%%
obs_len = 2
env_name = "MountainCarContinuous-v0" 
# [position of cart, velocity of cart, angle of pole, rotation rate of pole]
data = utils.ReplayBufferTorch(capacity = 10001, obs_dim = (obs_len,))
dl = torch.utils.data.DataLoader(data, batch_size = 100)

#%% Play Game
class Game:
    def __init__(self, device = "cpu"):
        self.env = gym.make(env_name)
        self.num_max_steps = 1000
        self.device = device
        self.obs_dim = (obs_len,)
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
        #return episode_data
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
#game.play_many_games()

ep_data = game.play_game(render=False)
ep_data['action'].size()
len(dl.dataset)
#episode_data = game.play_game(render=True)

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


class StateTransitionCont(PyroModule):
    def __init__(self, obs_dim, action_dim = (1,), seed = 1, device="cpu"):
        super(StateTransitionCont, self).__init__()
        self.num_data = 1000
        h_num = 10
        self.enc_action = PyroModule[nn.Linear](action_dim[0], h_num)
        set_noninform_prior(self.enc_action)

        self.enc_obs = PyroModule[nn.Linear](obs_dim[0], h_num)
        set_noninform_prior(self.enc_obs)

        self.linear1 = PyroModule[nn.Linear](h_num, h_num)
        set_noninform_prior(self.linear1)

        self.head_mean = PyroModule[nn.Linear](h_num, obs_dim[0])
        set_noninform_prior(self.head_mean)

        self.guide = None

    def forward(self, batch):
        obs_encoded = torch.tanh(self.enc_obs(batch['obs']))
        action_encoded = torch.tanh(self.enc_action(batch['action']))#.squeeze()

        x = obs_encoded + action_encoded
        x = torch.relu(self.linear1(x))
        mu = self.head_mean(x)
        return mu + batch['obs']

    def model(self, batch):
        mu_hat = self.forward(batch)
        with plate("data", size = self.num_data, subsample = batch['obs']):
            obsdist = dist.Normal(mu_hat, 0.0001*torch.ones_like(mu_hat)).to_event(1)
            pyro.sample("data", obsdist, obs = batch['obs_next'])
        return mu_hat

    def build_guide(self):
        self.guide = AutoDiagonalNormal(self.model)

    def init_opt(self):
        if self.guide is None:
            self.build_guide()
        adam_params = {"lr": 0.005, "betas": (0.90, 0.999)}
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

    def sample_guide_trace(self, batch=None):
        return poutine.trace(self.guide).get_trace(batch)

    def predict(self, batch, guide_trace = None):
        if guide_trace is None:
            guide_trace = self.sample_guide_trace(batch)

        model_trace = poutine.trace(poutine.replay(self.model, guide_trace)).get_trace(batch)
        return model_trace.nodes['_RETURN']['value']

class StateTransition(PyroModule):
    def __init__(self, obs_dim, action_dim = (1,), seed = 1, device="cpu"):
        super(StateTransition, self).__init__()
        self.num_data = 1000
        h_num = 10
        self.actionemb = PyroModule[nn.Embedding](3, h_num)
        set_noninform_prior(self.actionemb)

        self.enc_obs = PyroModule[nn.Linear](obs_dim[0], h_num)
        set_noninform_prior(self.enc_obs)

        self.h1 = PyroModule[nn.Linear](h_num, h_num)
        set_noninform_prior(self.h1)


        self.h2 = PyroModule[nn.Linear](h_num, h_num)
        set_noninform_prior(self.h2)

        self.head_mean = PyroModule[nn.Linear](h_num, obs_dim[0])
        set_noninform_prior(self.head_mean)

        self.guide = None

    def forward(self, batch):
        obs_encoded = torch.relu(self.enc_obs(batch['obs']))
        action_encoded = self.actionemb(batch['action'].long()).squeeze()

        x = obs_encoded + action_encoded
        x = self.h1(x)
        x = self.h2(x)
        mu = self.head_mean(x)
        return mu

    def model(self, batch):
        mu_hat = self.forward(batch)
        with plate("data", size = self.num_data, subsample = batch['obs']):
            obsdist = dist.Normal(mu_hat, 0.0001*torch.ones_like(mu_hat)).to_event(1)
            pyro.sample("data", obsdist, obs = batch['obs_next'])
        return mu_hat

    def build_guide(self):
        self.guide = AutoDiagonalNormal(self.model)

    def init_opt(self):
        if self.guide is None:
            self.build_guide()
        adam_params = {"lr": 0.01, "betas": (0.90, 0.999)}
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

    def sample_guide_trace(self, batch=None):
        return poutine.trace(self.guide).get_trace(batch)

    def predict(self, batch, guide_trace = None):
        if guide_trace is None:
            guide_trace = self.sample_guide_trace(batch)

        model_trace = poutine.trace(poutine.replay(self.model, guide_trace)).get_trace(batch)
        return model_trace.nodes['_RETURN']['value']

#%%
pyro.clear_param_store()
pyro.enable_validation(True)
world_model = StateTransitionCont(obs_dim = (obs_len,))
world_model.init_opt()

#%% TRAIN DYNAMICS
L = []
for _ in range(600):
    loss = world_model.train_epoch(dl)
    L.append(loss)

import matplotlib.pyplot as plt
plt.plot(L)
plt.yscale("log")
plt.show()


#%% PREDICT FROM START
with torch.no_grad():
    num_traces = 30
    ep_data = game.play_game(render=False)
    num_timesteps = len(ep_data['obs'])

    posterior_traces = [world_model.sample_guide_trace() for _ in range(num_traces)]

    pred_states = torch.zeros((num_traces, num_timesteps, obs_len))
    obs_states = torch.zeros((1, num_timesteps, obs_len))

    for s, trace in enumerate(posterior_traces):
        for t in range(num_timesteps-1):
            if (t )==0:
                pred_states[s,t,:] = ep_data['obs'][t,].unsqueeze(0).float()
            obs_states[0,t,:] = ep_data['obs'][t,].unsqueeze(0).float()

            state_action = {
                'obs' : pred_states[s,t,:].unsqueeze(0), 
                'action' : ep_data['action'][t].unsqueeze(0), 
                'obs_next' :None}

            pred_states[s,t+1,:] = world_model.predict(state_action, trace).squeeze().clamp(-10,10)

    # PLOT TRAJECTORIES
    for i in [0,1]:
        plt.plot(obs_states[0,:,i], color = "green")
        plt.plot(pred_states.detach()[:,:,i].mean(0), color = "red", alpha = 0.5)
        for k in range(num_traces):
            plt.plot(pred_states.detach()[k,:,i], alpha = 5/num_traces)
        
        
        # plot size costmetics:
        r = obs_states[0,:,i].max()-obs_states[0,:,i].min()
        plt.ylim(obs_states[0,:,i].min()-r/2,obs_states[0,:,i].max()+r/2)
        #plt.xlim(0,300)
        plt.show()

#%% COST
def state_cost(obs):
    T = len(obs)
    pos_cost = (obs[:,:,0]/2.4)**2
    angle_cost = (obs[:,:,2]/0.15)**2
    return pos_cost + angle_cost

pred_cost = state_cost(pred_states).t()
true_cost = state_cost(obs_states).t()
plt.plot(true_cost, color="green")
_ = plt.plot(pred_cost, alpha = 0.1)
_ = plt.plot(pred_cost.mean(1), color="red")
plt.ylim( true_cost.min()-2, true_cost.max()+2)

#%% Find optimal policy given world model
class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim = (1,), num_actions=3, device = "cpu"):
        super(Policy, self).__init__()
        h_num = 3
        self.obs2h = nn.Linear(obs_dim[0], h_num)
        self.h2action = nn.Linear(h_num, num_actions)

    def forward(self, x):
        x = torch.relu(self.obs2h(x))
        x = self.h2action(x)
        x = nn.Softmax(-1)(x)
        return x

        
policy = Policy(obs_dim = (4,))
ep_data['obs'].size()
policy(ep_data['obs'])

# %% Play simulated game

num_traces = 2
traces = [world_model.sample_guide_trace() for _ in range(num_traces)]
states = torch.zeros((num_traces, num_timesteps, obs_len))
# NB remember to sample starting positions over num_traces here (can we just take some random ones from memory?)

s = 0

#for t in range(num_timesteps):
t = 0

cur_state = states[:,t,:]#.unsqueeze(0)
cur_state.size()
action_probs = policy(cur_state)
action_probs.argmax(1).unsqueeze(0).size()
world_model.predict({'obs' : states[0,t,:].unsqueeze(0), 'action' : })

# %%
