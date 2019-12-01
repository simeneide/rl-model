#%%
import gym
import torch
import torch.nn as nn
import utils
from pyro import poutine
from pyro.poutine.util import prune_subsample_sites
import warnings
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

class StateTransition(PyroModule):
    def __init__(self, obs_dim, action_dim = (1,), seed = 1, device="cpu"):
        super(StateTransition, self).__init__()
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
world_model = StateTransition(obs_dim = (4,))
world_model.init_opt()

#%% TRAIN DYNAMICS
L = []
for _ in range(100):
    loss = world_model.train_epoch(dl)
    L.append(loss)

import matplotlib.pyplot as plt
plt.plot(L)
plt.yscale("log")
plt.show()


#%% PREDICT FROM START
with torch.no_grad():
    num_traces = 50
    ep_data = game.play_game(render=False)
    num_timesteps = len(ep_data['obs'])

    posterior_traces = [world_model.sample_guide_trace() for _ in range(num_traces)]

    pred_states = torch.zeros((num_traces, num_timesteps, obs_len))
    obs_states = torch.zeros((1, num_timesteps, obs_len))

    for s, trace in enumerate(posterior_traces):
        for t in range(num_timesteps-1):
            if (t == 0) | (t==100):
                pred_states[s,t,:] = ep_data['obs'][t,].unsqueeze(0).float()
            obs_states[0,t,:] = ep_data['obs'][t,].unsqueeze(0).float()

            state_action = {
                'obs' : pred_states[s,t,:].unsqueeze(0), 
                'action' : ep_data['action'][t].unsqueeze(0), 
                'obs_next' :None}

            pred_states[s,t+1,:] = world_model.predict(state_action, trace).squeeze()

    # PLOT TRAJECTORIES
    for i in [0,2]:
        plt.plot(obs_states[0,:,i], color = "green")
        plt.plot(pred_states.detach()[:,:,i].mean(0), color = "red", alpha = 0.5)
        for k in range(num_traces):
            plt.plot(pred_states.detach()[k,:,i], alpha = 5/num_traces)
        
        
        # plot size costmetics:
        r = obs_states[0,:,i].max()-obs_states[0,:,i].min()
        plt.ylim(obs_states[0,:,i].min()-r/2,obs_states[0,:,i].max()+r/2)
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