# Imports specifically so we can render outputs in Jupyter.
#from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display
import matplotlib.pyplot as plt

import os
import argparse
import gym
import time
import random
import numpy as np
from collections import  namedtuple, deque
from torch.utils.data import Dataset
import torch

from matplotlib import animation
from IPython.display import display
import matplotlib.pyplot as plt
def animate_game(frames):
    
    """
    Displays a list of frames as a gif, with controls
    """
    frames = [x.permute(1,2,0).squeeze().detach().cpu().numpy() for x in frames]
    #plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    display(display_animation(anim, default_mode='loop'))

#%% REPLAY BUFFER

class ReplayBufferTorch(torch.utils.data.Dataset):
    def __init__(self, capacity, obs_dim, action_dim = (1,), seed = 1, device="cpu"):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.seed = seed
        self.device = device # possibility to put data on gpu for speedup

        self.pos = 0
        self.cur_size = 0

        self.data = {
            'obs' :         torch.zeros( (self.capacity,) +self.obs_dim).to(self.device),
            'obs_next' :    torch.zeros( (self.capacity,) +self.obs_dim).to(self.device),
            'action' :      torch.zeros( (self.capacity,) + self.action_dim).to(self.device),
            'reward' :      torch.zeros( (self.capacity,1) ).to(self.device),
            'done' :        torch.zeros( (self.capacity,1) ).to(self.device)
        }

    def __len__(self):
        return min(self.cur_size, self.capacity)

    def push(self, ep_data):
        """ Saves a episode of batch of users to the dataset """
        with torch.no_grad():
            bs = len(ep_data['obs'])
            start = self.pos
            end = (self.pos+bs)

            # If at end of batch, clip first steps of episode:
            if end >= self.capacity:
                avail = self.capacity-start
                for key, val in ep_data.items():
                    ep_data[key] = ep_data[key][-avail]
                end = self.capacity
            
            for key, val in self.data.items():
                self.data[key][start:end,] = ep_data[key].float().to(self.device)
            
            
            self.cur_size += bs
            self.pos = (self.pos + bs) % self.capacity

    def __getitem__(self, idx):
        return {key : val[idx,] for key, val in self.data.items()}

#%% IMAGE TRANSFORM

