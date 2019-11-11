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

        self.position = 0
        self.cnt_push = 0

        self.data = {
            'obs' :         torch.zeros( (self.capacity,) +self.obs_dim).to(self.device),
            'obs_next' :    torch.zeros( (self.capacity,) +self.obs_dim).to(self.device),
            'action' :      torch.zeros( (self.capacity,) + self.action_dim).to(self.device),
            'reward' :      torch.zeros( (self.capacity,1) ).to(self.device),
            'done' :        torch.zeros( (self.capacity,1) ).to(self.device)
        }

    def __len__(self):
        return min(self.cnt_push, self.capacity)

    def push(self, data):
        """Saves a transition dictionary with same dimensions as data (minus batch dim)."""
        for key, val in self.data.items():
            self.data[key][self.position,] = torch.tensor(data[key]).float().to(self.device)
        
        self.cnt_push +=1
        self.position = (self.position + 1) % self.capacity
    def __getitem__(self, idx):
        return {key : val[idx,] for key, val in self.data.items()}

    #def sample_batch(self, batch_size):
    #    idx = torch.randperm(min(self.cnt_push, self.capacity))[:batch_size]
    #    batch = {key : val[idx,] for key, val in self.data.items()}
    #    return batch



#%% IMAGE TRANSFORM

