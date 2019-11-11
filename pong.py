#%%
import setGPU
import DATAHelper
import os
if DATAHelper.in_ipynb():
    try:
        os.chdir(os.path.join(os.getcwd(), 'personal-scratch/atari-vae'))
        print(os.getcwd())
    except:
        pass
import gym
import numpy as np
import utils
import torch
from torch import nn
from torchvision import transforms

#%%

class Game:
    def __init__(self, device = "cpu"):
        self.device = device

        self.process_image = transforms.Compose([
            transforms.Lambda(lambda x: x[34:194,]),
            transforms.ToPILImage(),
            transforms.Resize((50,50)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mean(0, keepdims=True))
        ])
        self.env = gym.make("Pong-v0")
        self.obs_dummy = self.process_image(self.env.reset())
        self.replay_buffer = utils.ReplayBufferTorch(capacity = 5000, obs_dim = self.obs_dummy.size(), device = self.device)

    def play_game(self, render=False):
        obs = self.env.reset()
        obs = self.process_image(obs)
        #print(obs)
        cum_reward = 0
        obslist = [obs]

        for t in range(5000):
            # Policy
            action = self.env.action_space.sample()

            # Take action, see future and save to memory:
            obs_next, reward, done, info = self.env.step(action)
            obs_next = self.process_image(obs_next)
            obslist.append(obs_next)
            cum_reward += reward
            data = {'obs' : obs,'action' :  action, 'obs_next' :  obs_next, 'reward' : reward, 'done' : done}
            self.replay_buffer.push(data)
            #print(data['obs'])

            # Update observation and check if done:
            obs = obs_next
            if done:
                break

        #print(f"t= {t}, reward={cum_reward}: game done")
        out = {'tot_reward' : cum_reward}
        if render:
            out['frames'] = obslist
        return out
#%%
device = "cuda"
game = Game(device=device)
for _ in range(1):
    out = game.play_game(render=True)

#game.replay_buffer.data['obs'][0]
out = game.play_game(render=True)
#utils.animate_game(out['frames'])
#game.replay_buffer.sample_batch(10)


#%% VAE
import pyro
import variational_autoencoder
from importlib import reload
reload(variational_autoencoder)
batch = game.replay_buffer.sample_batch(512)
pyro.clear_param_store()
vae = variational_autoencoder.VAE(
    z_dim = 100, 
    dummy_batch = batch['obs'], 
    device="cuda", 
    lr = 1e-3).to(device)

#%%
vae.step(batch['obs'])

#%%
import matplotlib.pyplot as plt
def plt_tensorimg(x):
    plt.imshow(x.permute(1,2,0).squeeze().detach().cpu().numpy())

def plot_reconstructed_img(img):
    gen_img = vae.reconstruct_img(img).squeeze()
    concat = torch.cat((img.squeeze(), gen_img), dim =1)
    plt.imshow(concat.detach().cpu().numpy())

vae.init_opt(lr=1e-3)
for ep in range(1000):
    with torch.no_grad():
        game.play_game()
        batch = game.replay_buffer.sample_batch(4096)

    #step: 
    loss = vae.step(batch['obs'])
    if (ep%5)==0:
        print(f"ep : {ep} \t -elbo: {loss/len(batch)/1e6:.3f}")
    if (ep%30==0):
        plot_reconstructed_img(batch['obs'][:1])
        plt.show()

    if ((ep%100==0) & (ep!=0)):
        out = game.play_game(render=True)
        frames = vae.reconstruct_img(torch.cat(out['frames'][:300]).to(device))
        utils.animate_game(frames)
#%%
