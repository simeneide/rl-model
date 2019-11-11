import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch import nn
import torch
import numpy as np
class EncoderConv(nn.Module):
    """ TAKES AND IMAGE AND PRODUCES A Z
    input = tensor of shape [batch_size, channels, h, w]
    output: two tensors mu and sigma in dim z_dim.
    """
    def __init__(self, z_dim, dummy_batch, device = "cpu"):
        super(EncoderConv, self).__init__()
        self.z_dim = z_dim
        self.img_dim = dummy_batch.size()[1:]
        self.img_pixels = np.prod(self.img_dim)
        self.device= device
        print(f"Initializing Encoder with dim(z) = {self.z_dim}")
        self.softplus = nn.Softplus()
        self.fc1 = nn.Linear(self.img_pixels, self.img_pixels)
        # CONVOLUTIONAL LAYERS
        self.layers = nn.ModuleList([])
        channel_layers = [self.img_dim[0], 32, 128, 256]

        for i in range(len(channel_layers)-1):
            l = nn.Conv2d(
                in_channels=channel_layers[i], 
                out_channels = channel_layers[i+1], 
                kernel_size = 4, stride=2).to(self.device)
            print(l)
            self.layers.append(l)

        self.head_dim = None
        self.to(self.device)
        self.forward(dummy_batch)

    def init_linear_layers(self, dim):
        print(f"initialize encoder linear layers: in_channels: {dim}.")
        self.head_dim = dim
        self.linear_layer = nn.Linear(dim, dim).to(self.device)
        self.mu_layer = nn.Linear(dim, self.z_dim).to(self.device)
        self.sigma_layer = nn.Linear(dim, self.z_dim).to(self.device)

    def forward(self, x):
        bs = x.size()[0]
        x = x.view(bs, -1)
        # then compute the hidden units
        x = self.softplus(self.fc1(x))
        x = x.view((bs,)+ self.img_dim)
        
        for i, layer in enumerate(self.layers):
            x = torch.relu(layer(x))

        x = x.view(bs,-1)

        if self.head_dim is None:
            self.init_linear_layers(x.size()[1])
            
        x = self.linear_layer(x)
        mu = self.mu_layer(x)
        logvar = self.sigma_layer(x)
        sigma = torch.exp(logvar)
        return mu, sigma
class DecoderConvBeta(nn.Module):
    def __init__(self, z_dim, dummy_batch, **kwargs):
        super(DecoderConvBeta, self).__init__()
        self.z_dim = z_dim
        self.img_dim = dummy_batch.size()[1:]
        self.img_pixels = np.prod(self.img_dim)

        num_layers = 4# int(np.log(self.img_dim[1]/self.z_dim))+3
        print(f"Initializing {num_layers} deconv layers.")
        self.layers = nn.ModuleList([])
        channel_sizes = [self.z_dim] + list(np.arange(1,num_layers)[::-1]*32) + [self.img_dim[0]]
        print(channel_sizes)
        kernel = [4]*len(channel_sizes)

        self.l1 = nn.Linear(self.z_dim, 4*channel_sizes[0])

        for i in range(len(channel_sizes)-1):
            l = nn.ConvTranspose2d(channel_sizes[i], channel_sizes[i+1], kernel[i], stride=2, padding=1)
            print(l)
            self.layers.append(l)
        
        self.upsample = torch.nn.Upsample(self.img_dim[1:])
        #self.last_conv = nn.Conv2d(in_channels=self.img_dim[0], out_channels = self.img_dim[0], kernel_size = 1, stride=1)
        self.linear_alpha = nn.Linear(self.img_pixels, self.img_pixels)
        self.linear_beta = nn.Linear(self.img_pixels, self.img_pixels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = torch.relu(self.l1(z))
        x = x.view(-1,self.z_dim, 2, 2)
        #print(x.size())

        for l in self.layers:
            x = torch.relu(l(x))
            #print(x.size())

        x = self.upsample(x)
        x = x.view((-1,self.img_pixels))
        alpha = self.sigmoid(self.linear_alpha(x))
        beta = self.sigmoid(self.linear_beta(x))
        alpha = alpha.view((-1,)+ self.img_dim)
        beta = beta.view((-1,)+ self.img_dim)
        return alpha, beta

class DecoderConv(nn.Module):
    def __init__(self, z_dim, dummy_batch, **kwargs):
        super(DecoderConv, self).__init__()
        self.z_dim = z_dim
        self.img_dim = dummy_batch.size()[1:]
        self.img_pixels = np.prod(self.img_dim)

        num_layers = 4# int(np.log(self.img_dim[1]/self.z_dim))+3
        print(f"Initializing {num_layers} deconv layers.")
        self.layers = nn.ModuleList([])
        channel_sizes = [self.z_dim] + list(np.arange(1,num_layers)[::-1]*32) + [self.img_dim[0]]
        print(channel_sizes)
        kernel = [4]*len(channel_sizes)

        self.l1 = nn.Linear(self.z_dim, 4*channel_sizes[0])

        for i in range(len(channel_sizes)-1):
            l = nn.ConvTranspose2d(channel_sizes[i], channel_sizes[i+1], kernel[i], stride=2, padding=1)
            print(l)
            self.layers.append(l)
        
        self.upsample = torch.nn.Upsample(self.img_dim[1:])
        #self.last_conv = nn.Conv2d(in_channels=self.img_dim[0], out_channels = self.img_dim[0], kernel_size = 1, stride=1)
        self.last_linear = nn.Linear(self.img_pixels, self.img_pixels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = torch.relu(self.l1(z))
        x = x.view(-1,self.z_dim, 2, 2)
        #print(x.size())

        for l in self.layers:
            x = torch.relu(l(x))
            #print(x.size())

        x = self.upsample(x)
        x = x.view((-1,self.img_pixels))
        x = self.last_linear(x)
        x = x.view((-1,)+ self.img_dim)
        x = self.sigmoid(x)
        return x


class DecoderFC(nn.Module):
    def __init__(self, z_dim, dummy_batch, hidden_dim = 1000, device = "cpu"):
        super(DecoderFC, self).__init__()
        self.img_shape = dummy_batch.size()[1:]
        img_pixels = np.prod(dummy_batch.size()[1:])
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)

        
        self.fc21 = nn.Linear(hidden_dim, img_pixels)
        # setup the non-linearities
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        print(f"initialized decoder with 2 layers, hidden_dim={hidden_dim}, output_shape: {self.img_shape}")

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x img_pixels
        loc_img = self.sigmoid(self.fc21(hidden))
        loc_img = loc_img.view((-1,) + self.img_shape)
        return loc_img

class EncoderFC(nn.Module):
    def __init__(self, z_dim, dummy_batch, hidden_dim = 1000, device = "cpu"):
        super(EncoderFC, self).__init__()
        # setup the three linear transformations used
        self.img_shape = dummy_batch.size()[1:]
        img_pixels = np.prod(dummy_batch.size()[1:])
        self.fc1 = nn.Linear(img_pixels, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        print(f"initialized encoder with 2 layers, hidden_dim={hidden_dim}, input_shape: {self.img_shape}")

        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.view(x.size()[0], -1)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale

class VAE_FC(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, z_dim, dummy_batch, device="cpu", lr = 1e-3):
        super(VAE_FC, self).__init__()
        self.z_dim = z_dim
        self.dummy_batch = dummy_batch
        self.device = device
        # create the encoder and decoder networks
        self.encoder = EncoderConv(z_dim = z_dim, dummy_batch = dummy_batch, device = self.device)#.to(self.device)
        self.decoder = DecoderConvBeta(z_dim = z_dim, dummy_batch = dummy_batch, device = self.device).to(self.device)

        self.to(self.device)    

        self.init_opt(lr)

    def init_opt(self, lr):
        optimizer = Adam({"lr": lr})
        self.svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO(), num_samples= 10000)    

    def step(self, batch):
        return self.svi.step(batch)

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", size=10000, subsample=x):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            alpha, beta = self.decoder.forward(z)
            # score against actual images
            obsdist = pyro.sample("obs", dist.Beta(alpha, beta).to_event(1).to_event(2), 
                obs=x)

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", subsample = x, size = int(1e7)):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        alpha, beta = self.decoder(z)
        return alpha / (alpha+beta)

class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, z_dim, dummy_batch, device="cpu", lr = 1e-3):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.dummy_batch = dummy_batch
        self.device = device
        # create the encoder and decoder networks
        self.encoder = EncoderConv(z_dim = z_dim, dummy_batch = dummy_batch, device = self.device)#.to(self.device)
        self.decoder = DecoderConvBeta(z_dim = z_dim, dummy_batch = dummy_batch).to(self.device)

        self.to(self.device)    

        self.init_opt(lr)

    def init_opt(self, lr):
        optimizer = Adam({"lr": lr})
        self.svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO(), num_samples= 10000)    
    def step(self, batch):
        return self.svi.step(batch)

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", size=int(1e6), subsample=x):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            mu, sd = self.decoder.forward(z)
            alpha = mu / (sd**2)
            beta = (1-mu) / (sd**2)
            # score against actual images
            obsdist = pyro.sample("obs", dist.Beta(alpha, beta).to_event(1).to_event(2), 
                obs=x)

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        mu, sd = self.decoder(z)
        
        return mu
