# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 16:51:27 2022

@author: nayut
"""
import torch
from torch import nn
from torch.nn import functional as F
def kl_q_p(zs, phi):
    """Given [b,n,k] samples of z drawn from q, compute estimate of KL(q||p).
    phi must be size [b,k+1]

    This uses mu_p = 0 and sigma_p = 1, which simplifies the log(p(zs)) term to
    just -1/2*(zs**2)
    """
    #print("zs shape: ", zs.shape)
    b, n, k = zs.size()
    mu_q, log_sig_q = phi[:,:-1], phi[:,-1]
    log_p = -0.5*(zs**2)
    log_q = -0.5*(zs - mu_q.view(b,1,k))**2 / log_sig_q.exp().view(b,1,1)**2 - log_sig_q.view(b,1,-1)
    # Size of log_q and log_p is [b,n,k]. Sum along [k] but mean along [b,n]
    return (log_q - log_p).sum(dim=2).mean(dim=(0,1))

def log_p_x(x, mu_xs, sig_x):
    """Given [batch, ...] input x and [batch, n, ...] reconstructions, compute
    pixel-wise log Gaussian probability

    Sum over pixel dimensions, but mean over batch and samples.
    """
    #print("x: " , x.shape)
    #print("mu xs: " , mu_xs.shape)
    #print("sig_x: ", sig_x.shape)
    b, n = mu_xs.size()[:2]
    # Flatten out pixels and add a singleton dimension [1] so that x will be
    # implicitly expanded when combined with mu_xs
    x = x.reshape(b, 1, -1)
    _, _, p = x.size()
    squared_error = (x - mu_xs.view(b, n, -1))**2 / (2*sig_x**2)
    #print(squared_error)
    #print(squared_error.shape)
    #print("sigx log: ", torch.log(sig_x))

    # Size of squared_error is [b,n,p]. log prob is by definition sum over [p].
    # Expected value requires mean over [n]. Handling different size batches
    # requires mean over [b].
    return -(squared_error + torch.log(sig_x)).sum(dim=2).mean(dim=(0,1))

def rsample(phi, n_samples):
    """Sample z ~ q(z;phi)
    Ouput z is size [b,n_samples,K] given phi with shape [b,K+1]. The first K
    entries of each row of phi are the mean of q, and phi[:,-1] is the log
    standard deviation
    """
    b, kplus1 = phi.size()
    k = kplus1-1
    mu, sig = phi[:, :-1], phi[:,-1].exp()
    eps = torch.randn(b, n_samples, k, device=phi.device)
    return eps*sig.view(b,1,1) + mu.view(b,1,k)

class ConvVAE(nn.Module):
    WINDOW_LENGTH = 48
    def __init__(self, track_order, latent_size):
        super(ConvVAE, self).__init__()
        self.track_order = track_order
        self.track_num = len(track_order)
        self.enc_conv1 = nn.Conv2d(in_channels = self.track_num, out_channels=64, kernel_size = (4,4), stride = (2, 2))
        self.enc_conv2 = nn.Conv2d(in_channels = 64, out_channels=128, kernel_size = (3,3), stride = (2,2))
        self.enc_conv3 = nn.Conv2d(in_channels = 128, out_channels=256, kernel_size = (3,3), stride = (2,2))
        self.enc_conv4 = nn.Conv2d(in_channels = 256, out_channels=512, kernel_size = (3,3), stride = (2,2))
        self.trans_enc_lin1 = nn.Linear(7168, 512)
        self.trans_enc_lin2 = nn.Linear(512, 256)
        self.trans_enc_lin3 = nn.Linear(256, latent_size + 1)
        h = 2
        w = 7
        self.trans_dec1 = nn.Linear(latent_size, 256)
        self.trans_dec2 = nn.Linear(256, 512)
        self.trans_dec3 = nn.Linear(512, 7168)
        self.dec_conv1 = nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = (3, 3), stride = (2,2))
        self.dec_conv2 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = (3, 3), stride = (2, 2))
        self.dec_conv3 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = (3, 3), stride = (2,2))
        self.dec_conv4 = nn.ConvTranspose2d(in_channels = 64, out_channels = 1, kernel_size = (4, 4), stride = (2, 2))

        # Define a special extra parameter to learn scalar sig_x for all music piece
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, (512, h, w))
        self.log_sig_x = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout(0.5)
        
        self.batch_norm64 = nn.BatchNorm2d(64)
        self.batch_norm128 = nn.BatchNorm2d(128)
        self.batch_norm256 = nn.BatchNorm2d(256)
        self.batch_norm512 = nn.BatchNorm2d(512)
        self.batch_norm_enc1 = nn.BatchNorm1d(512)
        self.batch_norm_enc2 = nn.BatchNorm1d(256)
        self.batch_norm_dec1 = nn.BatchNorm1d(256)
        self.batch_norm_dec2 = nn.BatchNorm1d(512)
    
    def encode(self, x):
        #x in shape (batch_size,  tracks, time_steps, piches)
        o = self.dropout(self.batch_norm64(F.relu(self.enc_conv1(x))))
        #print(o.shape)
        o = self.dropout(self.batch_norm128(F.relu(self.enc_conv2(o))))
        #print(o.shape)
        o = self.dropout(self.batch_norm256(F.relu(self.enc_conv3(o))))
        #print(o.shape)
        o = self.dropout(self.batch_norm512(F.relu(self.enc_conv4(o))))
        #print(o.shape)
        
        o = self.flatten(o)
        #print(o.shape)
        o = self.dropout(self.batch_norm_enc1(self.trans_enc_lin1(o)))
        #print(o.shape)
        o = self.dropout(self.batch_norm_enc2(self.trans_enc_lin2(o)))
        #print(o.shape)
        o = self.trans_enc_lin3(o)
        #print(o.shape)
        return o

    def decode(self, x):
        b,n,k = x.shape
        o = x.view(b*n, -1)
        #print(o.shape)
        o = F.relu(self.trans_dec1(o))
        #print(o.shape)
        o = F.relu(self.trans_dec2(o))
        #print(o.shape)
        o = F.relu(self.trans_dec3(o))
        #print(o.shape)
        o = F.relu(self.unflatten(o))
        #print(o.shape)
        o = F.relu(self.dec_conv1(o))
        #print(o.shape)
        o = F.relu(self.dec_conv2(o))
        #print(o.shape)
        o = F.relu(self.dec_conv3(o))
        #print("?", o.shape)
        o = F.relu(self.dec_conv4(o))
        #print(o.shape)
        o = o.reshape(b,n, self.track_num, self.WINDOW_LENGTH, 128)
        return o
    
    def forward(self, x):
        
        phi = self.encode(x)
        zs = rsample(phi, 1)
        generate_data = self.decode(zs)
        return generate_data

    def elbo(self, x, n=1):
        """Run input end to end through the VAE and compute the ELBO using n
        samples of z
        """
        b_n = x.shape[0]
        phi = self.encode(x)
        #print("phi shape: ", phi.shape)
        zs = rsample(phi, n)
        mu_xs = self.decode(zs).reshape(b_n, n, -1)
        #print("log: ", log_p_x(x, mu_xs, self.log_sig_x.exp()))
        #print("kl: ", kl_q_p(zs, phi))
        return log_p_x(x, mu_xs, self.log_sig_x.exp()) - kl_q_p(zs, phi)
    
    def get_track_order(self):
        return self.track_order
    
