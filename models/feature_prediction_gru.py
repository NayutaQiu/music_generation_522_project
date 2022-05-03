# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 18:03:09 2022

@author: nayut
"""
import torch
import torch.nn.functional as F
class FeaturePredictionGRU(torch.nn.Module):
    def __init__(self, track_order, latent_size, hidden_size, batch_size, n_layers, device):
        super().__init__()
        self.latent_size = latent_size
        self.track_order = track_order
        self.number_of_tracks = len(self.track_order)
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.device = device
        self.gru = torch.nn.GRU(self.number_of_tracks*latent_size,
                                hidden_size, n_layers, batch_first = True)
        self.fc1 = torch.nn.Linear(hidden_size , round(hidden_size/2))
        self.bn1 = torch.nn.BatchNorm1d(round(hidden_size/2))
        self.fc2 = torch.nn.Linear(round(hidden_size/2), round(hidden_size*n_layers/2))
        self.bn2 = torch.nn.BatchNorm1d( round(hidden_size*n_layers/2))
        self.fc3 = torch.nn.Linear( round(hidden_size*n_layers/2), latent_size)
    
    def forward(self, x, hidden):
        # before permute x: (batchsize x tracks x seq_length x latent)
        x = x.permute(0, 2, 1, 3)
        # after permute x: (batchsize x seq_length x tracks x latent)
        x = x.flatten(2,3)
        # after flatten x: (batchsize x seq_length x feature)
        output, hidden = self.gru(x, hidden)
        #output (N, L, H_out)
        b_n, seq_len, H_out = output.shape
        output = output.reshape(-1, H_out)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        output = output.reshape(b_n, seq_len, self.number_of_tracks, self.latent_size).permute(0, 2, 1, 3)
        return output, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size).to(self.device)
    
    def get_track_order(self):
        return self.track_order
