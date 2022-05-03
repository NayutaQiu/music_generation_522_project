"""
Created on Fri Apr 22 18:03:09 2022

@author: nayutaQiu
"""
from torch import nn
import torch.nn.functional as F
class MusicClassificationCNN(nn.Module):
    def __init__(self, track_order, number_of_classes, linear):
        super().__init__()
        self.track_order = track_order
        self.linear = linear
        self.input_channel = len(self.track_order)
        self.number_of_classes = number_of_classes
        self.number_of_tracks = len(self.track_order)
        self.conv1 = nn.Conv2d(in_channels = self.number_of_tracks, out_channels = self.number_of_tracks * 5, kernel_size = (8,2), stride=(8,2))
        self.conv2 = nn.Conv2d(in_channels = self.number_of_tracks * 5, out_channels = self.number_of_tracks * 5*2, kernel_size = (8,2), stride=(8,2))
        self.conv3 = nn.Conv2d(in_channels = self.number_of_tracks * 5*2, out_channels = self.number_of_tracks * 5*(2**2), kernel_size = (3,3), stride=(3,3))
        self.conv4 = nn.Conv2d(in_channels = self.number_of_tracks * 5*(2**2), out_channels = self.number_of_tracks * 5*(2**3), kernel_size = (3,3))
        self.conv5 = nn.Conv2d(in_channels = self.number_of_tracks * 5*(2**3), out_channels = self.number_of_tracks * 5*(2**4), kernel_size = (3,3), stride=(3,3))
        self.fc1 = nn.Linear(self.number_of_tracks * 1280, self.number_of_tracks * 480)
        self.fc2 = nn.Linear(self.number_of_tracks * 480, 120)
        self.fc3 = nn.Linear( 120, self.number_of_classes)
        self.flatten = nn.Flatten()
        self.dropout2d1 = nn.Dropout2d(0.4)
        self.dropout2d2 = nn.Dropout2d(0.4)
        self.dropout2d3 = nn.Dropout2d(0.4)
        self.dropout2d4 = nn.Dropout2d(0.4)
        self.batchnorm2d1 = nn.BatchNorm2d(self.number_of_tracks * 5)
        self.batchnorm2d2 = nn.BatchNorm2d(self.number_of_tracks * 5*2)
        self.batchnorm2d3 = nn.BatchNorm2d(self.number_of_tracks * 5*(2**2))
        self.batchnorm2d4 = nn.BatchNorm2d(self.number_of_tracks * 5*(2**3))
    
    def forward(self, x):
        #x shape (batch_size, tracks, time_seq, pitches)
        if self.linear:
            output = self.conv1(x)
            output = self.conv2(output)
            output = self.conv3(output)
            output = self.conv4(output)
            output = self.conv5(output)
            output = self.flatten(output)
            output = self.fc1(output)
            output = self.fc2(output)
            output = self.fc3(output)
        else:
            output = self.dropout2d1(self.batchnorm2d1(F.relu(self.conv1(x))))
            output = self.dropout2d2(self.batchnorm2d2(F.relu(self.conv2(output))))
            output = self.dropout2d3(self.batchnorm2d3(F.relu(self.conv3(output))))
            output = self.dropout2d4(self.batchnorm2d4(F.relu(self.conv4(output))))
            output = F.relu(self.conv5(output))
            output = self.flatten(output)
            output = F.relu(self.fc1(output))
            output = F.relu(self.fc2(output))
            output = self.fc3(output)
        return output
            
    
    def get_track_order(self):
        return self.track_order
