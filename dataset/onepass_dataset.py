# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:27:45 2022

@author: nayut
"""
import numpy as np
STACK_ORDER = ["Drums", "Piano", "Guitar", "Bass", "Strings"]
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
class OnePassSongsDataset(Dataset):
    def __init__(self, data, seq_len, instruments, time_slice, normalize):
        """
        

        Parameters
        ----------
        data : list<pianoroll>
            Read and Preprocessed pianorolls
        seq_len : int
            Window Size
        time_slice : bool
            if true only return a time step as y other wise return a window as y
        instruments : list<str>
            wanted intruments track
        normalize: boolean
            whether to normalize by track

        Returns
        -------
        None.

        """
        self.scaler_dict = dict()
        self.data_dict = dict()
        self.seq_len = seq_len
        self.time_slice = time_slice
        self.matrix_order = [track for track in STACK_ORDER if track in instruments]
        self.normalize = normalize
        for pianoroll in data:
            contains_instruments = [track.name for track in pianoroll.tracks]
            if all([instrument in contains_instruments for instrument in self.matrix_order]):
                for track in pianoroll.tracks:
                    self.data_dict.setdefault(track.name, []).append(track.pianoroll)
        for k in self.matrix_order:
            self.data_dict[k] = np.concatenate(self.data_dict[k], axis=0)
            if self.normalize:
                self.scaler_dict[k] = StandardScaler()
                flatten_view = self.data_dict[k].reshape(-1, 1)
                self.data_dict[k] = self.scaler_dict[k].fit_transform(flatten_view).reshape(-1, 128)
        self.length = self.data_dict[self.matrix_order[0]].shape[0]/seq_len - 1
                

    def __getitem__(self, index):
        """
        

        Parameters
        ----------
        index : TYPE
            DESCRIPTION.

        Returns
        -------
        x_train : np array 
            shape: (tracks, timestep, pitches)
            specified tracks at index window
        y_train : TYPE
            shape: (tracks, timestep, pitches)
            specified tracks at index window shifted one time step or 
            specified tracks at one step behind train_x if time_slice=true
        """
        x_list = []
        y_list = []
        for track in self.matrix_order:
            x_list.append(self.data_dict[track][index*self.seq_len:(index+1)*self.seq_len, :])
            if self.time_slice:
                y_list.append(self.data_dict[track][(index+1)*self.seq_len, : ])
            else:
                y_list.append(self.data_dict[track][(index*self.seq_len)+1:(index+1)*self.seq_len+1, :])
        x_train = np.stack(x_list)
        y_train = np.stack(y_list)
        return x_train, y_train


    def get_scaler_dict():
        if self.normalize:
            return self.scaler_dict
        else:
            raise AttributeError("data is not normalized")
    
    def get_matrix_order():
        return self.matrix_order