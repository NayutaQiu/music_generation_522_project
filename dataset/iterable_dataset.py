# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:48:50 2022

@author: nayut
"""
from torch.utils.data import IterableDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
from utils import data_reading
from dataset.onepass_dataset import STACK_ORDER
class IterableSongDataset(IterableDataset):
    def __init__(self, filepaths, seq_len, resolution, instruments, 
                 method, normalize, binarize, buffer_size, files_to_read, scaler_dict=None):
        """
        

        Parameters
        ----------
        dirpaths : str
            path to npz files directory
        seq_len : int
            window length
        resolution : int
            time steps at each beat
        instruments : list<str>
            wanted intruments track
        time_slice : TYPE
            if true only return a time step as y other wise return a window as y
        normalize: boolean
            whether to normalize by track
        files_to_read : int
            how many files to read for a complete iteration/one epoch
        method: str
            can be :time_slice, one_step_window, shift_window

        Returns
        -------
        None.

        """
        self.cursor = 0
        self.fileslist = filepaths
        self.seq_len = seq_len
        self.buffer_size = buffer_size
        self.files_to_read = files_to_read
        self.instruments = instruments
        self.resolution = resolution
        self.normalize = normalize
        self.scaler_dict = scaler_dict
        self.data_dict = None
        self.binarize = binarize
        self.method = method
        self.track_order = [track for track in STACK_ORDER if track in instruments]
    
    def __iter__(self):
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
        index = 0
        buffer_seq_len = 0
        file_cursor = 0
        while True:
            if index>=buffer_seq_len:
                print("reloading")
                del self.data_dict
                self.data_dict = dict()
                pprs, last = data_reading.read_and_preprocess_pianorolls_using_cursor(self.fileslist, self.seq_len, self.resolution, self.binarize, self.buffer_size, file_cursor)
                for pianoroll in pprs:
                    contains_instruments = [track.name for track in pianoroll.tracks]
                    if all([instrument in contains_instruments for instrument in self.track_order]):
                        for track in pianoroll.tracks:
                            self.data_dict.setdefault(track.name, []).append(track.pianoroll)
                for k in self.track_order:
                    self.data_dict[k] = np.concatenate(self.data_dict[k], axis=0)
                if self.normalize:
                    if self.scaler_dict == None:
                        self.scaler_dict = dict()
                        for k in self.track_order:
                            self.scaler_dict[k] = StandardScaler()
                            flatten_view = self.data_dict[k].reshape(-1, 1)
                            self.scaler_dict[k].fit(flatten_view)
                    flatten_view = self.data_dict[k].reshape(-1, 1)
                    self.data_dict[k]  = self.scaler_dict[k].transform(flatten_view).reshape(-1, 128)
                buffer_seq_len = self.data_dict[self.track_order[0]].shape[0]/self.seq_len - 2
                file_cursor += len(pprs)
                if file_cursor >= self.files_to_read:
                    last = True
                index = 0
            if index < buffer_seq_len:
                train_x_list = []
                train_y_list = []
                for track in self.track_order:
                    train_x_list.append(self.data_dict[track][index*self.seq_len:(index+1)*self.seq_len, :])
                    if self.method == "time_slice":
                        train_y_list.append(self.data_dict[track][(index+1)*self.seq_len, : ])
                    elif self.method == "one_step_window":
                        train_y_list.append(self.data_dict[track][(index*self.seq_len)+1:(index+1)*self.seq_len+1, :])
                    elif self.method == "shift_window":
                        train_y_list.append(self.data_dict[track][((index+1)*self.seq_len):(index+2)*self.seq_len, :])
                    else:
                        raise Exception("invalid method")
                train_x = np.stack(train_x_list)
                train_y = np.stack(train_y_list)
                yield train_x, train_y
                index += 1
                if last==True and index >= buffer_seq_len:
                    break
                    
        
    def get_scaler_dict(self):
        if self.normalize:
            return self.scaler_dict
        else:
            raise AttributeError("data is not normalized")
    
    def get_track_order(self):
        return self.track_order
    
    def get_seq_len(self):
        return self.seq_len
    
    def get_resolution(self):
        return self.resolution