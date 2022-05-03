from torch.utils.data import IterableDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
from utils import data_reading
from dataset.onepass_dataset import STACK_ORDER
class IterableClassificationSongDataset(IterableDataset):
    def __init__(self, pprs, window_length, resolution, instruments, label_dict):
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
        self.pprs = pprs
        self.window_length = window_length
        self.instruments = instruments
        self.resolution = resolution
        self.label_dict = label_dict
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
        for pianoroll in self.pprs:
            train_x = pianoroll.stack()[:,:self.window_length]
            train_y = self.label_dict.loc[[pianoroll.name], "label"][0]
            yield train_x, train_y
                
                    
                    

    
    def get_track_order(self):
        return self.track_order
    
    def get_window_length(self):
        return self.window_length
    
    def get_resolution(self):
        return self.resolution