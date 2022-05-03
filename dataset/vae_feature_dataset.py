from torch.utils.data import IterableDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
from utils import data_reading
from dataset.onepass_dataset import STACK_ORDER
class IterableSongDataset(IterableDataset):
    def __init__(self, filepaths, seq_len, track_order,files_to_read, resolution):
        self.filepaths = filepaths
        self.seq_len = seq_len
        self.track_order = track_order
        self.files_to_read = files_to_read
        self.hold_data = None
        self.file_cursor = 0
        self.index = 0
        self.resolution = resolution
    
    def __iter__(self):
        self.file_cursor = 0
        while True:
            if self.hold_data is None or (self.index+1) * self.seq_len+1>=len(self.hold_data):
                if self.file_cursor >= self.files_to_read or self.file_cursor>=len(self.filepaths):
                    break
                else:
                    self.hold_data = np.load(self.filepaths[self.file_cursor])
                    self.index=0
                    self.file_cursor+=1
            if len(self.track_order) == 1:
                yield np.expand_dims(self.hold_data[self.index*self.seq_len: (self.index+1)*self.seq_len],0),\
                    np.expand_dims(self.hold_data[self.index*self.seq_len+1: (self.index+1)*self.seq_len+1],0)
            else:
                yield np.swapaxes(self.hold_data[self.index*self.seq_len: (self.index+1)*self.seq_len, 1:2],1,0),\
                    np.swapaxes(self.hold_data[self.index*self.seq_len: (self.index+1)*self.seq_len, 0:1], 1,0)
            
            self.index+=1
    
    def get_track_order(self):
        return self.track_order
    
    def get_resolution(self):
        return self.resolution
    def get_seq_len(self):
        return self.seq_len