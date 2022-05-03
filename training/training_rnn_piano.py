# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 22:05:10 2022

@author: nayut
"""
from dataset import iterable_dataset
from models.music_generation_gru import MusicGRU
import os
from utils import training_rnn, data_reading
import torch
from torch.utils.data import DataLoader
import pickle
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_DIR ="D:/522 project"
PPR_DIR = "lpd_5_pianoroll"
SEQ_LEN = 48
RESOLUTION = 12
TRAIN_SPLIT = 0.8
HIDDEN_SIZE = 512
BATCH_SIZE = 64
N_LAYERS = 6
ALPHA = 5
LR = 0.0001
LR_LAMBDA = 1
EPOCH = 40
AUTOSAVE = True

TRAIN_BUFFER_SIZE = 1000
TRAIN_FILES_TO_READ = 1000

TEST_BUFFER_SIZE = 300
TEST_FILE_TO_READ = 300

SCALER_DICT_SAVE_PATH = "./gru_saving/gru_piano_scaler_dict.pkl"
TRACK_ORDER_SAVE_PATH = "./gru_saving/gru_piano_track_order.pkl"
SAVE_ON_EPOCH = 10
PATH = os.path.join(PROJECT_DIR, PPR_DIR)
INSTRUMENT = ["Piano"]
all_file_paths = data_reading.read_all_npzs(PATH, True)
train_file_paths = all_file_paths[:round(len(all_file_paths)*TRAIN_SPLIT)]
test_file_paths = all_file_paths[round(len(all_file_paths)*TRAIN_SPLIT):]
train_piano_dataset = iterable_dataset.IterableSongDataset(train_file_paths, SEQ_LEN, RESOLUTION, INSTRUMENT, 
                                          time_slice=False, normalize=True, 
                                          buffer_size=TRAIN_BUFFER_SIZE, files_to_read=TRAIN_FILES_TO_READ)
train_piano_loader = DataLoader(train_piano_dataset, batch_size = BATCH_SIZE, drop_last=True)
test_piano_dataset = iterable_dataset.IterableSongDataset(test_file_paths, SEQ_LEN, RESOLUTION, INSTRUMENT, 
                                         time_slice=False, normalize=True, 
                                         buffer_size=TEST_BUFFER_SIZE, files_to_read=TEST_FILE_TO_READ)
test_piano_loader = DataLoader(test_piano_dataset, batch_size = BATCH_SIZE, drop_last=True)
model = MusicGRU(train_piano_dataset.get_track_order(), HIDDEN_SIZE, BATCH_SIZE, N_LAYERS, DEVICE).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: LR_LAMBDA ** epoch)
criterion = torch.nn.MSELoss()

training_rnn.training_rnn(model, optimizer, scheduler, criterion, train_piano_loader,
             test_piano_loader, EPOCH, DEVICE, BATCH_SIZE,  ALPHA, AUTOSAVE, SAVE_ON_EPOCH)
with open(SCALER_DICT_SAVE_PATH, "wb") as file:
    pickle.dump(train_piano_dataset.get_scaler_dict(), file)
with open(TRACK_ORDER_SAVE_PATH, "wb") as file:
    pickle.dump(train_piano_dataset.get_track_order(), file)

