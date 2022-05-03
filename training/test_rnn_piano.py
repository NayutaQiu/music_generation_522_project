from utils import data_reading, music_generation
from models import music_generation_gru
import os
import torch
import pickle
SEQ_LEN = 48
RESOLUTION = 12
TRAIN_SPLIT = 0.8
HIDDEN_SIZE = 512
BATCH_SIZE = 64
N_LAYERS = 6
CUT_TIME_RATIO = 0.1
TEST_FILE_NUMBER = 10
CUTOFF_P = 0.5
PROJECT_DIR ="D:/522 project"
PPR_DIR = "lpd_5_pianoroll"
SAVE_DIR = "./gru_music"
SCALER_DICT_SAVE_PATH = "./gru_saving/gru_piano_scaler_dict.pkl"
TRACK_ORDER_SAVE_PATH = "./gru_saving/gru_piano_track_order.pkl"
with open(SCALER_DICT_SAVE_PATH, "rb") as file:
    scaler_dict = pickle.load(file)
with open(TRACK_ORDER_SAVE_PATH, "rb") as file:
    track_order = pickle.load(file)
PATH = os.path.join(PROJECT_DIR, PPR_DIR)
MODEL_STATE_PATH = "./P_gru_trained_r12s48d24e1"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = music_generation_gru.MusicGRU(track_order, HIDDEN_SIZE, BATCH_SIZE, N_LAYERS, DEVICE)
model.load_state_dict(torch.load(MODEL_STATE_PATH))
model.to(DEVICE)
all_files_list = data_reading.read_all_npzs(PATH, shuffle=True)
all_files_list = all_files_list[:TEST_FILE_NUMBER]
all_pianorolls = data_reading.read_and_preprocess_and_filter_pianorolls(all_files_list, track_order, SEQ_LEN, RESOLUTION)


for i, pianoroll in enumerate(all_pianorolls):
    prelude_dict, rest_dict = music_generation.cut_prelude(pianoroll, CUT_TIME_RATIO)
    predict_time_step = rest_dict["Piano"].shape[0]
    generated_part_dict = music_generation.music_generation_rnn(model, scaler_dict, track_order
                                                                , prelude_dict, predict_time_step
                                                                , DEVICE)
    new_music_dict = music_generation.combine_tracks_dict(prelude_dict, generated_part_dict)
    new_music_dict = music_generation.binarize_generated_music_dict(new_music_dict, CUTOFF_P)
    new_music_mt = music_generation.create_multitrack(new_music_dict, RESOLUTION)
    pianoroll.write(f"{i} original.mid")
    new_music_mt.write(f"{i} generated.mid")
    

