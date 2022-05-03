"""
Created on Fri Apr 24 11:32:45 2022

@author: nayuta Qiu
"""
import torch
import pypianoroll as ppr
import numpy as np
TRACK_TO_PROGRAM_DICT = {"Drums":None, "Piano":0, "Guitar":24, "Bass":32, "Strings":48}
def cut_prelude(song, cut_time_ratio):
    """
    

    Parameters
    ----------
    song : pianoroll object
        DESCRIPTION.
    cut_time_ratio : p from 0 to 1
        DESCRIPTION.

    Returns
    -------
    prelude_tracks : dict[track_name] = tensor(time_step, pitches)
        DESCRIPTION.
    rest_tracks : dict[track_name] = tensor(time_step, pitches)
        DESCRIPTION.

    """
    prelude_tracks = dict()
    rest_tracks = dict()
    for track in song.tracks:
        track_np = track.pianoroll
        cut_time_step = round(track_np.shape[1]*cut_time_ratio)
        prelude_tensor = torch.from_numpy(track_np[:cut_time_step, :])
        rest_tensor = torch.from_numpy(track_np[cut_time_step:, :])
        prelude_tracks[track.name] = prelude_tensor
        rest_tracks[track.name] = rest_tensor
    return prelude_tracks,rest_tracks

def create_tracks_dict_from_tensor(song_tensor, track_order):
    generated_tracks_dict = dict()
    for track_name, track_tensor in zip(track_order, song_tensor):
        generated_tracks_dict[track_name] = track_tensor
    return generated_tracks_dict
def create_tracks_dict(song):
    """
    

    Parameters
    ----------
    song : pianoroll object
        DESCRIPTION.

    Returns
    -------
    generated_tracks_dict : dict

    """
    generated_tracks_dict = dict()
    for track in song.tracks:
        generated_tracks_dict[track.name] = track
    return generated_tracks_dict
#TODO: handle if not normalized
def music_generation_rnn(model, scaler_dict, track_order, prelude_tracks_dict, predict_time_step, device):
    """
    

    Parameters
    ----------
    model : torch model
        trained rnn model
    dataset: 
        training dataset
    prelude_tracks : dict
        dict[track_name] = tensor in shape (timestep, pitches)
    predict_time_step : int
        number of timesteps to predict

    Returns
    -------
    generated_tracks_dict : dict
    in shape dict[track]= tensor(predict_time_steps, pitches)
    
    
    """
    generated_tracks_dict = dict()
    transformed_tracks = []
    model.eval()
    for track in track_order:
        flatten_view = prelude_tracks_dict[track].reshape(-1, 1)
        transformed_track = scaler_dict[track].transform(flatten_view)
        transformed_track = transformed_track.reshape(-1, 128)
        transformed_tracks.append(transformed_track)
    transformed_tracks_tensor = torch.from_numpy(np.stack(transformed_tracks))\
        .float().to(device).unsqueeze(0)
    with torch.no_grad():
        results = []
        hidden = model.init_hidden(1)
        for i in range(predict_time_step):
            if i==0:
                o,hidden = model(transformed_tracks_tensor, hidden)
                #o shape (batch_size, tracks, time_steps, piches)
            else:
                o, hidden = model(o,hidden)
            results.append(o.squeeze(0))
    res = torch.cat(results, dim=1)
    for track_name, track in zip(track_order, res):
        generated_tracks_dict[track_name] = track
    return generated_tracks_dict

def combine_tracks_dict(left_tracks_dict, right_tracks_dict):
    """
    

    Parameters
    ----------
    left_tracks_dict : dict[track_name] = tensor(time_step, pitches)
        DESCRIPTION.
    right_tracks_dict : dict[track_name] = tensor(time_step, pitches)
        DESCRIPTION.

    Raises
    ------
    KeyError
        DESCRIPTION.

    Returns
    -------
    new_tracks_dict : dict[track_name] = tensor(time_step, pitches)
        DESCRIPTION.

    """
    new_tracks_dict = dict()
    if sorted(left_tracks_dict.keys()) != sorted(right_tracks_dict.keys()):
        raise KeyError("left tracks keys different from right tracks keys")
    else:
        for track_name in left_tracks_dict.keys():
            new_tracks_dict[track_name] = torch.concat((left_tracks_dict[track_name].cpu(),
                                                       right_tracks_dict[track_name].cpu()),
                                                       dim=0)
        return new_tracks_dict
        
def binarize_generated_music_dict(generated_tracks_dict, cutoff_p, vol=85):
    """
    

    Parameters
    ----------
    generated_tracks_dict : dict[track_name] = tensor(time_step, pitches)
        DESCRIPTION.
    cutoff_p : float from 0 to 1
        DESCRIPTION.
    vol : TYPE, track volume constant
        DESCRIPTION. The default is 85.

    Returns
    -------
    binarized_tracks_dict : dict[track_name] = tensor(time_step, pitches)
        DESCRIPTION.

    """
    binarized_tracks_dict = dict()
    for track_name,track in generated_tracks_dict.items():
        new_tensor = torch.zeros(track.shape)
        new_tensor[track>cutoff_p] = vol
        binarized_tracks_dict[track_name] = new_tensor
    return binarized_tracks_dict

def create_multitrack(tracks_dict, resolution):
    """
    

    Parameters
    ----------
    tracks_dict : dict[track_name] = tensor(time_step, pitches)
        DESCRIPTION.
    resolution : int
        DESCRIPTION.

    Returns
    -------
    multitrack : pianoroll object
        DESCRIPTION.

    """
    tracks = []
    for track_name, track in tracks_dict.items():
        if track_name == "Drums":
            track = ppr.StandardTrack(name=track_name, is_drum=True, 
                                      pianoroll=track.detach().cpu().numpy())
        else:
            track = ppr.StandardTrack(name=track_name, 
                                      program=TRACK_TO_PROGRAM_DICT[track_name], 
                                      pianoroll=track.detach().cpu().numpy(),
                                      is_drum=False)
        tracks.append(track)
    multitrack = ppr.Multitrack(name="Generated", resolution=resolution, tracks=tracks)
    return multitrack

def check_vae_generation(model, dataset, song, device):
    """
    

    Parameters
    ----------
    model : trained vae model
        DESCRIPTION.
    dataset : data loader that can contain any sequence of song
        DESCRIPTION.
    song : pianoroll object
        song to reconstruct

    Returns
    -------
    new_tracks_dict : dict[track_name] = tensor(time_step, pitches)

    """
    
    model.eval()
    original_tracks = []
    transformed_tracks = []
    track_order = dataset.get_track_order()
    for song_seq, _ in dataset:
        #shape(tracks, time_step, pitches)
        song_seq = torch.from_numpy(song_seq).unsqueeze(0).to(device)
        original_tracks.append(song_seq)
        transformed_track = model(song_seq)
        transformed_tracks.append(transformed_track)
    original_tracks_tensor = torch.cat(original_tracks, dim=1)
    transformed_tracks_tensor = torch.cat(transformed_tracks, dim=1)
    original_tracks_dict = create_tracks_dict_from_tensor(original_tracks_tensor, track_order)
    transformed_tracks_dict = create_tracks_dict_from_tensor(transformed_tracks_tensor, track_order)
    return original_tracks_dict, transformed_tracks_dict
    