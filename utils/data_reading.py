import numpy as np
import random
import pypianoroll as ppr
import glob
import os
import pandas as pd

#should be
def true_remove_empty(multitrack):
    """
    bug in pianoroll code:
    def remove_empty(self: MultitrackType) -> MultitrackType:

            self.tracks = [
                track for track in self.tracks if not np.any(track.pianoroll)
            ]
            return self
    """
    multitrack.tracks = [
        track for track in multitrack.tracks if np.any(track.pianoroll)
    ]
    return multitrack


def read_and_preprocess_pianorolls(filepaths, seq_len, resolution, binarize):
    """
    Read pianorolls into a list
    dir:a path str list that contain all the absolute paths to the npz file
    seq_len: window size
    resolution:time step per beat
    """
    res_list = []

    for path in filepaths:
        preprocessed_ppr = read_and_preprocess_pianoroll(path, seq_len, resolution, binarize)
        res_list.append(preprocessed_ppr)
    return res_list

def read_and_preprocess_and_filter_pianorolls(filepaths, required_tracks, seq_len, resolution, binarize ):
    """
    Read pianorolls into a dict with id as key and pianoroll object as value
    dir:a path str list that contain all the absolute paths to the npz file
    seq_len: window size
    resolution:time step per beat
    """
    res_list = []

    for path in filepaths:
        preprocessed_ppr = read_and_preprocess_pianoroll(path, seq_len, resolution, binarize)
        contained_tracks = [track.name for track in preprocessed_ppr.tracks]
        if all ([track in contained_tracks for track in required_tracks]):
            filtered_tracks = []
            for track in preprocessed_ppr.tracks:
                if track.name in required_tracks:
                    filtered_tracks.append(track)
            preprocessed_ppr.tracks=filtered_tracks
            res_list.append(preprocessed_ppr)
    return res_list

def read_and_preprocess_and_filter_pianorolls2(filepaths, required_tracks, seq_len, resolution, required_list, binarize ):
    """
    Read pianorolls into a dict with id as key and pianoroll object as value
    dir:a path str list that contain all the absolute paths to the npz file
    seq_len: window size
    resolution:time step per beat
    """
    res_list = []

    for i, path in enumerate(filepaths):
        if i%1000 == 0:
            print(i)
        preprocessed_ppr = read_and_preprocess_pianoroll(path, seq_len, resolution, binarize)
        contained_tracks = [track.name for track in preprocessed_ppr.tracks]
        if all ([track in contained_tracks for track in required_tracks]) and preprocessed_ppr.name in required_list:
            preprocessed_ppr.tracks = [track for track in preprocessed_ppr.tracks if track.name in required_tracks]
            res_list.append(preprocessed_ppr)
    return res_list

def read_and_preprocess_pianorolls_using_cursor(filepaths, seq_len, resolution, binarize, num_files, start_cursor):
    """
    dirpaths: a path str list that contain all the absolute paths to the npz file
    seq_len: window length
    resolution: time step per beat
    num_files: number of files to read
    start_cursor: start index of the list to read (inclusive)
    
    return
    load_list: list of loaded pianorolls
    no_more_files: boolean value indicate whether this is the final list. 
    /whether the last file is contained in the load list
   """
    if start_cursor >= len(filepaths):
       raise IndexError("Given start_curosr surpass given list limit")
    load_list = []
    cursor = start_cursor
    no_more_files = False
    while len(load_list) < num_files:
        preprocessed_ppr = read_and_preprocess_pianoroll(filepaths[cursor], seq_len, resolution, binarize)
        load_list.append(preprocessed_ppr)
        cursor +=1
        if cursor >= len(filepaths):
            no_more_files = True
            break
        
    return load_list, no_more_files

def read_all_npzs(dirpath, shuffle):
    """
    dirpath: path to the lakh dataset
    return a list of abs path of npz files
    """
    dirpath = os.path.join(dirpath, "**", "*.npz")
    paths = glob.glob(dirpath, recursive=True)
    if shuffle:
        random.shuffle(paths)
    return paths

def read_all_npys(dirpath, shuffle):
    """
    dirpath: path to the lakh dataset
    return a list of abs path of npz files
    """
    dirpath = os.path.join(dirpath, "**", "*.npy")
    paths = glob.glob(dirpath, recursive=True)
    if shuffle:
        random.shuffle(paths)
    return paths
def read_and_preprocess_pianoroll(filepath, seq_len, resolution, binarize):
    preprocessed_ppr = ppr.load(filepath)
    preprocessed_ppr = true_remove_empty(preprocessed_ppr).trim().set_resolution(resolution).pad_to_same().pad_to_multiple(seq_len)
    if binarize:
        preprocessed_ppr.binarize()
    return preprocessed_ppr

def read_label_tagtraum(dir_path):
    res_list = []
    for file_name in os.listdir(dir_path):
        label_name = file_name.split("_")[-1][:-4]
        df = pd.read_csv(os.path.join(dir_path, file_name), names=["id"])
        df["genre"] = label_name
        res_list.append(df)
    res = pd.concat(res_list,axis=0)
    return res
def read_lpd_to_msd_df(path):
    return pd.read_csv(path, sep="    ", names=["lpd_id", "msd_id"])