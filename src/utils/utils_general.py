import os
import datetime
import pandas as pd
import tensorflow as tf
import numpy as np

from tqdm import tqdm

BASE_DIR_NAME = "data"
PICKLE_PATH = "working/pkl/"

import sys

SRC_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(SRC_DIR)

from preprocess.submodule.vector_to_numpy_arr import vector_to_numpy_arr
from preprocess.submodule.normalize import min_max_normalization
from preprocess.submodule.get_features import get_features, convert_complex_to_real


# base_dir를 초기화하는 데코레이터
def initialize_base_dir(func) :
    def wrapper(*args, **kwargs) :
        if 'base_dir' not in kwargs or kwargs['base_dir'] is None :
            # base_dir을 새로 계산하거나 업데이트하는 로직
            base_dir = get_base_dir()
            kwargs['base_dir'] = base_dir
        return func(*args, **kwargs)
    return wrapper


# base_dir를 초기화하는 함수
def get_base_dir() :
    current_dir = os.path.abspath(__file__)
    while True :
        parent_dir = os.path.dirname(current_dir)
        if os.path.exists(os.path.join(parent_dir, BASE_DIR_NAME)) :
            print("base_dir is ", os.path.join(parent_dir, BASE_DIR_NAME))
            return os.path.join(parent_dir, BASE_DIR_NAME + '/')
        if current_dir == parent_dir :
            break
        current_dir = parent_dir
    return None


def get_dir_path_default_dataframe (df) : 
    path_split = df.iloc[0]["filename"].split("/")
    data_type = df.iloc[0]["type"]
    dir_info = data_type

    idx_data = None
    idx_is_normal = None

    for idx, path_ele in enumerate(path_split) :
        if "data" in path_ele :
            idx_data = idx
        if "normal" in path_ele :
            idx_is_normal = idx
            break
    
    tmp_dir_info = path_split[idx_data:idx_is_normal]
    drop_list_contains = [ "train", "test", "normal", "abnormal", "id", data_type ]
    tmp_dir_info = [ each for each in tmp_dir_info if not any(substring in each for substring in drop_list_contains) ]
    for each in tmp_dir_info :
        dir_info = os.path.join(dir_info, each)

    return dir_info


def get_abs_dir_path (dir_path) :
    return os.path.join(get_base_dir(), PICKLE_PATH, dir_path)


def make_dir_from_abs_path (dir_path) :
    print("Be careful with the path, This can make all directories in the path.")
    print("Path : ", dir_path)
    print("Are you sure you want to make directories in the path? (y/n)")

    while(True) :
        answer = input()
        if answer == 'y' :
            break
        elif answer == 'n' :
            return
        else :
            print("Please type 'y' or 'n'.")

    os.makedirs(dir_path, exist_ok=True)


def save_dataframe (df, abs_path, filename) :
    if os.path.exists(abs_path) :
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        data_path = os.path.join(abs_path, current_time + '_' + filename + ".pkl")

        try :
            df.to_pickle(data_path)
            print("Dataframe is saved in ", data_path)
            return data_path
        except Exception as e :
            print("Error occured while saving dataframe.")
            print(e)
            return None


def get_matrixes (df, feat = "mel") :
    '''
    get matrixes from dataframe

    input
    df : pandas.DataFrame, dataframe
    feat : string, feature name

    output
    train_data : numpy.array, training data
    validate_data : numpy.array, validation data
    y_data : numpy.array, labels
    inputDim : tuple, input dimension
    '''
    # vector_to_numpy_arr : list of vector to numpy array
    train_data = vector_to_numpy_arr(df[(df["train"] == 1)][feat].tolist())
    validate_data = vector_to_numpy_arr(df[(df["test"] == 1)][feat].tolist())
    # normal_data = vector_to_numpy_arr(df[(df["test"] == 1) & (df["label"] == 1)][feat].tolist())
    # abnormal_data = vector_to_numpy_arr(df[(df["test"] == 1) & (df["label"] == -1)][feat].tolist())
    y_data = df[(df["test"] == 1)]["label"].tolist()

    # normalization
    min_value = tf.reduce_min(train_data)
    max_value = tf.reduce_max(train_data)

    train_data = min_max_normalization(train_data, min_value, max_value)
    validate_data = min_max_normalization(validate_data, min_value, max_value)
    # normal_data = min_max_normalization(normal_data, min_value, max_value)
    # abnormal_data = min_max_normalization(abnormal_data, min_value, max_value)

    # cast to float32
    train_data = tf.cast(train_data, tf.float32)
    validate_data = tf.cast(validate_data, tf.float32)
    # normal_data = tf.cast(normal_data, tf.float32)
    # abnormal_data = tf.cast(abnormal_data, tf.float32)

    # return train_data, validate_data, normal_data, abnormal_data
    return train_data, validate_data, y_data, train_data.shape[1:]


def get_df_feat(df, n_fft, sr, means=False):
    ''' Used to extract Features from spectrograms 
    MFCC, Log mel energy and Chroma (CENS)
    '''
    feat_cols = []

    # Initialize the progress bar
    progress_bar = tqdm(total=len(df), position=0, leave=True)
    for i, row in df.iterrows():
        filename = row['filename']
        feat, labels = get_features(filename, n_fft, sr, frac=10, means=means)

        feat_cols.append(feat)
        lab_cols = labels
        # Update the progress bar
        progress_bar.update(1)

    feat_array = np.vstack(feat_cols)
    lab_array = lab_cols.flatten()

    feat_df = pd.DataFrame(feat_array, columns=list(lab_array), index=df.index)

    # Convert complex numbers to real values
    feat_df = feat_df.applymap(convert_complex_to_real)

    # Assign the columns to the original DataFrame
    df = pd.concat([df, feat_df], axis=1)

    return df