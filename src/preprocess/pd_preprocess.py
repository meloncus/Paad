# dataset 에서 불러온 데이터를 전처리하기 위한 파일
# TODO : save, load data from pickle from dataframes
# TODO : get preprocessed data from preprocess_tools.py

import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os

SRC_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(SRC_DIR)

from preprocess.submodule.file_to_vector import file_to_vector_mel, file_to_vector_chroma
from preprocess.submodule.get_features import get_df_feat
from preprocess.submodule.train_test_split import train_test_split


data_types = ["fan", "pump", "slider", "valve"]

specific_case = "MIMII" # or DCASE

specific_year = None # DCASE

specific_dev_eval_additional = None # DCASE

specific_train_or_test = None # DCASE

specific_db = "data_-6_db" # MIMII


def get_specific_data (data_dict) :
    '''
    get specific data from data_dict such as MIMII[case][db] or DCASE[case][year][dev_eval_additional][train_or_test]

    input
    data_dict : dictionary, data_dict from get_data_paths_and_labels_from_machine

    output
    data : dictionary, specific data from data_dict
    '''
    '''
    MIMII, DCASE 일 경우 특정 데이터를 가져오기 위한 함수, default 값이 global 변수로 설정되어 있음
    MIMII[case][db] 혹은 DCASE[case][year][dev_eval_additional][train_or_test] 데이터를 가져옴

    input :
        data_dict : dictionary, get_data_paths_and_labels_from_machine에서 나온 data_dict

    output :
        data : dictionary, data_dict에서 특정 데이터를 가져온 것
        MIMII, DCASE가 아닌 경우 None을 반환
    '''

    if specific_case == "MIMII" :
        data = data_dict[specific_case][specific_db]

    elif specific_case == "DCASE" :
        data = data_dict[specific_case][specific_year][specific_dev_eval_additional][specific_train_or_test]

    else : 
        print("Please check the specific_case")
        return None
    
    return data


def default_preprocess (data_dict, label_dict) :
    '''
    default preprocess for data_dict and label_dict

    input
    data_dict : dictionary, specific data from get_specific_data
    label_dict : dictionary, label_dict from get_specific_data, { key : path, value : label(1 or -1) }

    output
    df : pandas.DataFrame, dataframe from data_dict and label_dict, { "filename" : flatten_data, "label" : label_data, "type" : type of data(such as "fan")}
        if data is from MIMII, add "model" columns { "model" : id of data(such as "1")}
    '''
    '''
    data_dict와 label_dict를 받아서 기본적인 전처리를 하고 dataframe으로 만드는 함수 (데이터를 직접적으로 다루지는 않음)

    input :
        data_dict : dictionary, get_specific_data에서 나온 특정 데이터
        label_dict : dictionary, get_specific_data에서 나온 label_dict, { key : path, value : label(1 or -1) }

    output :
        df : pandas.DataFrame, data_dict와 label_dict로부터 만들어진 dataframe, { "filename" : flatten_data, "label" : label_data, "type" : type of data(such as "fan")}
            만약 데이터가 MIMII일 경우 "model" column을 추가 { "model" : id of data(such as "1")}
    '''

    flatten_data = get_flatten_data(data_dict)

    label_data = get_label_from_flatten_specifics(flatten_data, label_dict)

    df = get_dataframe_from_flatten_data_and_label(flatten_data, label_data)
    # df = put_mel_from_dataframe(df)

    return df


def get_flatten_data (data_dict) :
    '''
    get list of path from data_dict

    input 
    data_dict : dictionary, specific data from get_specific_data

    output
    data : list, flatten data from data_dict
    '''
    '''
    path data 가 dictionary 형태로 들어올 경우 flatten data를 반환하는 함수. list 형태로 반환

    input :
        data_dict : dictionary, get_specific_data에서 나온 특정 데이터

    output :
        data : list, data_dict를 flatten한 것
    '''

    data = list()

    for key in data_dict.keys() :
        data.extend(data_dict[key])

    return data


def get_dropped_data_through_specific_id_from_flatten_data (flatten_data_list, specific_id = 0) :
    '''
    get list of path from flatten_data_list that contains specific_id

    input
    flatten_data_list : list, list of path
    specific_id : int, specific id

    output
    data : list, list of path that contains specific_id
    '''
    '''
    flatten_data_list에서 specific_id를 포함하는 path를 반환하는 함수

    input :
        flatten_data_list : list, path의 list
        specific_id : int, 특정 id

    output :
        data : list, specific_id를 포함하는 path의 list
    '''
    return [each_path for each_path in flatten_data_list if f"id_{specific_id:02d}" in each_path]


def get_label_from_flatten_specifics (flatten_data_list, specific_label_dict) :
    '''
    get label from list of path

    input
    flatten_data_list : list, list of path
    specific_label_dict : dictionary, label_dict from get_specific_data, { key : path, value : label(1 or -1) }

    output 
    label_data : list, list of label(1 or -1)
    '''
    '''
    label dictionary 에서 flatten_data_list에 해당하는 label을 가져오는 함수

    input :
        flatten_data_list : list, path의 list
        specific_label_dict : dictionary, get_specific_data에서 나온 label_dict, { key : path, value : label(1 or -1) }

    output :
        label_data : list, label(1 or -1)의 list
    '''

    return [ specific_label_dict[raw_path] for raw_path in flatten_data_list ]


def get_dataframe_from_flatten_data_and_label (flatten_data, label_data) :
    '''
    get dataframe from flatten_data and label_data

    input
    flatten_data : list, list of path
    label_data : list, list of label(1 or -1)

    output
    df : pandas.DataFrame, dataframe from flatten_data and label_data, { "filename" : flatten_data, "label" : label_data, "type" : type of data(such as "fan")}
        if data is from MIMII, add "model" columns { "model" : id of data(such as "1")}
    '''
    '''
    flatten_data와 label_data로부터 dataframe을 만드는 함수, type이 추가된다. 
    { "filename" : flatten_data, "label" : label_data, "type" : type of data(such as "fan")}

    input :
        flatten_data : list, path의 list
        label_data : list, label(1 or -1)의 list

    output :
        df : pandas.DataFrame, flatten_data와 label_data로부터 만들어진 dataframe
    '''
    df = pd.DataFrame({"filename" : flatten_data, "label" : label_data})

    sample_df = df.iloc[0]
    id_idx = None
    for each_type in data_types :
        if each_type in sample_df["filename"] :
            df["type"] = each_type
            
            if (each_type == "fan") or (each_type == "valve") :
                id_idx = [ idx for idx, each_in_path in enumerate(sample_df["filename"].split('/')) if "id_" in each_in_path ][0]
                df["model"] = df["filename"].apply(lambda full_id : full_id.split("/")[id_idx][4])
                break

    return df


def put_mel_from_dataframe (df) :
    '''
    put mel spectrogram to dataframe

    input
    df : pandas.DataFrame, dataframe from get_dataframe_from_flatten_data_and_label

    output
    df : pandas.DataFrame, dataframe with "mel" column
    '''
    '''
    mel spectrogram으로 전처리 된 데이터를 dataframe에 넣는 함수

    input :
        df : pandas.DataFrame, get_dataframe_from_flatten_data_and_label에서 나온 dataframe

    output :
        df : pandas.DataFrame, "mel" column이 추가된 dataframe
    '''
    mels = []
    print("Mel Spectrogram is being put into dataframe...")
    for filename in tqdm(df["filename"], desc = "Processing mel ") :
        mel = file_to_vector_mel(filename)
        mels.append(mel)
    df["mel"] = mels

    return df


def put_chroma_from_dataframe (df) :
    '''
    put chroma to dataframe

    input
    df : pandas.DataFrame, dataframe from get_dataframe_from_flatten_data_and_label

    output
    df : pandas.DataFrame, dataframe with "chroma" column
    '''
    '''
    chroma를 dataframe에 넣는 함수

    input :
        df : pandas.DataFrame, get_dataframe_from_flatten_data_and_label에서 나온 dataframe

    output :
        df : pandas.DataFrame, "chroma" column이 추가된 dataframe
    '''
    chromas = []
    print("Chroma is being put into dataframe...")
    for filename in tqdm(df["filename"], desc = "Processing chroma ") :
        chroma = file_to_vector_chroma(filename)
        chromas.append(chroma)
    df["chroma"] = chromas

    return df


def remove_underbar_in_column (df) :
    '''
    remove underbar in column name

    input
    df : pandas.DataFrame, dataframe

    output
    df : pandas.DataFrame, dataframe with removed underbar in column name
    '''
    '''
    column name에서 underbar를 제거하는 함수

    input :
        df : pandas.DataFrame, dataframe

    output :
        df : pandas.DataFrame, column name에서 underbar가 제거된 dataframe
    '''
    df.columns = df.columns.str.replace("_", " ").astype("str")

    return df


def preprocess_for_all_feat (df) :
    '''
    전체 모델을 위한 전처리 함수

    input :
        df : pandas.DataFrame, dataframe

    output :
        df : pandas.DataFrame, 전처리된 dataframe
    '''
    df_fan = get_df_feat(df, n_fft = 2048, sr = 16000)
    df_fan_means = get_df_feat(df, n_fft = 2048, sr = 16000, means = True)

    df_fan = preprocess_for_all_with_mean_feat(df_fan)
    df_fan_means = preprocess_for_all_with_mean_feat(df_fan_means)

    return df_fan, df_fan_means


def preprocess_for_all_with_mean_feat (df) :
    '''
    전체 모델에서 underbar를 제거하고 train test split을 하는 함수

    input :
        df : pandas.DataFrame, dataframe, preprocess_for_all_feat에서 나온 dataframe

    output :
        df : pandas.DataFrame, 전처리된 dataframe
    '''
    df = remove_underbar_in_column(df)
    df = train_test_split(df)

    return df