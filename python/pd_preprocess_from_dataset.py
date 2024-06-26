# dataset 에서 불러온 데이터를 전처리하기 위한 파일
# TODO : save, load data from pickle from dataframes
# TODO : get preprocessed data from preprocess_tools.py

import pandas as pd
import numpy as np


data_types = ["fan", "pump", "slider", "valve"]

specific_case = "MIMII" # or DCASE

specific_year = None # DCASE

specific_dev_eval_additional = None # DCASE

specific_train_or_test = None # DCASE

specific_db = "data_-6_db" # MIMII


def get_specific_data (data_dict) :
    if specific_case == "MIMII" :
        data = data_dict[specific_case][specific_db]

    elif specific_case == "DCASE" :
        data = data_dict[specific_case][specific_year][specific_dev_eval_additional][specific_train_or_test]

    else : 
        print("Please check the specific_case")
        return None
    
    return data


def get_flatten_data (data_dict) :
    data = list()

    for key in data_dict.keys() :
        data.extend(data_dict[key])

    return data


def get_label_from_flatten_specifics (flatten_data_list, specific_label_dict) :
    return [ specific_label_dict[raw_path] for raw_path in flatten_data_list ]


def get_dataframe_from_flatten_data_and_label (flatten_data, label_data) :
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