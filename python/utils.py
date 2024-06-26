import os
import pandas as pd


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
        if os.path.exists(os.path.join(parent_dir, 'data')) :
            print("base_dir is initialized to", os.path.join(parent_dir, 'data'))
            return os.path.join(parent_dir, 'data/')
        if current_dir == parent_dir :
            break
        current_dir = parent_dir
    return None


def get_dir_info_default (df) : 
    path_split = df.iloc[0]["filename"].split("/")
    data_type = df.iloc[0]["type"]
    dir_info = [ data_type ]

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
    dir_info.extend(tmp_dir_info)

    return dir_info


def make_dir_from_list (base_dir, dir_list) :
    abs_path = base_dir