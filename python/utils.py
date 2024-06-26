import os
import datetime
import pandas as pd

BASE_DIR_NAME = "data"
PICKLE_PATH = "working/pkl/"


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
