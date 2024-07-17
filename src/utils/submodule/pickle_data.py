import os

DATA_PATH = "/mnt/d/silofox/paad/anomaly-example/exploring-AAD/notebooks/paad/tmp/data/df_an.pkl"
TARGET_FILE = "df_meta_5050"


def search_pickle_data (data_path = DATA_PATH, dir_name_contains = TARGET_FILE) :
    dirname_parents = os.path.dirname(data_path)
    dirs = os.listdir(dirname_parents)

    contains_list = []
    for dirname in dirs :
        if dir_name_contains in dirname :
            contains_list.append(dirname)
    
    if len(contains_list) == 0 :
        return None
    
    return contains_list


def make_pickle_path (dirnames, abs_path = DATA_PATH) :
    path_list = []
    for dirname in dirnames :
        path_list.append(abs_path[:abs_path.rfind("/") + 1] + dirname)
    return path_list


def filtered_dirnames (pickles, filter_name) :
    return [ pickle for pickle in pickles if filter_name in pickle ]


def decision_filter (pickles, filter_list = ["train_test", "feat", "pkl"]) :
    for filter_name in filter_list :
        for each_pickle in pickles :
            if filter_name in each_pickle :
                return filtered_dirnames(pickles, filter_name)
    return None


if __name__ == "__main__" :
    pickles = search_pickle_data()
    print(pickles)
    path_list = decision_filter(pickles)
    print(path_list)
    abs_path = make_pickle_path(path_list)
    print(abs_path)