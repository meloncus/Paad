import os

DATA_PATH = "/mnt/d/silofox/paad/anomaly-example/exploring-AAD/notebooks/paad/tmp/data/df_an.pkl"
TARGET_FILE = "df_meta_5050"


def search_pickle_data (data_path = DATA_PATH, dir_name_contains = TARGET_FILE) :
    '''
    pickle 파일을 찾는 함수

    input :
        data_path : string, 데이터 경로
        dir_name_contains : string, 포함하는 디렉토리 이름

    output :
        contains_list : list, 포함하는 디렉토리 이름 리스트
        or
        None
    '''
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
    '''
    pickle 파일 경로를 만드는 함수

    input :
        dirnames : list, 디렉토리 이름 리스트
        abs_path : string, 절대 경로

    output :
        path_list : list, 경로 리스트
    '''
    path_list = []
    for dirname in dirnames :
        path_list.append(abs_path[:abs_path.rfind("/") + 1] + dirname)
    return path_list


def filtered_dirnames (pickles, filter_name) :
    '''
    pickle 파일을 필터링하는 함수

    input :
        pickles : list, pickle 파일 이름 리스트
        filter_name : string, 필터 이름

    output :
        filtered_list : list, 필터링된 리스트
    '''
    return [ pickle for pickle in pickles if filter_name in pickle ]


def decision_filter (pickles, filter_list = ["train_test", "feat", "pkl"]) :
    '''
    pickle 파일을 분류해서 사용할 데이터를 결정하는 함수
    우선순위는 train_test > feat > pkl 순으로 높음

    input :
        pickles : list, pickle 파일 이름 리스트
        filter_list : list, 필터 리스트

    output :
        path_list : list, 사용할 데이터 경로 리스트
        or
        None
    '''
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