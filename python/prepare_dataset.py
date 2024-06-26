# dataset 을 분류하기 위한 함수들을 정의한 파일
# get_data_paths_and_labels 함수를 통해 데이터셋의 경로와 label을 반환
# data_dict, label_dict 가 결과값으로 나옴
# data_dict 에서 DCASE 데이터셋은 년도, dev||eval||add, train||test, normal||abnormal 으로 분류
# data_dict 에서 MIMII 데이터셋은 db(decibel), normal||abnormal 으로 분류
# label_dict 는 data_dict 의 value 값에 해당하는 데이터셋의 label을 저장
# 데이터셋은 id 로 구분해서 관리하지 않았음
# 추가적인 데이터셋은 또 구현하면 됨

# DCASE dir 형식 : "data/DCASE/{year}/{dev_eval_additional}/{train_or_test}/{data_class_and_file_name}"
# MIMII dir 형식 : "data/MIMII/data_{db}/{machine_type}/id_{id}/{data_class}/{file_name}"
# 추가 데이터셋 dir 형식 : 

import os
from utils import initialize_base_dir


data_case = [ "DCASE", "MIMII" ] # 분류할 데이터셋의 종류

# DCASE start
dcase_years = [ "2020", "2021", "2022", "2023", "2024" ]  # 분류할 데이터셋의 연도

# DCASE 데이터셋의 경우, 2020년부터 2024년까지의 데이터셋이 있으며, 2023년과 2024년의 데이터셋은 개발용 데이터셋만 존재
dcase_year_contains_all = [ # 분류할 데이터셋의 연도 (전체)
    "2020", 
    "2021", 
    "2022",
    # "2023",
    # "2024",
] 

dcase_year_contains_only_dev = [ # 분류할 데이터셋의 연도 (개발용 데이터만 있는 연도)
    "2023",
    "2024",
]

dcase_dev_eval_additional = [ # 분류할 데이터셋의 종류
    "dev",
    "eval",
    "add",
]

train_or_test = [ "train", "test" ]  # 분류할 데이터셋의 종류
# DCASE end

# MIMII start
mimii_dbs = [ # db에 따른 데이터 경로
    "data_-6_db",
    "data_0_db",
    "data_6_db",
]

mimii_ids = [
    "id_00",
    "id_02",
    "id_04",
    "id_06",
]
# MIMII end

machine_types = [ "fan", "valve" ]  # 분류할 기계의 종류
data_class = [ "normal", "abnormal", "unknown"]  # 분류할 데이터의 종류


@initialize_base_dir
def get_data_paths_and_labels_from_machine(machine, base_dir) :
    '''
    data_path를 조합 -> dcase 혹은 mimii 데이터셋의 디렉토리 경로와 label을 추출

    ex)
    data_dict[”DCASE”][”2020”][”dev”]["train"][”normal”] = {real_path},
    data_dict[”MIMII”][”data_-6_db”][”normal”] = {real_path}

    inputs
    machine : string, 분류할 기계의 종류
    base_dir : string, 데이터셋이 저장된 디렉토리 경로 (상대경로)

    outputs
    data_dict : dictionary, 데이터셋의 디렉토리 경로, { key : DCASE or MIMII, value : { key : year or db, value : { key : dev or eval or add or train or test, value : { key : normal or abnormal, value : 디렉토리 경로 } } } }
    label_dict : dictionary, 데이터셋의 label { key : 디렉토리 경로, value : label_list}
    '''

    data_dict = dict()
    label_dict = dict()
    
    # TODO : other iterations for each data_case

    # DCASE
    # 연도별로 데이터셋을 대분류
    data_dict[data_case[0]] = dict()
    label_dict[data_case[0]] = dict()

    for year in dcase_years :
        data_dict[data_case[0]][year] = dict()
        label_dict[data_case[0]][year] = dict()
        
        data_dict[data_case[0]][year], label_dict[data_case[0]][year] = get_from_dcase(machine, year, base_dir = base_dir)
    
    # MIMII
    # 데시벨 별로 데이터셋을 대분류
    data_dict[data_case[1]] = dict()
    label_dict[data_case[1]] = dict()

    for decibel in mimii_dbs :
        data_dict[data_case[1]][decibel] = dict()
        label_dict[data_case[1]][decibel] = dict()

        data_dict[data_case[1]][decibel], label_dict[data_case[1]][decibel] = get_from_mimii(machine, decibel, base_dir = base_dir)

    # TODO : Another additional data

    return data_dict, label_dict


def get_from_dcase(machine, year, base_dir) :
    '''
    DCASE 데이터셋의 디렉토리 경로와 label을 추출
    dev, eval, add 데이터셋으로 중분류
    train, test 데이터셋으로 소분류

    ex) data_dict["train"][”normal”] = {real_path}

    inputs
    machine : string, 분류할 기계의 종류
    year : string, 데이터셋의 연도
    base_dir : string, 데이터셋이 저장된 디렉토리 경로

    outputs
    data_dict : dictionary, 데이터셋의 디렉토리 경로, { key : dataset_class, value : data_path}
    labels : dictionary, 데이터셋의 label { key : data_path, value : label_list}
    '''

    data_dict = dict()
    label_dict = dict()
    for dataset_class in dcase_dev_eval_additional : 

        if year in dcase_year_contains_only_dev and dataset_class != dcase_dev_eval_additional[0] :
            continue

        data_path = base_dir + "DCASE/" + year + "/" # data/DCASE/2020/
        data_path = data_path + dataset_class + "/" + machine + "/" # data/DCASE/2020/dev/fan/

        data_dict[dataset_class] = dict()
        label_dict[dataset_class] = dict()

        for each_data_class in train_or_test :
            tmp_data_path = data_path + each_data_class + "/" # tmp_data_path : "data/DCASE/2020/dev/fan/train/"
            
            if dataset_class == dcase_dev_eval_additional[1] and each_data_class == train_or_test[0] :
                continue
            if dataset_class == dcase_dev_eval_additional[2] and each_data_class == train_or_test[1] :
                continue
            
            data_dict[dataset_class][each_data_class] = list()
            label_dict[dataset_class][each_data_class] = dict()

            data_dict[dataset_class][each_data_class], label_dict[dataset_class][each_data_class] = get_data_paths_and_labels_from_edge_dir(tmp_data_path)

    return data_dict, label_dict
        

def get_from_mimii(machine, decibel, base_dir) :
    '''
    MIMII 데이터셋의 디렉토리 경로와 label을 추출

    ex) data_dict["normal"] = {real_path}

    inputs
    machine : string, 분류할 기계의 종류
    year : string, 데이터셋의 연도
    base_dir : string, 데이터셋이 저장된 디렉토리 경로

    outputs
    data_dict : dictionary, 데이터셋의 디렉토리 경로, { key : dataset_class, value : data_path}
    labels : dictionary, 데이터셋의 label { key : data_path, value : label_list}
    '''

    data_dict = dict()
    label_dict = dict()

    for each_data_class in data_class :
        data_dict[each_data_class] = list()


    data_path = base_dir + "MIMII/" + decibel + "/" + machine + '/' # data/MIMII/data_-6_dB/fan/
    for each_id in mimii_ids :
        for each_data_class in data_class :
            if each_data_class == data_class[2] :
                break
            tmp_data_path = data_path + each_id + "/" # data/MIMII/data_-6_dB/fan/id_00/
            tmp_data_path = tmp_data_path + each_data_class + "/" # tmp_data_path : "data/MIMII/data_-6_dB/fan/id_00/normal/"

            tmp_data_list, tmp_label_dict = get_data_paths_and_labels_from_edge_dir(tmp_data_path)

            data_dict[each_data_class].extend(tmp_data_list)
            label_dict.update(tmp_label_dict)

    return data_dict, label_dict
        

# TODO : other dataset
def get_data_from_other_dataset(data_path) :
    pass


def get_data_paths_and_labels_from_edge_dir(data_path) :
    '''
    data_path 에 있는 모든 *.wav 파일의 경로와 label을 dircionary로 반환
    label_dict 가 의미 없는 값일 수 있으나, label 값을 binary 로 mapping 하기 위해 사용
    경로 상에서 의미있는 값은 출력하게 해놨음

    ex) data_list["normal"] = {real_path}

    inputs
    data_path : string, *.wav 파일이 저장된 디렉토리 경로

    outputs
    data_list : list, 디렉토리에 있는 모든 *.wav 파일의 경로를 저장한 리스트
    label_dict : dictionary, { key : 디렉토리에 있는 모든 *.wav 파일의 경로, value : label }
    '''
    
    data_list = list()
    label_dict = dict()


    data_path_list = os.listdir(data_path) # data_path에 있는 모든 파일 리스트
    for each_data in data_path_list :
        each_data_full_path = data_path + each_data # data_path에 있는 각 파일의 전체 경로
        dirname = data_path.split("/")[-2] # data_path의 마지막 디렉토리 이름

        data_list.append(each_data_full_path)
        if data_class[0] in each_data or dirname == data_class[0] :
            # data_dict[data_class[0]].append(each_data_full_path)
            label_dict[each_data_full_path] = 1
        elif data_class[1] in each_data or "anomal" in each_data or dirname == data_class[1] :
            # data_dict[data_class[1]].append(each_data_full_path)
            label_dict[each_data_full_path] = -1
        else : 
            # data_dict[data_class[2]].append(each_data_full_path)
            label_dict[each_data_full_path] = 0
        
    print(data_path.split('/')[data_path.split('/').index("data"):-1])
    
    return data_list, label_dict


if __name__ == "__main__" :
    data_dict, label_dict = get_data_paths_and_labels_from_machine("fan")
    
    # print(data_dict)
    # print(label_dict)