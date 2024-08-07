import os
import tensorflow as tf
import numpy as np
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from prepare_dataset.prepare_dataset import get_data_paths_and_labels_from_machine as get_data_from_machine
from preprocess.pd_preprocess import get_specific_data, default_preprocess, preprocess_for_all_feat, put_mel_from_dataframe
from utils.utils_general import load_pickle_data_for_all_feat


from preprocess.submodule.file_to_vector import file_to_vector_mel
from preprocess.submodule.normalize import min_max_normalization
from models.submodule.autoencoder import AutoEncoder
# from train_and_eval.submodule.predict import predict_only_autoencoder
from train_and_eval.submodule.metric import get_mse
from preprocess.submodule.train_test_split import get_train_test_vector_all_feat

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DATA_PATH = "tmp/audio/"
TMP_DATA_PATH = "tmp/data/"
MODEL_PATH = "tmp/model/autoencoder_weights.h5"

def get_data_paths (machine = "fan") :
    '''
    데이터 경로와 label을 가져오는 함수

    input :
        machine : string, 기계 이름

    output :
        data_paths : list, 데이터 경로 리스트
        labels : list, label 리스트
        or
        None
        None
    '''
    try :
        data_paths, labels = get_data_from_machine(machine)
    except Exception as e :
        print("Error in getting data paths")
        print(e)
        return None, None

    return data_paths, labels


def df_preprocess_from_paths (data_paths, labels) :
    '''
    데이터 경로와 label을 받아서 전처리된 dataframe 을 만드는 함수

    input :
        data_paths : list, 데이터 경로 리스트
        labels : list, label 리스트

    output :
        df : pandas.DataFrame, 전처리된 dataframe
    '''
    data_paths = get_specific_data(data_paths)
    labels  = get_specific_data(labels)

    return default_preprocess(data_paths, labels)


def feat_train_test_preprocess (df, train_ratio = 0.8) : # not work yet
    '''
    dataframe 을 train, test 로 나누는 함수 (미구현)

    input :
        df : pandas.DataFrame, dataframe
        train_ratio : float, train 비율

    output :
        train_df : pandas.DataFrame, train dataframe
        test_df : pandas.DataFrame, test dataframe
    '''
    train_df = df.sample(frac = train_ratio)
    test_df = df.drop(train_df.index)

    train_df.to_csv("tmp/train_df.csv", index = False)
    test_df.to_csv("tmp/test_df.csv", index = False)

    return train_df, test_df

def save_df_only_ae (df) :
    '''
    TMP_DATA_PATH/df_meta_5050.pkl로 dataframe 을 저장하는 함수

    input :
        df : pandas.DataFrame, dataframe

    output :
        None
    '''
    df.to_pickle(TMP_DATA_PATH + "df_meta_5050.pkl")


def preprocess_for_all_feat_except_ae (df) :
    '''
    submodule/preprocess/pd_preprocess.py 의 preprocess_for_all_feat 함수를 이용하여 dataframe 을 전처리하는 함수

    input :
        df : pandas.DataFrame, dataframe

    output :
        dataframe : pandas.DataFrame, 전처리된 dataframe
    '''
    return preprocess_for_all_feat(df)


def training_all_feat_except_ae (meta_data_path = TMP_DATA_PATH + "df_meta_5050.pkl") :
    '''
    전처리된 dataframe 을 이용하여 모든 feature 에 대해 학습하는 함수

    input :
        meta_data_path : string, meta data 경로

    output :
        df_fan : pandas.DataFrame, fan 데이터
        df_fan_means : pandas.DataFrame, mean 으로 전처리된 fan 데이터
    '''
    # caution : meta data 를 불러오지만 load_pickle_data_for_all_feat() funciton 에 의해 불러오지 않음
    df_list = load_pickle_data_for_all_feat()
    if df_list is None :
        raise Exception("Error in loading pickle data")
    if len(df_list) == 1 :
        try : # ignore if meta data is not exist, not yet developed
            df = pd.read_pickle(meta_data_path)

            df_fan, df_fan_means = preprocess_for_all_feat_except_ae(df)

            df_fan = get_train_test_vector_all_feat(df_fan)
            df_fan_means = get_train_test_vector_all_feat(df_fan_means)
        except Exception as e :
            print("Error in loading meta data")
            print(e)
            return None, None
    else :
        df_fan_means = df_list[0]
        df_fan = df_list[1]

    return df_fan, df_fan_means


def compact_serving_autoencoder_mel (audio_data_path = DATA_PATH, model_path = MODEL_PATH) :
    '''
    autoencoder 를 이용하여 audio data 를 학습하고, 예측하는 함수

    input :
        audio_data_path : string, audio data path
        model_path : string, model path

    output :
        prediction : numpy.array, 예측값
        or
        None
    '''
    # load audio mel spectrum from data path
    try :
        data_paths = os.listdir(audio_data_path)
        input_data = [ file_to_vector_mel(os.path.join(audio_data_path, data_path)) for data_path in data_paths ]
        input_data_np = np.array(input_data)
        print(input_data_np.shape)
    except Exception as e :
        print("Error in loading audio data")
        print(e)
        return None
    
    # preprocess audio data
    input_data = [ min_max_normalization(data) for data in input_data ]
    input_data = tf.cast(input_data, tf.float32)

    # load autoencoder
    autoencoder = AutoEncoder(input_data[0].shape)
    tmp_input = np.zeros((1, input_data[0].shape[0], input_data[0].shape[1]))
    autoencoder(tmp_input)
    try : 
        autoencoder.load_weights(model_path)
    except Exception as e :
        print("Error in loading autoencoder")
        print(e)
        return None

    # predict
    try : 
        # prediction = predict_only_autoencoder(autoencoder, input_data)
        reconstructions = autoencoder.predict(input_data)
        loss = get_mse(input_data, reconstructions)
    except Exception as e :
        print("Error in predicting")
        print(e)
        return None

    # return prediction
    # return prediction

    return loss, reconstructions


if __name__ == "__main__" :
    # TODO : merge tests

    # test1
    loss, reconstructions = compact_serving_autoencoder_mel()
    print(os.listdir(DATA_PATH))
    print("Loss : ", loss)
    print(type(reconstructions))
    # print("Reconstructions : ", reconstructions)

    # test2
    # data_paths, labels = get_data_paths()
    # df = df_preprocess_from_paths(data_paths, labels)
    # print(df.head())
    # print(df.shape)
    # save_df(df)