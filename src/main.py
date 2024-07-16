import os
import tensorflow as tf
import numpy as np

from prepare_dataset.prepare_dataset import get_data_paths_and_labels_from_machine as get_data_from_machine
from preprocess.pd_preprocess import get_specific_data, default_preprocess, put_mel_from_dataframe

from preprocess.submodule.file_to_vector import file_to_vector_mel
from preprocess.submodule.normalize import min_max_normalization
from models.submodule.autoencoder import AutoEncoder
# from train_and_eval.submodule.predict import predict_only_autoencoder
from train_and_eval.submodule.metric import get_mse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DATA_PATH = "tmp/audio/"
TMP_DATA_PATH = "tmp/data/"
MODEL_PATH = "tmp/autoencoder_weights.h5"

def get_data_paths (machine = "fan") :
    try :
        data_paths, labels = get_data_from_machine(machine)
    except Exception as e :
        print("Error in getting data paths")
        print(e)
        return None, None

    return data_paths, labels


def df_preprocess_from_paths (data_paths, labels) :
    data_paths = get_specific_data(data_paths)
    labels  = get_specific_data(labels)

    return default_preprocess(data_paths, labels)


def feat_train_test_preprocess (df, train_ratio = 0.8) : # not work yet
    train_df = df.sample(frac = train_ratio)
    test_df = df.drop(train_df.index)

    train_df.to_csv("tmp/train_df.csv", index = False)
    test_df.to_csv("tmp/test_df.csv", index = False)

    return train_df, test_df

def save_df (df) :
    df.to_pickle(TMP_DATA_PATH + "df_meta_5050.pkl")


def compact_serveing_autoencoder_mel (audio_data_path = DATA_PATH, model_path = MODEL_PATH) :
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
    # loss, reconstructions = compact_serveing_autoencoder_mel()
    # print(os.listdir(DATA_PATH))
    # print("Loss : ", loss)
    # print("Reconstructions : ", reconstructions)

    # test2
    data_paths, labels = get_data_paths()
    df = df_preprocess_from_paths(data_paths, labels)
    print(df.head())
    print(df.shape)
    # save_df(df)