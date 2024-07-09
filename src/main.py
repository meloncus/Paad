import os
import tensorflow as tf
import numpy as np

from prepare_dataset.prepare_dataset import get_data_paths_and_labels_from_machine as get_data_from_machine
from preprocess.pd_preprocess import get_specific_data, get_flatten_data, get_label_from_flatten_specifics, get_dataframe_from_flatten_data_and_label, put_mel_from_dataframe

from preprocess.submodule.file_to_vector import file_to_vector_mel
from preprocess.submodule.normalize import min_max_normalization
from models.submodule.autoencoder import AutoEncoder
# from train_and_eval.submodule.predict import predict_only_autoencoder
from train_and_eval.submodule.metric import get_mse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DATA_PATH = "tmp/audio/"
MODEL_PATH = "tmp/autoencoder_weights.h5"

def get_data_paths (machine = "fan") :
    try :
        data_paths, labels = get_data_from_machine(machine)
    except Exception as e :
        print("Error in getting data paths")
        print(e)
        return None, None

    return data_paths, labels


def df_from_paths (data_paths, labels) :
    data_paths = get_specific_data(data_paths)
    labels  = get_specific_data(labels)

    data_paths = get_flatten_data(data_paths)
    labels = get_label_from_flatten_specifics(data_paths, labels)

    df = get_dataframe_from_flatten_data_and_label(data_paths, labels)
    
    print("Mel Spectrogram is being put into dataframe...")
    df = put_mel_from_dataframe(df)

    return df


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

    return loss


if __name__ == "__main__" :
    prediction = compact_serveing_autoencoder_mel()
    print(os.listdir(DATA_PATH))
    print(prediction)

    # data_paths, labels = get_data_paths()
    # df = df_from_paths(data_paths, labels)
    # print(df.head())