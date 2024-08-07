import numpy as np
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(SRC_DIR)

from train_and_eval.submodule.metric import get_mse, calculate_stats
from train_and_eval.submodule.predict import predict_only_autoencoder
from models.submodule.autoencoder import AutoEncoder

def train_and_eval_for_autoencoder(
        train_set, 
        val_set, 
        y_val, 
        input_dim, 
        device = '', 
        feat = '', 
        epochs = 50, 
        batch_size = 56,
        save_mode = False, 
        save_path = None,
        ) :
    '''
    train autoencoder and evaluate it

    input
    train_set : numpy.array, training set
    val_set : numpy.array, validation set
    y_val : numpy.array, validation labels
    input_dim : int, input dimension
    device : string, device name
    feat : string, feature label
    epochs : int, number of epochs

    output
    autoencoder : tensorflow.keras.Model, trained autoencoder model
    df : pandas.DataFrame, calculated statistics
    '''
    '''
    Autoencoder를 학습하고 평가하는 함수

    input : 
        train_set : numpy.array, training set
        val_set : numpy.array, validation set
        y_val : numpy.array, validation labels
        input_dim : int, input dimension
        device : string, device name
        feat : string, feature label
        epochs : int, number of epochs

    output :
        autoencoder : tensorflow.keras.Model, trained autoencoder model
        df : pandas.DataFrame, calculated metrics
    '''
    # initialize autoencoder model
    autoencoder = AutoEncoder(input_dim)
    autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error')

    history  = autoencoder.fit(
        train_set,
        val_set,
        epochs = epochs,
        shuffle = True,
        batch_size = batch_size,
        validation_split = 0.1
    )

    print("Encoder :  ", autoencoder.encoder.summary())
    print("Decoder :  ", autoencoder.decoder.summary())

    reconstructions = autoencoder.predict(train_set)
    train_loss = get_mse(train_set, reconstructions)

    cut_off = np.mean(train_loss) + np.std(train_loss)
    print("Cutoff : {}\n".format(cut_off), end = '')
    print("Prop. of Training data over Threshold : {}\n".format(np.sum(train_loss > cut_off) / len(train_loss)))

    predictions = predict_only_autoencoder(autoencoder, val_set, cut_off)

    df = calculate_stats(predictions, y_val, "Machine", feat, "AE")

    # save model
    if save_mode :
        autoencoder.save_weights("tmp/autoencoder_weights.h5")
        print("Model saved at {}".format(save_path))
    
    return autoencoder, df