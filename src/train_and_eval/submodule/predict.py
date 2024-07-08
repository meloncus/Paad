import tensorflow as tf
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
sys.path.append(SRC_DIR)

from train_and_eval.submodule.metric import get_mse

def predict_only_autoencoder(model, data, threshold = 0.5) :
    '''
    predict whether data is normal or not using autoencoder

    input
    model : tensorflow.keras.Model, autoencoder model
    data : pandas.DataFrame, data to predict
    threshold : float, threshold for anomaly detection

    output
    predictions : numpy.array, predicted labels
    '''
    # this part is getting loss from autoencoder, need to develop for more expandable
    reconstruction = model.predict(data) 
    loss = get_mse(data, reconstruction)

    # if loss is less than threshold, it is normal data (1), otherwise it is abnormal data (-1)
    boolean_tensor = tf.math.less(loss, threshold)
    predictions = tf.where(boolean_tensor, 1, -1)

    return predictions

if __name__ == "__main__" :
    print(SRC_DIR)