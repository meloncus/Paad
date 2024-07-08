from sklearn import metrics
import pandas as pd
import numpy as np

def calculate_stats (predictions, labels, machine, feat_label, model = "AE") :
    '''
    calculate auc, precision, recall, f1 for abnormal

    input
    predictions : numpy.array, predicted labels
    labels : numpy.array, true labels
    machine : string, machine name
    feat_label : string, feature label ?
    model : string, model name

    output
    df : pandas.DataFrame, calculated statistics
    '''
    auc_score = metrics.roc_auc_score(labels, predictions)
    precision = metrics.precision_score(labels, predictions)
    recall = metrics.recall_score(labels, predictions)
    f1_for_abnormal = metrics.f1_score(labels, predictions, pos_label = -1, average = "binary")

    results = {
        "Model" : [model],
        "Machine" : [machine],
        "Feature_label" : [feat_label],
        "AUC" : [auc_score],
        "Precision" : [precision],
        "Recall" : [recall],
        "F1 for Abnormal" : [f1_for_abnormal],
    }

    df = pd.DataFrame(results)

    return df

def get_mse(source, target):
    '''
    get mean square error between source and target

    input
    source : numpy.array, source image
    target : numpy.array, target image

    output
    mse : numpy.array, mean square error between source and target
    '''
    # Returns the mean square error for each image in the array
    # TODO : I think this function is noly fit for autoencoder... recommand asap
    # TODO : use another optimized algorithm or library function
    return np.mean(np.power(source - target, 2), axis=(1,2)) 