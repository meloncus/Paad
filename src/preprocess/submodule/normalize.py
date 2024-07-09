import numpy as np


def min_max_normalization (target, max_value = None, min_value = None) :
    '''
    normalize target using max_value and min_value

    input
    target : numpy.array, target to normalize
    max_value : float, max value of target
    min_value : float, min value of target

    output
    normalized_target : numpy.array, normalized target
    '''
    if max_value is None :
        max_value = np.max(target)
        min_value = np.min(target)
        
    return (target - min_value) / (max_value - min_value)