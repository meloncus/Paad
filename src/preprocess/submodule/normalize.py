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
    '''
    target을 max_value와 min_value를 이용하여 normalize 하는 함수

    input :
        target : numpy.array, normalize할 target
        max_value : float, target의 최대값
        min_value : float, target의 최소값

    output :
        normalized_target : numpy.array, normalize된 target
    '''
    if max_value is None :
        max_value = np.max(target)
        min_value = np.min(target)
        
    return (target - min_value) / (max_value - min_value)