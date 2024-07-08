import numpy as np

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