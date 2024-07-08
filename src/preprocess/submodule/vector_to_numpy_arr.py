import numpy as np
from tqdm import tqdm

def vector_to_numpy_arr (vector_arr) :
    '''
    convert list of vectors to numpy array

    input
    vector_arr : list of numpy.array
        list of vectors

    output
    dataset : numpy.array
        numpy array of vectors
    '''
    len_list = len(vector_arr)

    for idx in tqdm(range(len_list)) :
        vector_list = vector_arr[idx]

        if idx == 0 :
            dataset = np.zeros((len_list, vector_list.shape[0], vector_list.shape[1]), float)
        
        dataset[idx, :, :] = vector_list

    return dataset