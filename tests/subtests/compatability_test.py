from tensorflow.python.client import device_lib
import tensorflow as tf
import torch

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_available_gpus_count():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    return len(physical_devices)

def get_available_gpus_for_torch():
    print('Is CUDA available for PyTorch:', torch.cuda.is_available())
    print('Num GPUs Available for PyTorch:', torch.cuda.device_count())


if __name__ == "__main__":
    try : 
        print(get_available_gpus())
        print("Number of GPUs: ", get_available_gpus_count())
        get_available_gpus_for_torch()
    except Exception as e:
        print(e)