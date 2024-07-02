from tensorflow.python.client import device_lib
import tensorflow as tf

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_available_gpus_count():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    return len(physical_devices)

if __name__ == "__main__":
    try : 
        print(get_available_gpus())
        print("Number of GPUs: ", get_available_gpus_count())
    except Exception as e:
        print(e)