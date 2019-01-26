import tensorflow as tf
import numpy as np
from PIL import Image

_forceCPU_ = False

def isCPUFprced():
    return _forceCPU_


def setCPUFlag(forceCPU = True):
    _forceCPU_ = forceCPU

def getDevice(CPU, index):
    if _forceCPU_ or CPU:
        return "/cpu:0"
    else:
        return "/gpu:"+str(index)

def createSession(CPU = False):
    setCPUFlag(CPU)

    
    
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction=0.8
  
    sess = tf.InteractiveSession("", config=config)
    return sess
