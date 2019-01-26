import tensorflow as tf
import numpy as np
from PIL import Image
import time
import random

import os
import sys

def create_session():
  config = tf.ConfigProto(log_device_placement=False)
  config.gpu_options.per_process_gpu_memory_fraction=0.8
  sess = tf.Session(config=config)
  return sess
