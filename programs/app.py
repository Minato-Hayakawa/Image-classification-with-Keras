import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib

data_dir=pathlib.Path('Input Picture Path')
new_model = tf.models.load_model('raccoon_redpanda.h5')