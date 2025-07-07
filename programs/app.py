import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib

data_dir=pathlib.Path('Input Picture Path')
img_hight=180
img_width=180
model = tf.models.load_model('raccoon_redpanda.h5')

predict_ds=tf.keras.utils.image_dataset_from_directory(
    data_dir,
    subset="validation"
    
)