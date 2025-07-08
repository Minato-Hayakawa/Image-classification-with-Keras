import numpy as np
import tensorflow as tf
import pathlib

data_dir=pathlib.Path('Input Picture Path')
img_hight=180
img_width=180

predict_ds=tf.keras.utils.image_dataset_from_directory(
    data_dir,
    subset="validation",
    image_size=(img_hight,img_width),
    batch_size=4
)
class_names = predict_ds.class_names

model = tf.models.load_model('raccoon_redpanda.h5')
prediction=model.predict(predict_ds)

print("image classification with Keras")

if(prediction[0]==0):
    print("This image is raccoon")
elif(prediction[0]==1):
    print("This image is red panda")