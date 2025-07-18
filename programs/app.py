import numpy as np
import tensorflow as tf
import pathlib

data_dir=pathlib.Path('Input Picture Path')
img_hight=180
img_width=180
predict_ds=tf.keras.utils.image_dataset_from_directory(
    data_dir,
    batch_size=4,
    image_size=(img_hight,img_width),
)
class_names = ["raccoon","red panda"]

model = tf.models.load_model('raccoon_redpanda.h5')
prediction=model.predict(predict_ds)  
raccoon_reliability=prediction[0].max
redpanda_reliability=prediction[1].max
for i in range(prediction[0].shape):
    if prediction[i][0]>prediction[i][1]:
        print("This image was predicted as raccoon reliability="+prediction[i][0])
    elif prediction[i][0]<prediction[i][1]:
        print("This image was predicted as red panda reliability="+prediction[i][0])