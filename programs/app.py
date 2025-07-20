import numpy as np
import tensorflow as tf
import pathlib

data_dir=pathlib.Path('C:/vscode/python/Keras2/Image classification with Keras/newimages')
img_hight=180
img_width=180
predict_ds=tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    batch_size=4,
    image_size=(img_hight,img_width),
    shuffle=False
)
class_names = ["raccoon","red panda"]

model = tf.keras.models.load_model('C:/vscode/python/Keras2/Image classification with Keras/model/raccoon_redpanda.h5')
prediction=model.predict(predict_ds)  
for i,prediction_array in enumerate(prediction):
    predicted_index_classes=np.argmax(prediction_array)
    predicted_class_names=class_names[predicted_index_classes]
    reliability=prediction_array[predicted_index_classes]
    print(f"{i+1}番目の画像は{predicted_class_names}です。信頼度={reliability:.4f}")