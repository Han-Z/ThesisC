import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
mobile_dir = './saved_model/MobileNetV2'

#This is model quantization script. It convert float Keras models to fully integer model.
#however, it requires representative dataset.
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range = 0.3,
    zoom_range = 0.3,
    horizontal_flip=True
)

train_flow = train_datagen.flow_from_directory(#use data flow from directory to create the representative
    './data/train/', #the path to the dataset 
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(train_flow.next()[0]).batch(1).take(600):
    yield [input_value]

converter = tf.lite.TFLiteConverter.from_saved_model(mobile_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
tflite_model = converter.convert()
open("./saved_model/mobile_q.tflite", "wb").write(tflite_model)