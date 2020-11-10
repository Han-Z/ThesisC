from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, load_img,img_to_array
from keras.models import Sequential, Model
from keras.layers import Dense,Flatten,Dropout,Input
from keras.layers.convolutional import Conv2D,MaxPooling2D
import keras.layers as layers
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
from keras.models import load_model
from keras import backend as K
import cv2
import numpy as np


train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range = 0.3,
    zoom_range = 0.3,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)

train_flow = train_datagen.flow_from_directory(
    './data/train/', #replace this path to your own dataset
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    horizontal_flip=False
)

test_flow = test_datagen.flow_from_directory(
    './data/test/',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

conv_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
conv_base.trainable = False
model = keras.Sequential()
model.add(conv_base)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Flatten())
model.add(Dropout(0.25))

model.add(layers.Dense(100, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(52, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(26, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit_generator(train_flow, epochs=10)

model.save('./saved_model/MobileNetV2')
scores = model.evaluate(test_flow)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
