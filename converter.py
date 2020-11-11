import tensorflow as tf

mobile_dir = './saved_model/MobileNetV2'
simple_dir = './saved_model/simple'
Res_dir = './saved_model/Res'
VGG_dir = './saved_model/VGG16'
Inc_dir = './saved_model/IncV3'
den_dir = './saved_model/dense'
res50_dir = './saved_model/ResNet50'
# converter = tf.lite.TFLiteConverter.from_saved_model(mobile_dir)
# tflite_model = converter.convert()
# open("./saved_model/mobilenet.tflite", "wb").write(tflite_model)

# converter = tf.lite.TFLiteConverter.from_saved_model(simple_dir)
# tflite_model = converter.convert()
# open("./saved_model/simple.tflite", "wb").write(tflite_model)

converter = tf.lite.TFLiteConverter.from_saved_model(VGG_dir) #create converter class by directory of the model
tflite_model = converter.convert() #convert it
open("./saved_model/VGG16.tflite", "wb").write(tflite_model)#save it to the file

# converter = tf.lite.TFLiteConverter.from_saved_model(VGG_dir)
# tflite_model = converter.convert()
# open("./saved_model/VGG.tflite", "wb").write(tflite_model)