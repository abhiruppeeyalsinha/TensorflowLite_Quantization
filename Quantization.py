from lib2to3.pytree import convert
from pyexpat import model
from statistics import mode
from telnetlib import SE
from winreg import EnumValue
import tensorflow as tf
# from tensorflow.lite import TFLiteConverter
import keras,os
from keras.layers import Flatten,Dense,Dropout
from keras.models import load_model
from keras import datasets
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')



(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()

# print(X_test[5].shape)
# for i in range(10): 
#     print(y_train[i])

#     plt.imshow(X_train[i])
#     plt.show()



# print(f"test_x:- {x_test.shape}")

# x_train = X_train/255
# x_test = X_test/255
# # x_train = x_train.reshape(len(x_train), 28*28)
# # x_test = x_test.reshape(len(x_test), 28*28)

# print(f"test_x:{x_test.shape}")
# print(f"test_xy:{x_train.shape}")




# model = keras.Sequential([Flatten(input_shape=(28,28)),
# Dropout(0.5),
# Dense(100,activation='relu'),
# Dropout(0.5),
# Dense(10,activation='sigmoid')])


# model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",
# metrics = ["accuracy"])

# model.fit(x_train,y_train,epochs = 10)
# eva = model.evaluate(x_test,y_test)
# print("eva",eva)
# save_model = model.save("./saved_model")



# model_file = r"E:\CNN Project\TL\save model\Copy_model_e10.h5"
# model = load_model(model_file)
# print(model)
# print(mode.summary())

converter = tf.lite.TFLiteConverter.from_saved_model(r"E:\CNN Project\TL\save model\saved_model")
tflit_model_ = converter.convert()
print(f"tflite_model: {len(tflit_model_)}")

converter.optimizer = [tf.lite.Optimize.DEFAULT]
tflit_quant_model_ = converter.convert()
print(f"tflit_quant_model: {len(tflit_quant_model_)}")

with open("tf_lite_model_.tflite","wb") as save_model:
    save_model.write(tflit_model_)
with  open("tfLite_quant_model_.tfLite","wb") as save_model_:
    save_model_.write(tflit_quant_model_)