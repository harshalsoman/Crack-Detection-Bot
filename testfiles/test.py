import os
import gc
import sys

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#import seaborn as sns
#import pickle
#import skimage
#from skimage.feature import greycomatrix, greycoprops
#from skimage.filters import sobel
#from skimage import color

#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OneHotEncoder

from keras import layers
import keras.backend as K
from keras.models import Sequential, Model
from keras.preprocessing import image
from keras.layers import Input, Dense, Activation, Dropout
from keras.layers import Flatten, BatchNormalization, Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.nasnet import NASNetMobile
from tensorflow.keras.models import Sequential, load_model, save_model
from PIL import Image
from tqdm import tqdm
import random as rnd
import cv2
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims

#!pip install livelossplot
#from livelossplot import PlotLossesKeras

#%matplotlib inline
#train_path = '/content/drive/MyDrive/imageDATA/Train/'
#val_path = '/content/drive/MyDrive/imageDATA/Validation/'
#test_path = '/content/drive/MyDrive/imageDATA/Test/'
train_path = 'C:/Users/Dell/Desktop/imageDATA/Train/'
val_path = 'C:/Users/Dell/Desktop/imageDATA/Validation/'
test_path = 'C:/Users/Dell/Desktop/imageDATA/Test/'


train_df_defective = pd.DataFrame(os.listdir(train_path+'/Defective'))
val_df_defective = pd.DataFrame(os.listdir(val_path+'/Defective'))
test_df_defective = pd.DataFrame(os.listdir(test_path+'/Defective'))
train_df_undefective = pd.DataFrame(os.listdir(train_path+'/Non Defective'))
val_df_undefective = pd.DataFrame(os.listdir(val_path+'/Non Defective'))
test_df_undefective = pd.DataFrame(os.listdir(test_path+'/Non Defective'))


train_df_defective = train_df_defective[0:1000]
val_df_defective = val_df_defective[0:300]
test_df_defective = test_df_defective[300:600]
train_df_undefective = train_df_undefective[0:1000]
val_df_undefective = val_df_undefective[0:300]
test_df_undefective = test_df_undefective[300:600]



defective=pd.concat([train_df_defective,test_df_defective, val_df_defective], axis=0)
undefective=pd.concat([train_df_undefective,test_df_undefective,val_df_undefective], axis=0)

widths, heights = [], []
defective_path_images = []

for path in tqdm(defective[0]):
    try:
        width, height = Image.open('C:/Users/Dell/Desktop/imageDATA/Train/Defective/' + path).size
        widths.append(width)
        heights.append(height)
        defective_path_images.append('C:/Users/Dell/Desktop/imageDATA/Train/Defective/' + path)
    except:
        try:
            width, height = Image.open('C:/Users/Dell/Desktop/imageDATA/Validation/Defective/' + path).size
            widths.append(width)
            heights.append(height)
            defective_path_images.append('C:/Users/Dell/Desktop/imageDATA/Validation/Defective/' + path)
        except:
            width, height = Image.open('C:/Users/Dell/Desktop/imageDATA/Test/Defective/' + path).size
            widths.append(width)
            heights.append(height)
            defective_path_images.append('C:/Users/Dell/Desktop/imageDATA/Test/Defective/' + path)
            continue

df_defective = pd.DataFrame()
df_defective["width"] = widths
df_defective["height"] = heights
df_defective["path"] = defective_path_images
df_defective["dimension"] = df_defective["width"] * df_defective["height"]

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.10,
    brightness_range=[0.6,1.4],
    channel_shift_range=0.7,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

widths, heights = [], []
undefective_path_images = []

for path in tqdm(undefective[0]):
    try:
        width, height = Image.open('C:/Users/Dell/Desktop/imageDATA/Test/Non Defective/' + path).size
        widths.append(width)
        heights.append(height)
        undefective_path_images.append('C:/Users/Dell/Desktop/imageDATA/Test/Non Defective/' + path)
    except:
        try:
            width, height = Image.open('C:/Users/Dell/Desktop/imageDATA/Train/Non Defective/' + path).size
            widths.append(width)
            heights.append(height)
            undefective_path_images.append('C:/Users/Dell/Desktop/imageDATA/Train/Non Defective/' + path)
        except:
            width, height = Image.open('C:/Users/Dell/Desktop/imageDATA/Validation/Non Defective/' + path).size
            widths.append(width)
            heights.append(height)
            undefective_path_images.append('C:/Users/Dell/Desktop/imageDATA/Validation/Non Defective/' + path)
            continue

df_undefective = pd.DataFrame()
df_undefective["width"] = widths
df_undefective["height"] = heights
df_undefective["path"] = undefective_path_images
df_undefective["dimension"] = df_undefective["width"] * df_undefective["height"]

df_defective["Target"] = "Defective"
df_undefective["Target"] = "Non Defective"

combined  = [df_defective, df_undefective]
combined_result = pd.concat(combined)

print(combined_result)
combined_result = combined_result.sample(frac = 1)

result_train = combined_result[0:1800]
result_val = combined_result[2001:2600]
result_test = combined_result[2601:3200]

train_generator = datagen.flow_from_dataframe(
    dataframe=result_train,
    directory = None,
    x_col = "path",
    y_col = "Target",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="binary",
    shuffle=True,
    seed=42
)
val_generator = val_datagen.flow_from_dataframe(
    dataframe=result_val,
    directory = None,
    x_col = 'path',
    y_col = 'Target',
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="binary",
    shuffle=True,
    seed=42
)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=result_test,
    directory = None,
    x_col = 'path',
    y_col='Target',
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=1,
    class_mode="binary",
    shuffle=True,
    seed=42
)
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
import keras
restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
#output = restnet.layers[-1].output
#output = keras.layers.Flatten()(output)
#restnet = Model(restnet.input, output=predictions)
for layer in restnet.layers:
    layer.trainable = False
restnet.summary()
# Flatten the output layer to 1 dimension
x2= layers.Flatten()(restnet.output)

# Add a fully connected layer with 512 hidden units and ReLU activation
x2 = layers.Dense(512, activation='relu')(x2)

# Add a dropout rate of 0.5
x2 = layers.Dropout(0.5)(x2)

# Add a final sigmoid layer with 1 node for classification output
x2 = layers.Dense(1, activation='sigmoid')(x2)

model2 = tf.keras.models.Model(restnet.input, x2)

model2.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])
#print(result_test.path[0])
reshist = model2.fit_generator(
      train_generator,
      validation_data=val_generator,
      epochs=25,
      verbose=2)
model2.save("my_h5_model_resnet50epochs.h5")