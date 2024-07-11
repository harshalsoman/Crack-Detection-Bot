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
from sklearn.model_selection import train_test_split
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


#train_df_defective = train_df_defective[0:1000]
#val_df_defective = val_df_defective[0:300]
#test_df_defective = test_df_defective[300:600]
#train_df_undefective = train_df_undefective[0:1000]
#val_df_undefective = val_df_undefective[0:300]
#test_df_undefective = test_df_undefective[300:600]




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

combined_result
combined_result = combined_result.sample(frac = 1)

result_train = combined_result[0:1500]
result_val = combined_result[2001:2500]
result_test = combined_result[2601:3100]

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
    batch_size=32,
    class_mode="binary",
    shuffle=True,
    seed=42
)



model = load_model('C:/Users/Dell/PycharmProjects/RailwayCrack/my_h5_model_new.h5')
preds = model.predict_generator(test_generator)
labels = test_generator.labels
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


y_pred = np.argmax(preds, axis=1)

print('Confusion Matrix')
print(confusion_matrix(test_generator.labels, y_pred))

print('Classification Report')
target_names = ['Defective', 'Non Defective']
print(classification_report(test_generator.labels, y_pred, target_names=target_names))
res = tf.math.confusion_matrix(labels, preds)
#print(res)