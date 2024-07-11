# importing the libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
#from tensorflow.keras.models import Sequential, load_model, save_model
# for reading and displaying images
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
#%matplotlib inline

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

# torchvision for pre-trained models
#from torchvision import models
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

df_defective["Target"] = 0
df_undefective["Target"] = 1

combined  = [df_defective, df_undefective]
combined_result = pd.concat(combined)

combined_result
combined_result = combined_result.sample(frac = 1)

result_train = combined_result[0:5000]
result_val = combined_result[5001:600]
result_test = combined_result[600:6000]

# loading training images
train_img = []
for img_name in tqdm(result_train['path']):
    # defining the image path
    #image_path = '../Hack Session/images/' + img_name
    # reading the image
    img = imread(img_name)
    # normalizing the pixel values
    img = img/255
    # resizing the image to (224,224,3)
    img = resize(img, output_shape=(32,32,3), mode='constant', anti_aliasing=True)
    # converting the type of pixel to float 32
    img = img.astype('float32')
    # appending the image into the list
    train_img.append(img)

# converting the list to numpy array
train_x = np.array(train_img)
print(train_x.shape)
train_y = result_train['Target'].values

# create validation set
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.1, random_state = 13, stratify=train_y)
print((train_x.shape, train_y.shape), (val_x.shape, val_y.shape))
torch.manual_seed(0)
train_x = train_x.reshape(4500, 3, 32, 32)
train_x  = torch.from_numpy(train_x)

# converting the target into torch format
train_y = train_y.astype(int)
train_y = torch.from_numpy(train_y)

# converting validation images into torch format
val_x = val_x.reshape(500, 3, 32, 32)
val_x  = torch.from_numpy(val_x)

# converting the target into torch format
val_y = val_y.astype(int)
val_y = torch.from_numpy(val_y)

import torch
import torch.nn as nn
import torch.nn.functional as F

in_channel = 3
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(720, 60)
        self.fc2 = nn.Linear(60, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
# defining the model
model = CNN()
# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.0001)
# defining the loss function
criterion = CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

print(model)
torch.manual_seed(0)

# batch size of the model
batch_size = 128

# number of epochs to train the model
n_epochs = 15

for epoch in range(1, n_epochs + 1):

    # keep track of training and validation loss
    train_loss = 0.0

    permutation = torch.randperm(train_x.size()[0])

    training_loss = []
    for i in tqdm(range(0, train_x.size()[0], batch_size)):

        indices = permutation[i:i + batch_size]
        batch_x, batch_y = train_x[indices], train_y[indices]

        if torch.cuda.is_available():
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        optimizer.zero_grad()
        # in case you wanted a semi-full example
        #print(batch_x)
        outputs = model(batch_x)
        batch_y = batch_y.long()
        loss = criterion(outputs, batch_y)

        training_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    training_loss = np.average(training_loss)
    print('epoch: \t', epoch, '\t training loss: \t', training_loss)

# prediction for training set
prediction = []
target = []
permutation = torch.randperm(train_x.size()[0])
for i in tqdm(range(0, train_x.size()[0], batch_size)):
    indices = permutation[i:i + batch_size]
    batch_x, batch_y = train_x[indices], train_y[indices]

    if torch.cuda.is_available():
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

    with torch.no_grad():
        output = model(batch_x)

    softmax = torch.exp(output)
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    prediction.append(predictions)
    target.append(batch_y)

# training accuracy
accuracy = []
for i in range(len(prediction)):
    accuracy.append(accuracy_score(target[i], prediction[i]))

print('training accuracy: \t', np.average(accuracy))

# prediction for validation set
prediction_val = []
target_val = []
permutation = torch.randperm(val_x.size()[0])
for i in tqdm(range(0, val_x.size()[0], batch_size)):
    indices = permutation[i:i + batch_size]
    batch_x, batch_y = val_x[indices], val_y[indices]

    if torch.cuda.is_available():
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

    with torch.no_grad():
        output = model(batch_x)

    softmax = torch.exp(output)
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    prediction_val.append(predictions)
    target_val.append(batch_y)

# validation accuracy
accuracy_val = []
for i in range(len(prediction_val)):
    accuracy_val.append(accuracy_score(target_val[i], prediction_val[i]))
print('validation accuracy: \t', np.average(accuracy_val))
#model.save_model("my_h5_model_cnnmodeltf.h5")
torch.save(model.state_dict(), "my_h5_model_cnnmodel500015epochs.h5")

#path = "my_h5_model_cnnmodeltf.pth"
#torch.save(model, path)
