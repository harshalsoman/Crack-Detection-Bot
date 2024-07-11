import requests
import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
import cv2
import threading
from keras.preprocessing import image
from skimage.io import imread
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.nn.functional as F
output = ''

def video_stream():
    vid = cv2.VideoCapture(0)
    global output
    while True:
        ret, output = vid.read()

        cv2.imshow('livestream', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()


def ml():
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(720, 1024)
            self.fc2 = nn.Linear(1024, 2)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(x.shape[0], -1)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return x

    # defining the model
    model = CNN()
    # new_model = load_model('C:/Users/Dell/PycharmProjects/RailwayCrack/my_h5_model_cnnmodel.h5')
    model.load_state_dict(torch.load("my_h5_model_cnnmodel5000.h5"))
    model.eval()
    while (True):
        global output
        output = np.expand_dims(output, axis=0)
        #img = imread(output)
        # normalizing the pixel values
        #print(output.shape)
        #output = output.astype(float)
        img = output
        # resizing the image to (224,224,3)
        img = resize(img, output_shape=(32,32,3), mode='constant', anti_aliasing=True)
        #print(img.shape)
        # converting the type of pixel to float 32
        #dsize = (32,32)
        #img= cv2.resize(img, dsize)
        img = img.astype('float32')
        # appending the image into the list
        #test_img.append(img)
        test_x = np.array(img)
        print(test_x.shape)
        test_x = test_x.reshape(1,3,32,32)
        test_x = torch.from_numpy(test_x)
        with torch.no_grad():
            output = model(test_x)

        softmax = torch.exp(output)
        prob = list(softmax.numpy())
        predictions = np.argmax(prob, axis=1)
        #y_pred = model.predict(pred_image)
        #print(y_pred)
        if (predictions == 0):
            print("Defective")

def main():
    t1 = threading.Thread(target=video_stream)
    t2 = threading.Thread(target=ml)

    t1.start()
    t2.start()

if __name__ == '__main__':
    main()

