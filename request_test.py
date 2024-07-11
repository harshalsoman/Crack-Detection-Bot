import requests
import cv2
import numpy as np
import torch
import cv2
#from keras.preprocessing import image
from skimage.io import imread
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.nn.functional as F

import imutils
from imutils.video import VideoStream
vid = cv2.VideoCapture(0)


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
while True:
    try:
        ret, output = vid.read()

        #ret, output = vid.read()
        # dsize = (224,224)
        # output = cv2.resize(output, dsize)
        # Display the resulting frame
        img = output / 255
        # resizing the image to (224,224,3)
        img = resize(img, output_shape=(32, 32, 3), mode='constant', anti_aliasing=True)
        # converting the type of pixel to float 32
        img = img.astype('float32')
        # appending the image into the list
        # test_img.append(img)
        test_x = np.array(img)
        test_x = test_x.reshape(1, 3, 32, 32)
        test_x = torch.from_numpy(test_x)
        with torch.no_grad():
            output1 = model(test_x)

        softmax = torch.exp(output1)
        prob = list(softmax.numpy())
        predictions = np.argmax(prob, axis=1)
        # y_pred = model.predict(pred_image)
        #print(output1)
        if (predictions == 1):
            print("Non Defective")
        else:
            print("Defective")
        cv2.imshow('livestream', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except cv2.error as error:
        print(error)




vid.release()
# Destroy all the windows
cv2.destroyAllWindows()