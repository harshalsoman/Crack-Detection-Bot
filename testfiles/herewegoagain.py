import requests
import threading

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.transform import resize
#import vlc
#import pygame
#import playsound
output = ''
#y=[]
def video_stream():
    while True:
        try:

            vid = cv2.VideoCapture(0)
            global output
            while True:
                ret, output = vid.read()

                cv2.imshow('livestream', output)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if cv2.waitKey(1) & 0xFF == ord('x'):
                    exit(0)
            vid.release()
            cv2.destroyAllWindows()
        except cv2.error as error:
            print(error)



def ml():
    new_model = load_model('C:/Users/Dell/PycharmProjects/RailwayCrack/Aaditya_50epochs.h5')
    while (True):
        try:
            global output

            output = resize(output, output_shape=(224, 224, 3), mode='constant', anti_aliasing=True)
            img_batch = np.expand_dims(output, axis=0)
            pred_generator = tf.keras.preprocessing.image.ImageDataGenerator()
            pred_image = pred_generator.flow(
                img_batch,
                batch_size=1,
                shuffle=False
            )

            y_pred = new_model.predict(img_batch)
            #y.append(y_pred)
            #print(y_pred)
            if (y_pred.round() == 1):
                print("Non Defective")
            else:
                print("Defective")

                #requests.get('http://192.168.43.106:5000/response');
                #if (len(y)>1):
                    #for i in range(len(y)):
                        #if (y[i-1]!=y[i]):
                            #pygame.mixer.init()
                            #pygame.mixer.music.load('sound.mp3')
                            #pygame.mixer.music.play()
                #playsound.playsound("sound.mp3")
                #p.play()
                #winsound.PlaySound("sound.mp3", winsound.SND_FILENAME)
            #else:
                #if (len(y)>1):
                    #for i in range(len(y)):
                        #if (y[i-1]!=y[i]):
                            #pygame.mixer.init()
                            #pygame.mixer.music.load('danger.mp3')
                            #xpygame.mixer.music.play()
                #playsound.playsound("danger.mp3")
                #winsound.PlaySound("danger.mp3", winsound.SND_FILENAME)
        except Exception as error:
            print(error)

def main():
    t1 = threading.Thread(target=video_stream)
    t2 = threading.Thread(target=ml)

    t1.start()
    t2.start()

if __name__ == '__main__':
    main()