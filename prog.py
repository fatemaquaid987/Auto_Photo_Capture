import cv2
import sys
import datetime
import argparse
import pandas as pd
import numpy as np
import os

import pyglet
from pyglet.window import Window, mouse, gl, key
from pyglet.gl import *

from scipy.spatial import distance

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import BatchNormalization, Conv2D, Activation, MaxPooling2D, Dense, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten
from keras import optimizers


platform = pyglet.window.get_platform()
display = platform.get_default_display()
screen = display.get_default_screen()
 
myscreen = pyglet.window.Window(760, 707,                   # setting window
              resizable = False,  
              caption="AI Shot Capture",  
              config=pyglet.gl.Config(double_buffer=True),  #Avoids flickers 
              vsync=False                                   #For flicker-free animation 
              )                                             
myscreen.set_location(screen.width // 2 - 300,screen.height//2 - 350)
 


bgimage= pyglet.resource.image('screen.png')               #loading screen

''' load json and create model'''
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

''' load weights into new model'''
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

'''converts string to image[96x96]'''
def string2image(string):
    """Converts a string to a numpy array."""
    return np.array([int(item) for item in string.split()]).reshape((96, 96))

'''read file'''
df = pd.read_csv('training.csv')

'''dropping incomplete records'''
fully_annotated = df.dropna()

''' training images'''
X = np.stack([string2image(string) for string in fully_annotated['Image']]).astype(np.float)[:, :, :, np.newaxis]

''' marked points'''
y = np.vstack(fully_annotated[fully_annotated.columns[:-1]].values)
#X.shape, X.dtype
#y.shape, y.dtype
X_train = X / 255.

'''scaling'''
scaler = MinMaxScaler(feature_range=(-1, 1))

'''fitting keypoints points '''
y_train = scaler.fit_transform(y)

'''captures image'''
def takeSnapshot(frame):
    # grab the current timestamp and use it to construct the
    # output path
    s, img = frame.read()
    shot_idx = 0
    ts = datetime.datetime.now()
    filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
    outputPath = os.getcwd()
    p = os.path.sep.join((outputPath, filename))
    # save the file
    if (s):
        cv2.imwrite(p, img)
        print("[INFO] saved {}".format(filename))

        
def run(expression):
    global loaded_model , output_pipe
    haarcascade_Path = 'haarcascade_frontalface_default.xml' 
    faceCascade = cv2.CascadeClassifier(haarcascade_Path) #loading haar cascade classifier'''


    webcam = cv2.VideoCapture(0)
    while True:
        ret, frame = webcam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converting to grayscale'''
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) #drawing rectangle on face'''
            crop = frame[y:h+y, x:w+x] #cropping the rect'''
            height, width = crop.shape[:2]
            cropped = cv2.resize(crop,(96, 96), interpolation = cv2.INTER_CUBIC)#resizing the cropped face image to 96x96 dimensions'''
            gray2 = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY);
            gray2 = cv2.normalize(gray2.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) #converting image type to float'''
            
            
            img = gray2.reshape(1,96,96,1)           
            predictions = loaded_model.predict(img) #sending image to the classifier'''
            xy_predictions = scaler.inverse_transform(predictions).reshape(15, 2)#getting the points in desired format'''
            
            
            for i in range(0,15):
                cv2.circle(cropped,(xy_predictions[i, 0],xy_predictions[i, 1]),1,(255,255,255), -11) #drawing points on image'''
            
            lc= (xy_predictions[11, 0],xy_predictions[11, 1])
            rc= (xy_predictions[12, 0],xy_predictions[12, 1])
            up = (xy_predictions[13, 0],xy_predictions[13, 1])
            down = (xy_predictions[14, 0],xy_predictions[14, 1])
            dst1 = distance.euclidean(lc,rc)
            dst2 = distance.euclidean(up,down)
            font = cv2.FONT_HERSHEY_SIMPLEX
            print("d1", dst1)
            print("d2", dst2)
            '''detecting exprressions'''
            '''1 = smile, 2 = poker, 3 = amazed'''
            if(expression == 1):
                if(dst1 >= 29.5) and (dst2 > 7):
                    takeSnapshot(webcam)
                    
                    cv2.putText(cropped, "smiling", (10, 50), font, 0.2, (0, 255, 0), 1, cv2.LINE_AA)
                
            elif expression == 2:
                if (dst2 <= 10) and (dst1 >= 29):
                    cv2.putText(cropped, "poker", (10, 50), font, 0.2, (0, 255, 0), 1, cv2.LINE_AA)
                    takeSnapshot(webcam)
            else:
                if (dst2 >=11) and (dst1 <= 26):
                    cv2.putText(cropped, "amazed", (10, 50), font, 0.2, (0, 255, 0), 1, cv2.LINE_AA)
                    takeSnapshot(webcam)
                
            cv2.imshow('crop',cropped) 
        cv2.imshow('Video', frame)
        if cv2.waitKey(1)  & 0xFF == ord('q'):
            break
        
    ''' When everything is done, release the capture'''
    webcam.release()
    cv2.destroyAllWindows()

@myscreen.event

def on_draw():
    myscreen.clear()
    bgimage.blit(0,0)

@myscreen.event
def on_mouse_release(x, y, button, modifiers):
    if (36 <= x <= 201) and (157 <= y <= 326):
        run(1)
    elif (300 <= x <= 466) and (157 <= y <= 326):
        run(2)
    
    elif (566 <= x <= 723) and (157 <= y <= 326):
        run(3)  
        

def update(dt):
    dt

pyglet.clock.schedule_interval(update, 1/20.)
pyglet.app.run()
