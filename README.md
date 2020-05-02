# Auto_Photo_Capture
An AI powered program which automatically captures photos based on an expression selected by user. 

## Overview
As the name implies, this is a program which automatically captures expression in real time and clicks just the right shots! 

## Required Libraries/Tools 
|Library | version|
|--------|---------|
|Python         |3.5|  
|OpenCV         |3.4.0.12|
|Numpy          |1.14.1|
|Pandas         |0.22.0|
|Tensorflow    | 1.5.0|
|Pyglet       |  1.3.1|
|Scipy        |  1.0.0|
|Scikit Learn|   0.17.1|
|Keras       |   2.1.3|

 
## How to run
Install the necessary libraries.  
Extract training.csv out of the training.zip folder into the root folder.  
Open the prog.py file in a python IDE and hit run. 

## Controls
Click on the expression you want to capture. The program will keep taking shots as long as the program is running and save it in the same folder as your program. Hit q to exit.  

## Specification

This project intensively uses image processing techniques and computer vision.I have used the BioID face database which contains 1521 gray level images with a resolution of 384×286 pixel as our dataset. The images also consist of 20 manually marked facial features. The OpenCV Haar cascade classifier- based on the face detection method proposed by viola and Jones- is used for face detection.I trained a convolutional neural network to detect special facial features using marked images as ground truth. Different techniques such as optical flow are used to detect expressions such as smile, pout etc. in real time. Once a feature say smile has been detected, the camera will click the shot and save it.  

## Related Areas
Face detection is pretty common these days and you will find this in almost every camera in the world these days! Many apps like snapchat use face detection to apply different features and focus more on the faces while capturing photos. Facebook uses face detection and face recognition to propose tags. Recently many cameras including Sony DSC T300 have introduced a feature of smile detection in which the camera automatically detects smile and clicks on just the right moment. We intend to use these techniques to come up with a similar program but in addition to smiles, we intend to detect more expressions such as pout face, laughing face, serious face etc.   

## Sources and References:  
1.	http://coding-robin.de/2013/07/22/train-your-own-opencv-haar-classifier.html
2.	https://www.csie.ntu.edu.tw/~fuh/personal/FaceDetectionandSmileDetection.pdf
3.	https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html
4.	https://www.bioid.com/facedb/








