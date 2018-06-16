from skimage.filters.rank import entropy
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from math import log
import statistics

def caculate_entropy(face_folder):
    eye_cascade = cv2.CascadeClassifier("model/haarcascade_eye_tree_eyeglasses.xml")

    list_faces = []
    list_file = os.listdir(face_folder)
    # imgfile = list_file[4]
    for imgfile in list_file:
        # if imgfile != "lowcontrast.png":
        #     continue
        img = cv2.imread(face_folder + imgfile)
        face_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # histogram = cv2.calcHist([face_gray],[0],None,[256],[0,256])
        histogram = cv2.calcHist([face_gray],[0],None,[256],[0,256])
        # plt.plot(histogram)
        # plt.show()
        height, width = face_gray.shape[:2]
        num_pixel = height* width
        
        hist_freq = [i[0]/num_pixel for i in histogram]
        # print("His_fre:",hist_freq)
        stderr = statistics.stdev(hist_freq)
        print("stderr:" ,stderr)
        plt.title(imgfile)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.plot(hist_freq)
        plt.savefig("graph/" + imgfile)
        plt.clf() 
        entropy = 0
        for i in hist_freq:
                # print(i)
                if i >0:
                    entropy -= i * log(i,2)
                    # print("entropyi", entropy)
        print ("entropy: ", imgfile, entropy)
        print("-----")
    

# caculate_entropy("test1/")

def face_detect(imgfile):
    
    face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
    # eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
    img = cv2.imread(imgfile)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=10,
        minSize=(200, 200)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('img',img)
    cv2.imwrite("anhdetected.jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

face_detect("final.jpg")

