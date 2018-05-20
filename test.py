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
    

caculate_entropy("output/test/")

    