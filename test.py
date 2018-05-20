from skimage.filters.rank import entropy
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from math import log

def caculate_entropy(face_folder):
    eye_cascade = cv2.CascadeClassifier("model/haarcascade_eye_tree_eyeglasses.xml")

    list_faces = []
    list_file = os.listdir(face_folder)
    # imgfile = list_file[4]
    for imgfile in list_file:
        img = cv2.imread(face_folder + imgfile)
        face_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist([face_gray],[0],None,[256],[0,256])
        # plt.plot(histogram)
        # plt.show()
        height, width = face_gray.shape[:2]
        num_pixel = height* width
        hist_freq = [i/num_pixel for i in histogram]
        # plt.plot(hist_freq)
        # plt.show()
        entropy = 0
        for i in hist_freq:
                # print(i)
                if i[0] >0:
                    entropy -= i[0] * log(i[0])
                    # print("entropyi", entropy)
        print ("entropy: ", imgfile, entropy)
    

caculate_entropy("output/Tiffany/")

    