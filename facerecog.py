import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import statistics
from math import log

label_names = []
def face_dectect(img):

    face_cascade = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    list_face_gray = []

    face_positions = face_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for(x,y,w,h) in face_positions:
        face_gray = img_gray[y:y+h,x:x+w]
        list_face_gray.append(face_gray)

    return face_positions , list_face_gray

def train_and_save_model(input_folder):
    listfaces = []
    listlabels = []

    dirs = os.listdir(input_folder)
    for subdir in dirs:
        label = subdir.split('-')
        label_int = int(label[0]) #label must be int
        # label = int(subdir[:1]) # label mustbe int
        label_names.append(label[1])

        sub_path = input_folder + subdir + "/"
        for imgfile in os.listdir(sub_path):
            img = cv2.imread(sub_path + imgfile)
            print("process: "+ sub_path + imgfile)

            #Detect face:
            face_positions, list_face_gray = face_dectect(img)
            
            if (isinstance(face_positions, tuple) ): #Not have face
                continue
            
            for face_gray in list_face_gray:
                face_gray = cv2.resize(face_gray, (64,64))
                listfaces.append(face_gray)
                listlabels.append(label_int)
    
    listlabels = np.array(listlabels) 
    
    # face_recognizer = cv2.face.LBPHFaceRecognizer_create() # EigenFaceRecognizer_create() or FisherFaceRecognizer_create()
    face_recognizer.train(listfaces, listlabels)
    face_recognizer.write("model/lbph_model.yml")
    print("save model complete!")



def predict_img(img, count):
    
    eye_cascade = cv2.CascadeClassifier("model/haarcascade_eye_tree_eyeglasses.xml") 
    # haarcascade_eye_tree_eyeglasses haarcascade_lefteye_2splits haarcascade_eye

    face_positions, list_face_gray = face_dectect(img)

    for (x,y,w,h) , face_gray in zip(face_positions, list_face_gray):
        face_color = img[y:y+h, x:x+w]
        face_crop = cv2.resize(face_gray, (64,64))
        label, confidence = face_recognizer.predict(face_crop)
        
        if(confidence<170):
            text = label_names[label] + " " + str(int(confidence))
            
            if not os.path.exists("output/"+label_names[label] +"/"):
                os.makedirs("output/"+label_names[label] +"/")
            file_name = "output/"+label_names[label] +"/"+label_names[label] +"-" + str(int(confidence)) +"-" + str(count)+ ".jpg"
            cv2.imwrite(file_name,face_color)
        else:
            text = "Unknown " + str(int(confidence))
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
    
    cv2.imshow("img",img)
    cv2.imwrite("face.jpg",img)

    # return min_confidence
    
def predict_video(input_source):
    
    # fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_per_cap = 2
    count = 0
    capture = cv2.VideoCapture(input_source) 
    while(capture.isOpened()):
        ret, frame = capture.read()
        if(count%frame_per_cap ==0):
            predict_img(frame, count)

        count+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()


def find_best_face(face_folder, range_conf):

    eye_cascade = cv2.CascadeClassifier("model/haarcascade_eye_tree_eyeglasses.xml")
    
    vector_faces = []
    label_faces  = []
    list_confidence = []
    list_straight = []
    list_stdev = []
    list_entropy = []
    
    list_file = os.listdir(face_folder)
    threshold_confidence = int(list_file[0].split("-")[1]) + range_conf
    for imgfile in list_file:
    
        confidence = int(imgfile.split('-')[1])
        if confidence > threshold_confidence:
            continue

        img = cv2.imread(face_folder + imgfile)
        face_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        height, width = face_gray.shape[:2]
        x_face = 0 # Position x 
        eyes = eye_cascade.detectMultiScale(face_gray)
        if (isinstance(eyes, tuple) or eyes.shape != (2,4) ):
            continue

        straight_score = ( abs(eyes[1][1] -eyes[0][1]) + abs(2*x_face + width - (eyes[0][0] + eyes[1][0] +eyes[1][2]) ) )/width
        

        histogram = cv2.calcHist([face_gray],[0],None,[256],[0,256])
        num_pixel = height* width
        hist_freq = [i[0]/num_pixel for i in histogram]
        stdev = statistics.stdev(hist_freq)
        

        entropy = 0
        for i in hist_freq:
                if i >0:
                    entropy -= i * log(i,2)

        print("File:", imgfile)
        print("Confidence:", confidence)
        print("Straight score:",straight_score)
        print("StandardError:", stdev)
        print ("entropy: ", entropy)
        print("-----")
        list_confidence.append(confidence)
        list_straight.append(straight_score)
        list_stdev.append(stdev)
        list_entropy.append(entropy)
        
        label_faces.append(imgfile)
    

    # Confidence --> Acuracy: Min
    # straight_score --> Straight Face: Min
    # stdev --> Contrast : Min
    # entropy --> Bluring : Max 
    
    confi_norm = normalize_data(list_confidence, 0)
    strai_norm = normalize_data(list_straight, 0)
    stdev_norm = normalize_data(list_stdev, 0)
    entro_norm = normalize_data(list_entropy, 1)

    for i in range(0, len(label_faces)):
        vector_faces.append([confi_norm[i], strai_norm[i], stdev_norm[i], entro_norm[i]])
    
    for i, element in enumerate(vector_faces):
        print(label_faces[i])
        print(element)
    norm = np.linalg.norm(vector_faces, axis=1)
    print("Norm:", norm)
    index_min = np.argmax(norm)
    print("best_face: ", label_faces[index_min], min(norm))


def normalize_data(data, type): # Type = 0: normalize normal (for entropy), Type = 1: Normalize revert
    min_data = min(data)
    max_data = max(data)

    if type ==0:
        for i, element in enumerate(data):
            data[i] = (element - min_data)/ (max_data - min_data)
    else:
        for i, element in enumerate(data):
            data[i] = (max_data - element)/ (max_data - min_data)
    return data


face_recognizer = cv2.face.LBPHFaceRecognizer_create()  # EigenFaceRecognizer_create() or FisherFaceRecognizer_create()
# train_and_save_model("train/")

if not label_names:
    label_names = ["Nga", "Linh", "Jvermind","Tiffany"]

face_recognizer.read("model/lbph_model.yml")
# predict_video("input/aslongas.mp4") # "input/aslongas.mp4"
find_best_face("output/Tiffany/", 10)


