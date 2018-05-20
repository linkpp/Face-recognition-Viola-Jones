import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
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
    
def predict_camera(input_source):
    
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

    list_faces = []
    list_file = os.listdir(face_folder)
    threshold_confidence = int(list_file[0].split("-")[1]) + range_conf
    for imgfile in list_file:
        confidence = int(imgfile.split('-')[1])
        if confidence > threshold_confidence:
            continue
        img = cv2.imread(face_folder + imgfile)
        # print(width)
        face_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = face_gray.shape[:2]
        x_face = 0 # Position x 
        eyes = eye_cascade.detectMultiScale(face_gray)
        cv2.imshow(imgfile, img)
        cv2.waitKey(100)
        histogram = cv2.calcHist([face_gray],[0],None,[256],[0,256])
        # plt.plot(histogram)
        # plt.show()
        print(histogram.shape)
        entropy = 0
        # print(histogram)
        for i in histogram:
            # print(i)
            if i[0] >0:
                entropy -= i[0] * log(i[0])
                # print("entropyi", entropy)
        print ("entropy: ",entropy)

        print("----")
        mu1, sigma1 = 0, 1
        mu2, sigma2 = 10, 1
        s1 = np.random.normal(mu1, sigma1, 100000)
        s2 = np.random.normal(mu2, sigma2, 100000)

        hist1 = np.histogram(s1, bins=100, range=(-20,20), density=True)
        data1 = hist1[0]
        ent1 = -(data1*np.log(np.abs(data1))).sum() 
        print("ent1:",s1)

        if (isinstance(eyes, tuple) or eyes.shape != (2,4) ):
            continue

        straight_score =  abs(eyes[1][1] -eyes[0][1]) + abs(2*x_face + width - (eyes[0][0] + eyes[1][0] +eyes[1][2]) )
        print(straight_score)



def find_best_face2(face_folder, range_conf):
    
    eye_cascade = cv2.CascadeClassifier("model/haarcascade_eye_tree_eyeglasses.xml")
    # haarcascade_eye_tree_eyeglasses haarcascade_lefteye_2splits haarcascade_eye
    
    list_file = os.listdir(face_folder)
    min_straight_score = 1000
    best_face = list_file[0]
    threshold_confidence = int(list_file[0].split("-")[1]) + range_conf
    best_face_crop = best_face
    for imgfile in list_file:
        confidence = imgfile.split("-")[1]
        confidence = int(confidence)
        if confidence > threshold_confidence:
            continue
        img = cv2.imread(face_folder + imgfile)
        face_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_crop = cv2.resize(face_gray,(100,100))

        eyes = eye_cascade.detectMultiScale(face_crop)
        
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(face_gray,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
        # cv2.imshow("Face process", face_crop)
        
        if (isinstance(eyes, tuple) or eyes.shape != (2,4) ):
            continue
        
        straight_score = 1000
        straight_score = abs(eyes[1][1] -eyes[0][1]) + abs(eyes[0][0] + eyes[1][0] +eyes[1][2] - 100)
        if(straight_score< min_straight_score):
            min_straight_score = straight_score
            best_face = imgfile
            best_face_crop = face_crop
        
    
    print("best face: ", best_face, min_straight_score)
    face = cv2.imread("output/Linh/"+ best_face)
    cv2.imshow("best face", face)
    # cv2.imshow("best face crop", best_face_crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



face_recognizer = cv2.face.LBPHFaceRecognizer_create()  # EigenFaceRecognizer_create() or FisherFaceRecognizer_create()
# train_and_save_model("train/")

if not label_names:
    label_names = ["Nga", "Linh", "Jvermind","Tiffany"]

face_recognizer.read("model/lbph_model.yml")
predict_camera("input/aslongas.mp4") # "input/aslongas.mp4"
# find_best_face("output/Linh/", 20)


# imgfile = cv2.imread("12.jpg")
# predict_img(imgfile, 1)

