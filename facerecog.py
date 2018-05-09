import cv2
import numpy as np
import os

def face_dectect(img):

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )
    
    # for (x,y,w,h) in faces:
        # face_crop = img[y:y+h,x:x+w]
        # cv2.imwrite(face_save, face_crop)
        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    
    return faces

def process_train(input_folder):
    listfaces = []
    listlabels = []

    dirs = os.listdir(input_folder)
    for subdir in dirs:
        label = int(subdir[:1]) # label mustbe int

        sub_path = input_folder + subdir + "/"
        for imgfile in os.listdir(sub_path):
            img = cv2.imread(sub_path + imgfile)
            print("process: "+ sub_path + imgfile)

            #Detect face:
            faces = face_dectect(img)
            # cv2.imgshow(img_rectangle)
            if (isinstance(faces, tuple) ):
                continue
            
            for x,y,w,h in faces:
                face_crop = img[y:y+h,x:x+w]
                face_crop = cv2.resize(face_crop, (100,100))
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                listfaces.append(face_crop)
                listlabels.append(label)
    
    listlabels = np.array(listlabels) #opencv face_recognizer.train must be numpy array
    # print(listlabels)

    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return listfaces, listlabels



def predict_img(img):

    label_name = ["","Nga", "Linh"]
    
    # img = cv2.imread(imgfile)
    # print("process:"+ img)
    faces = face_dectect(img)
    # print("Detect faces:"+ imgfile)

    for (x,y,w,h) in faces:
        face_crop = img[y:y+h,x:x+w]
        face_crop = cv2.resize(face_crop, (100,100))
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        label, confidence = face_recognizer.predict(face_crop)
        print("Detected: "+ label_name[label])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img, label_name[label], (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    
    cv2.imshow("img",img)
    # cv2.imwrite("out.jpg",img)
    
def predict_camera():
    
    # fps = int(capture.get(cv2.CAP_PROP_FPS))
    # fps = 1
    count =0
    capture = cv2.VideoCapture(0)
    while(capture.isOpened()):
        ret, frame = capture.read()
        predict_img(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()


def train_and_save_model():
    faces, labels = process_train("train/")
    face_recognizer.train(faces, labels)
    face_recognizer.write("train.yml")


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# face_recognizer = cv2.face.EigenFaceRecognizer_create()
# face_recognizer = cv2.face.FisherFaceRecognizer_create()

# train_and_save_model()


face_recognizer.read("train.yml")
predict_camera()

# imgfile = cv2.imread("linh.jpg")
# predict_img(imgfile)

