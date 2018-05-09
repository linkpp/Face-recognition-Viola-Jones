import cv2
import numpy as np
import os

label_names = []

def face_dectect(img):

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )
    
    return faces

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
            faces = face_dectect(img)
            
            if (isinstance(faces, tuple) ): #Not have face
                continue
            
            for x,y,w,h in faces:
                face_crop = img[y:y+h,x:x+w]
                face_crop = cv2.resize(face_crop, (100,100))
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                listfaces.append(face_crop)
                listlabels.append(label_int)
    
    listlabels = np.array(listlabels) 
    
    face_recognizer = cv2.face.LBPHFaceRecognizer_create() # Can use: EigenFaceRecognizer_create() or FisherFaceRecognizer_create()
    face_recognizer.train(listfaces, listlabels)
    face_recognizer.write("model.yml")
    print("save model complete!")



def predict_img(img):

    faces = face_dectect(img)
    # print("Detect faces:"+ imgfile)

    for (x,y,w,h) in faces:
        face_crop = img[y:y+h,x:x+w]
        face_crop = cv2.resize(face_crop, (100,100))
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        label, confidence = face_recognizer.predict(face_crop)
        text = label_names[label] + " " + str(int(confidence))
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    
    cv2.imshow("img",img)
    # cv2.imwrite("out.jpg",img)
    
def predict_camera():
    
    # fps = int(capture.get(cv2.CAP_PROP_FPS))
    fps = 2
    count = 0
    capture = cv2.VideoCapture(0)
    while(capture.isOpened()):
        ret, frame = capture.read()
        if(count%fps ==0):
            predict_img(frame)
        count+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()


train_and_save_model("train/")

if not label_names:
    label_names = ["Nga", "Linh", "Hang", "Tiffany"]
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("model.yml")
predict_camera()

# predict_img(imgfile)
# imgfile = cv2.imread("linh.jpg")

