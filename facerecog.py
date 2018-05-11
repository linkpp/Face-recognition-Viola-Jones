import cv2
import numpy as np
import os

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
    
    eye_cascade = cv2.CascadeClassifier("model/haarcascade_lefteye_2splits.xml") # haarcascade_eye_tree_eyeglasses haarcascade_lefteye_2splits
    face_positions, list_face_gray = face_dectect(img)
    
    for (x,y,w,h) , face_gray in zip(face_positions, list_face_gray):
        face_color = img[y:y+h, x:x+w]
        face_crop = cv2.resize(face_gray, (64,64))
        label, confidence = face_recognizer.predict(face_crop)
        print("face: ", x,y,w,h)
        eyes = eye_cascade.detectMultiScale(face_gray)

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(face_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
            print("eyes: ",ex, ey, ew, eh)
        
        
        if(confidence<170):
            text = label_names[label] + " " + str(int(confidence))
            if(count%6==0):
                if not os.path.exists("output/"+label_names[label] +"/"):
                    os.makedirs("output/"+label_names[label] +"/")
                cv2.imwrite("output/"+label_names[label] +"/"+label_names[label]+str(int(confidence)) + ".jpg",face_color)
        else:
            text = "Unknown " + str(int(confidence))
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
    
    cv2.imshow("img",img)
    
def predict_camera():
    
    # fps = int(capture.get(cv2.CAP_PROP_FPS))
    fps = 2
    count = 0
    capture = cv2.VideoCapture(0) # "input/aslongas.mp4"
    while(capture.isOpened()):
        ret, frame = capture.read()
        if(count%fps ==0):
            predict_img(frame, count)
        count+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()


face_recognizer = cv2.face.LBPHFaceRecognizer_create()  # EigenFaceRecognizer_create() or FisherFaceRecognizer_create()
# train_and_save_model("train/")

if not label_names:
    label_names = ["Nga", "Linh", "JVmind","Tiffany"]

face_recognizer.read("model/lbph_model.yml")
predict_camera()

# predict_img(imgfile)
# imgfile = cv2.imread("linh.jpg")

