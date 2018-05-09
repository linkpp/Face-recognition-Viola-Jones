import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
capture = cv2.VideoCapture(0)
# capture = cv2.VideoCapture("aslongas.mp4")

count = 0
num_face = 0
# fps = int(capture.get(cv2.CAP_PROP_FPS))
fps = 1
print(fps)

while(capture.isOpened()):
	ret, frame = capture.read()
	if(count%fps==0):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(100, 100)
		)
		# print(faces)
		for (x,y,w,h) in faces:
			num_face += 1
			face_crop = frame[y:y+h,x:x+w]
			face_save = "img/face" + str(num_face) + ".jpg"
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
			
	cv2.imwrite('img/frame'+str(count) + '.jpg', frame)
	count +=1
	cv2.imshow("frame", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
capture.release()
cv2.destroyAllWindows()
