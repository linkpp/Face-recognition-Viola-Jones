import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# capture = cv2.VideoCapture(0)
capture = cv2.VideoCapture("aslongas.mp4")
count = 0

while(capture.isOpened()):
	ret, frame = capture.read()
	print(capture.get(5))
	count+=1
	if(True):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(100, 100)
		)
		
		for (x,y,w,h) in faces:
			face_crop = frame[y:y+h,x:x+w]
			face_save = "img/face" + str(count) + ".jpg"
			cv2.imwrite(face_save, face_crop)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = frame[y:y+h, x:x+w]
	cv2.imshow("frame", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
capture.release()
cv2.destroyAllWindows()
