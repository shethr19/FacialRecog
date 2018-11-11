import numpy as np
import cv2
import pickle


face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read("trainner.yml")

labels={"person_name":1}
with open("labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while (True):
	ret, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	face = face_cascade.detectMultiScale(gray, scaleFactor =1.5, minNeighbors=5)
	for (x, y, w, h) in face: #face is your project folder
		#print(x,y,w,h) 

		#region of interest 
		#instead of 'gray' --> 'frame' work too
		roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end) same concept for x
		roi_color = frame[y:y+h, x:x+h]
		#Recognize.... deep learned model predict keras, tensorflow, pytorch, scikit learn

		id_, conf = recognizer.predict(roi_gray)
		if conf>= 45 and conf <=85:
			#print(id_)
			#print(labels[id_])

			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255,255,255)
			stroke = 2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

		img_item = "7.png"
		cv2.imwrite(img_item, roi_color)

		#Now draw the rectangle
		color=(0,0,255)
		coloreye=(0,255,0)
		colorsmile=(255,0,0)
		stroke = 2 #line tickness
		end_cord_x = x+w
		end_cord_y = y+h
		#parenthesis consist of frame, starting coordinates, ending coordinates
		#color and strokes --> draws the rectangles
		cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)
		eyes = eye_cascade.detectMultiScale(roi_gray)
		smiles = smile_cascade.detectMultiScale(roi_gray)
		#for (ex, ey, ew, eh) in eyes:
		#	cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), coloreye, stroke)
		for (sx, sy, sw, sh) in smiles:
			cv2.rectangle(roi_color, (sx,sy), (sx+sw, sy+sh), colorsmile, stroke)


	#Displays the resulting frame
	cv2.imshow('frame', frame) #imgshow
	
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break


#When everything done, release the capture
cap.release()
cv2.destroyAllWindows()