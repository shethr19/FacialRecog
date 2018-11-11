import os
import cv2
import pickle
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
#eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eyexml')

recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id =0
label_ids={}
y_labels=[]
x_train =[]


for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
			path=os.path.join(root, file)
			label = os.path.basename(root).replace(" ", "-").lower() ## os.path.dirname(path) can replace 'root'. same thing
			print(label,path)
			if not label in label_ids:
				label_ids[label]=current_id
				current_id += 1
			id_ = label_ids[label]
			print(label_ids)
			#labels.append(label) #some number for our labels
			#x_train.append(path) #verify this image, turn into a NUMPY array, covert it to gray
			pil_image = Image.open(path).convert("L") #python image library #grayscale
			size= (550,550)
			final_image=pil_image.resize(size, Image.ANTIALIAS)
			image_array = np.array(final_image, "uint8")#image into numbers then stored into arrays
			#print(image_array)

			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+h]
				x_train.append(roi) 
				y_labels.append(id_)

#print(y_labels)
#print(x_train)

with open("labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")