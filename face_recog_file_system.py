import pyttsx3
import cv2
import os
import numpy as np
from PIL import Image
import pickle
import webbrowser

def boli(name):
        engine = pyttsx3.init() #text to speech initiation
        engine.say("Hey "+name+"i am helping you in finding your file")
        engine.setProperty('rate',150)  #120 words per minute
        engine.setProperty('volume',0.9) 
        engine.runAndWait()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")
#print(image_dir)

eye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_smile.xml')
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []


for root, dirs, files in os.walk(image_dir):
	for file in files:
                
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			#print(path)
			label = os.path.basename(root)
			#print(label)
			#print(label, path)
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]
			#print(id_)
			#print(label_ids)
			#y_labels.append(label) # some number
			#x_train.append(path) # verify this image, turn into a NUMPY arrray, GRAY
			pil_image = Image.open(path).convert("L") # grayscale
			size = (550, 550)
			final_image = pil_image.resize(size, Image.ANTIALIAS)
			image_array = np.array(final_image, "uint8")
			#print(image_array)
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.30, minNeighbors=4)

			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)


#print(y_labels)
#print(x_train)

with open("face-labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("recognizors/face-trainner.yml")


import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('D:\Personal\Aishwarya\haarcascade/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('D:\Personal\Aishwarya\haarcascade/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('D:\Personal\Aishwarya\haarcascade/haarcascade_smile.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizors/face-trainner.yml")

labels = {"person_name": 1}
with open("face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)
i=0
list=[]
while(True):
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.30, minNeighbors=4)
    for (x, y, w, h) in faces:
    	print(x,y,w,h)
    	roi_gray = gray[y:y+h, x:x+w] 
    	roi_color = frame[y:y+h, x:x+w]

    	# recognize? deep learned model predict keras tensorflow pytorch scikit learn
    	id_, conf = recognizer.predict(roi_gray)
    	#print(conf)
    	if conf>=4 and conf <= 150:
    		#print(id_)
    		print(labels[id_])
    		font = cv2.FONT_HERSHEY_SIMPLEX
    		name=labels[id_]
    		print(id_)
    		list.append(id_)
    		
    		print(list)
    		color = (255, 255, 255)
    		stroke = 2
    		
                
    		cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

                      
    	img_item = "7.png"
    	cv2.imwrite("images/"+img_item, roi_color)

    	color = (255, 0, 0) #BGR 0-255 
        
    	end_cord_x = x + w
    	end_cord_y = y + h
    	cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, 2)
    	
    name=labels[id_]
    
    if(name=="images"):
                        print("Unknown Face Detection")
                        continue
    if(len(list)>=20):
            break
    if(i>=1):
         try:
                list[i]==list[i-1]
         except IndexError:
                 continue
                 
         if(list[i]==list[i-1]):
                    if cv2.waitKey(50):
                            path=("D://Personal//Aishwarya//images//")
                            webbrowser.open(path+name)
                            boli(name)
                            break
            
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    i+=1


cap.release()
cv2.destroyAllWindows()


