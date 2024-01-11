import cv2
import numpy as np
from keras.models import model_from_json
import pafy

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

#Load the model structure from json file
emotion_model = model_from_json(open("model.json", "r").read())

#Load the model weights
emotion_model.load_weights('emotion_model.h5')
print("Loaded model from disk")

cap = cv2.VideoCapture('C:\\Users\\morab\\Youssef_Jedha\\Final\\venv\\Hugh.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    bounding_box = cv2.CascadeClassifier('C:\\Users\\morab\\Youssef_Jedha\\Final\\haarcascades\\haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x,y-50), (x+w, y+h+10), (255,0,0), 2)
        roi_gray_frame = gray_frame[y:y+h, x:x+w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48,48)), -1), 0)
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    cv2.imshow('Video', cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break