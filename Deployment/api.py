import uvicorn
import pandas as pd 
from fastapi import FastAPI , UploadFile
import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json
import numpy as np
import io
from PIL import Image #C'est une bibliothèque de traitement d'images pour ouvrir, manipuler et sauvegarder de nombreux formats de fichier image différents.
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import base64
import os

app = FastAPI ()

@app.get("/")
async def greet () :
    return {"message": "bonjour"}

#fonction pour chargement du model
#def load():
#    model_path = "youssef.json"
#    model = load_model(model_path, compile=False)
#    return model

# Load the model structure from json file
emotion_model = model_from_json(open("youssef.json", "r").read())

# Load the model weights
emotion_model.load_weights('youssef.h5')

# Chargement du model
#emotion_model = load()

# Dictionnaire des émotions
emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad"}


# Fonction pour traitement d'image
def preprocess(image_stream):

    # Conversion of a PIL image to an OpenCV image (numpy array)
    frame = np.array(image_stream)

    ## Define the maximum size
    #max_size = 800
#
    ## Get the aspect ratio of the image
    #aspect_ratio = frame.shape[1] / frame.shape[0]
#
    #if frame.shape[1] > frame.shape[0]:
    #    # If width is greater than height
    #    new_width = max_size
    #    new_height = int(new_width / aspect_ratio)
    #else:
    #    # If height is greater than width
    #    new_height = max_size
    #    new_width = int(new_height * aspect_ratio)

    # Resize the image
    #frame = cv2.resize(frame, (new_width, new_height))

    # Get the shape of the frame
    height, width, channels = frame.shape

    # Conversion du format RGB au format BGR (OpenCV utilise BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_copy = frame.copy()
    
    # Détection des visages
    bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=3)
    
    # Initialisation des listes
    face_list = []
    pred_list = []
    full_pred_list = []
    # Initialize a list to store the coordinates of the ROIs
    roi_coordinates = []
    
    # Pour chaque visage détecté
    for (x, y, w, h) in num_faces:
        
        # Récupérer le visage en couleur
        #roi_color_frame = frame[y:y+h, x:x+w]
        
        # Convert the ROI to grayscale
        roi_gray_frame = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        
        # Dessiner un rectangle autour du visage
        cv2.rectangle(frame, (x,y), (x+w, y+h+10), (255,255,0), 2)
        
        # Change le format de l'image en 48x48 pixels
        #roi_resized = cv2.resize(roi_color_frame, (48,48))
        
        # Convertir l'image en RGB
        #roi_resized_1 = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48,48)), -1), 0)
        # Addition de la dimension du batch
        #cropped_img = np.expand_dims(roi_resized_1, 0)
        
        # Prédiction de l'émotion
        emotion_prediction = emotion_model.predict(cropped_img)
        
        # Récupérer l'émotion avec la plus grande probabilité
        maxindex = int(np.argmax(emotion_prediction))
        
        # Afficher l'émotion prédite
        cv2.putText(frame, emotion_dict[maxindex], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

        # Store the coordinates of the ROI
        roi_coordinates.append((x, y, x + w, y + h))
        
        # Ajouter le visage et l'émotion prédite aux listes
        
        pred_list.append(maxindex)
        full_pred_list.append(emotion_prediction)
    
    for (x, y, x1, y1) in roi_coordinates:
        # Crop the ROI from the frame
        face_list.append(frame_copy[y:y1, x:x1])
    
    full_img = cv2.resize(frame,(width, height),interpolation = cv2.INTER_CUBIC)
    full_img = np.expand_dims(full_img, 0)  # Adds the batch and channel dimensions
    
    return face_list, pred_list, full_img, full_pred_list

@app.post("/predect")
async def predect(file:UploadFile): # async: executer plusieurs images en meme temps
    # Lecture de l'image
    image_data = await file.read()
    image_stream = Image.open(io.BytesIO(image_data))
    image_stream.seek(0)

    # Prétraitement de l'image avec la fonction preprocess()
    img_processed = preprocess(image_stream)
    
    # Récupérer les prédictions
    predictions = img_processed[1]
    
    img_base64_list = []
    
    # Convertir les images en base64 pour les envoyer au client
    for i in range(len(img_processed[0])):
        _, img_encoded = cv2.imencode('.jpg', img_processed[0][i])
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
        img_base64_list.append(img_base64)
    
    # Convertir les images en base64 pour les envoyer au client
    _, full_img_encoded = cv2.imencode('.jpg', img_processed[2][0])
    full_img_base64 = base64.b64encode(full_img_encoded.tobytes()).decode('utf-8')
    
    pred = [None] * len(predictions)
    # Convertir les prédictions en émotions
    for i in range(len(predictions)):
        pred[i] = emotion_dict[predictions[i]]
    
    full_pred = img_processed[3]

    return {'prediction':pred, 'image': img_base64_list, 'full_image': full_img_base64, 'full_prediction': [prediction.tolist() for prediction in full_pred]}

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)