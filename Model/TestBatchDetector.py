import cv2
import numpy as np
from keras.models import model_from_json
import pafy
import os

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

#Load the model structure from json file
emotion_model = model_from_json(open("model.json", "r").read())

#Load the model weights
emotion_model.load_weights('emotion_model.h5')
print("Loaded model from disk")

# Load the cascade classifier
bounding_box = cv2.CascadeClassifier('content\\Final\\haarcascades\\haarcascade_frontalface_default.xml')

# Directory containing images
dir_path = 'content\\Final\\Model\\batch'

# Output directory
output_dir = 'content\\Final\\Model\\result'
os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

# Initialize an empty list to store the ROI images
roi_images = []

# Loop over each file in the directory
for filename in os.listdir(dir_path):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".webp"):
        # Load the image
        frame = cv2.imread(os.path.join(dir_path, filename))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=3)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x,y-10), (x+w, y+h+10), (255,0,0), 2)
            roi_gray_frame = gray_frame[y:y+h, x:x+w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48,48)), -1), 0)
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x-20, y+70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, str(np.max(emotion_prediction)), (x-20, y+90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

            # Resize the ROI to a fixed size (e.g., 100x100) and append it to the list
            roi_resized = cv2.resize(roi_gray_frame, (100, 100))
            # Convert the grayscale image to BGR
            roi_resized_color = cv2.cvtColor(roi_resized, cv2.COLOR_GRAY2BGR)

            # Add the prediction text to the ROI image
            cv2.putText(roi_resized_color, emotion_dict[maxindex], (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
            cv2.putText(roi_resized_color, str(np.max(emotion_prediction)), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)

            # Append the ROI image to the list
            roi_images.append(roi_resized_color)
            
            # Save the ROI image in the output directory
            cv2.imwrite(os.path.join(output_dir, f'output_{filename}_{maxindex}.png'), roi_gray_frame)

# Convert the list of images to a single numpy array
roi_images = np.array(roi_images)

# Calculate the number of tiles in x and y directions
nx, ny = int(np.ceil(np.sqrt(len(roi_images)))), int(np.ceil(np.sqrt(len(roi_images))))

# Create a black image with the size of the mosaic
mosaic = np.zeros((100*ny, 100*nx, 3))

# Fill the mosaic with the ROI images
for i in range(ny):
    for j in range(nx):
        if i*nx+j < len(roi_images):
            mosaic[i*100:(i+1)*100, j*100:(j+1)*100] = roi_images[i*nx+j]

# Display the mosaic image
cv2.imshow('Mosaic', mosaic)
cv2.waitKey(0)  # Wait for the user to close the window

# Save the mosaic image in the output directory
cv2.imwrite(os.path.join(output_dir, 'mosaic.png'), mosaic)


cv2.destroyAllWindows()  # Close the window