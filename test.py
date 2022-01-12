import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load existing model
model = load_model('facial-expression-model.h5')

# Label
CLASSES = ['angry', 'happy', 'neutral', 'sad', 'surprise']

# Start video capture
cap = cv2.VideoCapture(0)

# Find out the path to cascade classifier XML file
cascPath = os.path.dirname(cv2.__file__) + '/data/haarcascade_frontalface_default.xml'
haarcascade = cv2.CascadeClassifier(cascPath)

while True:
    rval, im = cap.read()
    im = cv2.flip(im, 1, 1)

    # Convert RGB to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray_three = cv2.merge([gray, gray, gray])

    # Detect faces
    faces = haarcascade.detectMultiScale(gray, 
                                         scaleFactor=1.05, 
                                         minNeighbors=5, 
                                         minSize=(40,40), 
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    
    for face in faces:
        # The coordinates of the face
        x, y, w, h = face
        face_im = gray_three[y:y+h, x:x+w]
        
        # Resize frame to 128x128 and normalize
        resized = cv2.resize(face_im, (128,128))
        normalized = resized / 255.0
        
        # Expand dimensions from (128,128,3) to (1,128,128,3)
        reshaped = np.expand_dims(normalized, axis=0)
        
        # Predict the facial expression
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]

        # Draw bounding box
        cv2.rectangle(im, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.rectangle(im, (x,y-40), (x+w,y), (255,0,0), -1)

        # Put label name on bounding box
        cv2.putText(im, CLASSES[label], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    
    cv2.imshow('Facial Expression Recognition', im)
    key = cv2.waitKey(10)
    
    if key == 27: 
        break

cap.release()
cv2.destroyAllWindows()