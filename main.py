# Import libraries
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import pygame
import threading

# Initialize detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Alarm function
def alarm(msg):
    pygame.mixer.music.play()

# EAR function    
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    ear = (A + B) / (2.0 * dist.euclidean(eye[0], eye[3]))
    return ear

# MAR function
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    mar = (A + B + C) / 3.0
    return mar

# Initialize camera and pygame
cap = cv2.VideoCapture(0)
pygame.mixer.init()
pygame.mixer.music.load("alert.wav") 

# Get indexes of facial landmarks
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Thresholds
EAR_THRESH = 0.3
EAR_CONSEC_FRAMES = 40  
MAR_THRESH = 0.7
MOUTH_CONSEC_FRAMES = 40

ear_counter = 0 
mar_counter = 0

while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          
    # Detect faces
    faces = detector(gray)
    
    # No faces detected
    if len(faces) == 0:
        continue
            
    # Detect landmarks 
    for face in faces:
        
        # Detect facial landmarks
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        
        # Extract left and right eye coordinates
        leftEye = landmarks[lstart:lend]
        rightEye = landmarks[rstart:rend]       
        
        # Calculate EAR values
        left_ear = eye_aspect_ratio(leftEye)
        right_ear = eye_aspect_ratio(rightEye)
        
        # Check if both EAR values below threshold
        if left_ear < EAR_THRESH and right_ear < EAR_THRESH:
            ear_counter += 1
        
        else:
            ear_counter = 0
        
        # Check if mouth region exists
        if landmarks[mstart:mend].any():
            
            # Calculate MAR
            mouth = landmarks[mstart:mend]
            mar = mouth_aspect_ratio(mouth)
            
            # Check if MAR above threshold
            if mar > MAR_THRESH:
                mar_counter += 1
            else:
                mar_counter = 0
        
        # Trigger alarm
        if ear_counter >= EAR_CONSEC_FRAMES or mar_counter >= MOUTH_CONSEC_FRAMES:
            threading.Thread(target=alarm, args=("Drowsiness Detected!",)).start()
            ear_counter = 0
            mar_counter = 0
     
    # Display
    cv2.imshow("Drowsiness Detection", frame)

    k = cv2.waitKey(1)
    if k == 27:
        break
        
# Release resources        
cap.release()
cv2.destroyAllWindows()