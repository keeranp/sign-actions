import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image,model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, landmarks):
    mp_drawing.draw_landmarks(image, landmarks.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

#Access the webcam
capture = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while capture.isOpened():
        #Read the webcam feed
        ret, frame = capture.read()

        #Make the detections
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)

        #Show the feed
        cv2.imshow("OpenCV Feed", image)

        if cv2.waitKey(1) == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()