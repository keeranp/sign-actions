import cv2
import numpy as np
import mediapipe as mp

FINGERTIP_IDS = [8, 12, 16, 20]

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, landmarks):
    mp_drawing.draw_landmarks(
        image, landmarks.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )


def draw_prediction(image, finger_count):
    cv2.rectangle(image, (0, 0), (640, 40), (26, 188, 156), -1)
    cv2.putText(
        image,
        str(finger_count),
        (300, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

# Access the webcam
capture = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:
    while capture.isOpened():
        # Read the webcam feed
        ret, frame = capture.read()

        # Make the detections
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)

        # Count the fingers
        if results.right_hand_landmarks:
            right_hand = np.array(
                [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
            )

            fingers = []

            #Thumb
            if right_hand[4][0] > right_hand[3][0]:
                fingers.append(1)
            else:
                fingers.append(0)

            for id in FINGERTIP_IDS:
                if right_hand[id][1] < right_hand[id-2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            draw_prediction(image, fingers.count(1))

        # Show the feed
        cv2.imshow("OpenCV Feed", image)

        if cv2.waitKey(1) == ord("q"):
            break

capture.release()
cv2.destroyAllWindows()
