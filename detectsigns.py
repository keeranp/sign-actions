import cv2
import os
import numpy as np
import mediapipe as mp
from model import build_model

DATA_PATH = os.path.join("MP_Data")
actions = ["one", "two", "three", "four", "five"]
# Thirty videos
n_sequences = 30
# Thirty frames per video
sequence_length = 30

sequence = []
sentence = []
predictions = []
threshold = 0.7

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


def draw_prediction(sign, image):
    cv2.rectangle(image, (0, 0), (640, 40), (255, 128, 0), -1)
    cv2.putText(
        image,
        sign,
        (3, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


colors = [
    (26, 188, 156),
    (52, 152, 219),
    (52, 73, 94),
    (241, 196, 15),
    (231, 76, 60),
]


def draw_prob(image, res):
    for i, prob in enumerate(res):
        cv2.rectangle(
            image, (0, 60 + i * 40), (int(prob * 100), 90 + i * 40), colors[i], -1
        )
        cv2.putText(
            image,
            actions[i],
            (0, 85 + i * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


# Build the model
model = build_model(len(actions))
model.compile(
    optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
)

model.load_weights("action.h5")


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

        # Prediction
        if results.right_hand_landmarks:
            right_hand = np.array(
                [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
            ).flatten()
        else:
            right_hand = np.zeros(21 * 3)

        sequence.append(right_hand)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))

            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

        if len(sentence) > 5:
            sentence = sentence[-5:]

        draw_prediction(sentence, image)

        # Show the feed
        cv2.imshow("OpenCV Feed", image)

        if cv2.waitKey(1) == ord("q"):
            break

capture.release()
cv2.destroyAllWindows()
