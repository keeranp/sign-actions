import cv2
import os
import numpy as np
import mediapipe as mp

DATA_PATH = os.path.join("MP_Data")
actions = ["one", "two", "three", "four", "five"]
# Thirty videos
n_sequences = 30
# Thirty frames per video
sequence_length = 30

for action in actions:
    for sequence in range(n_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

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


# Access the webcam
capture = cv2.VideoCapture(0)

def draw_information(action, sequence, frame_num, image):
    if frame_num == 0:
        cv2.putText(
                        image,
                        "STARTING COLLECTION",
                        (120, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

        cv2.putText(
                        image,
                        "COLLECTING SEQUENCE {} OF ACTION {}".format(sequence, action),
                        (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    # Show the feed
        cv2.imshow("OpenCV Feed", image)
        cv2.waitKey(2000)
    else:
        cv2.putText(
                        image,
                        "COLLECTING SEQUENCE {} OF ACTION {}".format(sequence, action),
                        (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    # Show the feed
        cv2.imshow("OpenCV Feed", image)

with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:
    for action in actions:
        for sequence in range(n_sequences):
            for frame_num in range(sequence_length):
                # Read the webcam feed
                ret, frame = capture.read()

                # Make the detections
                image, results = mediapipe_detection(frame, holistic)
                draw_landmarks(image, results)

                draw_information(action, sequence, frame_num, image)

                if results.right_hand_landmarks:
                    right_hand = np.array(
                        [
                            [res.x, res.y, res.z]
                            for res in results.right_hand_landmarks.landmark
                        ]
                    ).flatten()
                else:
                    right_hand = np.zeros(21 * 3)

                # Save landmarks
                npy_path = os.path.join(
                    DATA_PATH, action, str(sequence), str(frame_num)
                )
                np.save(npy_path, right_hand)

                if cv2.waitKey(1) == ord("q"):
                    break

capture.release()
cv2.destroyAllWindows()
