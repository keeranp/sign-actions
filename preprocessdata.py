import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

DATA_PATH = os.path.join("MP_Data")
actions = ["one", "two", "three", "four", "five"]
# Thirty videos
n_sequences = 30
# Twenty frames per video
sequence_length = 20

def load_dataset():
    label_map = {label: num for num, label in enumerate(actions)}

    sequences, labels = [], []

    for action in actions:
        for sequence in range(n_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(
                    os.path.join(DATA_PATH, action, str(sequence), str(frame_num) + ".npy")
                )
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    x = np.array(sequences)
    y = to_categorical(labels).astype(int)

    return train_test_split(x, y, test_size=0.2)
