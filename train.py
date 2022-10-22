import os
import numpy as np
from model import build_model
from keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from preprocessdata import load_dataset

DATA_PATH = os.path.join("MP_Data")
actions = ["one", "two", "three", "four", "five"]
# Thirty videos
n_sequences = 30
# Thirty frames per video
sequence_length = 30

label_map = {label: num for num, label in enumerate(actions)}

model = build_model(len(actions))

log_dir = os.path.join("Logs")
tb_callback = TensorBoard(log_dir=log_dir)

model.compile(
    optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
)

x_train, x_test, y_train, y_test = load_dataset()

model.fit(x_train, y_train, epochs=2000, callbacks=[tb_callback])
model.save("action.h5")

#Test the model
model.load_weights("action.h5")

yhat = model.predict(x_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(multilabel_confusion_matrix(ytrue, yhat))
print(accuracy_score(ytrue, yhat))