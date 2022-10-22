from keras.models import Sequential
from keras.layers import LSTM, Dense

def build_model(n_actions):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation="relu", input_shape=(30, 63)))
    model.add(LSTM(128, return_sequences=True, activation="relu"))
    model.add(LSTM(64, return_sequences=False, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(n_actions, activation="softmax"))
    return model