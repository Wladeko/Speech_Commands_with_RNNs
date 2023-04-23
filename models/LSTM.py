from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation, BatchNormalization
from sklearn.preprocessing import OneHotEncoder

def LSTM(X_t, y_t, dropout):
    model = Sequential()
    model.add(BatchNormalization())
    model.add(LSTM(
        256,
        input_shape=X_t.shape[1:],
        return_sequences=True
    ))
    model.add(Dropout(dropout))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(dropout))
    model.add(Dense(y_t.shape[1]))
    model.add(Activation('softmax'))

    return model