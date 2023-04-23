from tensorflow.keras import Sequential
from tensorflow.keras.layers import SimpleRNN, Dropout, Dense, Activation, BatchNormalization
from sklearn.preprocessing import OneHotEncoder

def GRU(X_t, y_t, dropout):
    model = Sequential()
    model.add(BatchNormalization())
    model.add(GRU(
        128,
        input_shape=X.shape[1:],
        return_sequences=True
    ))
    model.add(Dropout(dropout))
    model.add(GRU(128, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(GRU(128))
    model.add(Dense(128))
    model.add(Dropout(dropout))
    model.add(Dense(y.shape[1]))
    model.add(Activation('softmax'))

    return model