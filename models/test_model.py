from keras import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation, BatchNormalization
from sklearn.preprocessing import OneHotEncoder


def get_test_model(input_shape, dropout=0.3, num_classes=12):
    model = Sequential()
    model.add(BatchNormalization())
    model.add(LSTM(
        256,
        input_shape=input_shape,
        return_sequences=True
    ))
    model.add(Dropout(dropout))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model