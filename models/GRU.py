from keras import Sequential
from keras.layers import Dropout, Dense, Activation, BatchNormalization, GRU


def GRU1(input_shape, output_nodes=12, dropout=0.3):
    model = Sequential()
    model.add(BatchNormalization())
    model.add(GRU(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(GRU(128, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(output_nodes))
    model.add(Activation('softmax'))
    return model


def GRU2(input_shape, output_nodes=12, dropout=0.3):
    model = Sequential()
    model.add(BatchNormalization())
    model.add(GRU(256, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(GRU(128, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(GRU(128, return_sequences=False))
    model.add(Dense(128))
    model.add(Dropout(dropout))
    model.add(Dense(output_nodes))
    model.add(Activation('softmax'))
    return model

def GRU3(input_shape, output_nodes=12, dropout=0.3):
    model = Sequential()
    model.add(BatchNormalization())
    model.add(GRU(512, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(GRU(256, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(GRU(256, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(GRU(256, return_sequences=False))
    model.add(Dense(256))
    model.add(Dropout(dropout))
    model.add(Dense(output_nodes))
    model.add(Activation('softmax'))
    return model