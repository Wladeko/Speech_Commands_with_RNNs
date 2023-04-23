from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, Dense, Activation, BatchNormalization
from sklearn.preprocessing import OneHotEncoder

def GRU1(input_shape, output_nodes, dropout):
    model = Sequential()
    model.add(BatchNormalization())
    model.add(GRU(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(GRU(128, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(Dense(output_nodes))
    model.add(Activation('softmax'))
    return model


def GRU2(input_shape, output_nodes, dropout):
    model = Sequential()
    model.add(BatchNormalization())
    model.add(GRU(256, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(GRU(128, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(GRU(128))
    model.add(Dense(128))
    model.add(Dropout(dropout))
    model.add(Dense(output_nodes))
    model.add(Activation('softmax'))
    return model

def GRU3(input_shape, output_nodes, dropout):
    model = Sequential()
    model.add(BatchNormalization())
    model.add(GRU(512, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(GRU(256, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(GRU(256, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(GRU(256))
    model.add(Dense(256))
    model.add(Dropout(dropout))
    model.add(Dense(output_nodes))
    model.add(Activation('softmax'))
    return model