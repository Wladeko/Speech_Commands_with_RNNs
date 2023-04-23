from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation, BatchNormalization
from sklearn.preprocessing import OneHotEncoder

def LSTM1(input_shape, output_nodes, dropout):
    model = Sequential()
    model.add(BatchNormalization())
    model.add(LSTM(256,input_shape=input_shape,return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(Dense(output_nodes))
    model.add(Activation('softmax'))
    return model

def LSTM2(input_shape, output_nodes, dropout):
    model = Sequential()
    model.add(BatchNormalization())
    model.add(LSTM(256, input_shape=input_shape,return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(dropout))
    model.add(Dense(output_nodes))
    model.add(Activation('softmax'))
    return model

def LSTM3(input_shape, output_nodes, dropout):
    model = Sequential()
    model.add(BatchNormalization())
    model.add(LSTM(512, input_shape=input_shape,return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(512))
    model.add(Dropout(dropout))
    model.add(LSTM(256))
    model.add(Dropout(dropout))
    model.add(Dense(256))
    model.add(Dropout(dropout))
    model.add(Dense(output_nodes))
    model.add(Activation('softmax'))
    return model