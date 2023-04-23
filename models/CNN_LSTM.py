from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation, BatchNormalization, Permute, Conv2D, MaxPooling2D
from kapre.utils import Normalization2D

def CNN_LSTM1(input_shape, output_nodes, dropout):
    ## CNN
    model = Sequential()
    model.add(Normalization2D())
    model.add(Conv2D(20, (5, 1), input_shape=input_shape, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 1)))
    model.add(Dropout(dropout))
    model.add(Conv2D(40, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout))
    model.add(Conv2D(1, (5, 1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    
    ## LSTM
    model.add(Lambda(lambda x: K.squeeze(x, -1), name='sqeeze_last_dim'))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(Lambda(lambda q: q[:, -1]))

    ## Dense
    model.add(Dense(128))
    model.add(Dense(32))
    model.add(Dense(output_nodes, activation='softmax', name='output'))

    return model