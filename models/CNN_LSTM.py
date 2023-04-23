from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, BatchNormalization, Conv2D, MaxPooling2D, Lambda
from keras import backend


def CNN_LSTM1(input_shape, output_nodes=12, dropout=0.3):
    ## CNN
    model = Sequential()
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
    model.add(Lambda(lambda x: backend.squeeze(x, -1), name='sqeeze_last_dim'))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(Lambda(lambda q: q[:, -1]))

    ## Dense
    model.add(Dense(128))
    model.add(Dense(32))
    model.add(Dense(output_nodes, activation='softmax', name='output'))

    return model