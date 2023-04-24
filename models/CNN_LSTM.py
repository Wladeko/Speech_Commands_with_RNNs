from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation, BatchNormalization, Permute, Conv2D, MaxPooling2D
from kapre.utils import Normalization2D
from tensorflow.keras import layers as L
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model

def CNN_LSTM1(input_shape, output_nodes, dropout):
        # CNN part
    input = L.Input(input_shape, name='input')

    x = Normalization2D(int_axis=0, name='mel_stft_norm')(x)

    x = L.Permute((2, 1, 3))(x)

    x = L.Conv2D(20, (5, 1), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.MaxPooling2D((2, 1))(x)
    x = L.Dropout(dropout)(x)

    x = L.Conv2D(40, (3, 3), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.MaxPooling2D((2, 2))(x)
    x = L.Dropout(dropout)(x)

    x = L.Conv2D(1, (5, 1), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)


    # LSTM part
    x = L.Lambda(lambda x: K.squeeze(x, -1), name='sqeeze_last_dim')(x)
    x = L.Bidirectional(L.LSTM(64, return_sequences=True))(x)
    x = L.Bidirectional(L.LSTM(64, return_sequences=True))(x)  # it returns [b_s, seq_len, vec_dim]

    xFirst = L.Lambda(lambda q: q[:, -1])(x)  # to [b_s, vec_dim]
    query = L.Dense(128)(xFirst)

    # Attention Part
    attScores = L.Dot(axes=[1, 2])([query, x])
    attScores = L.Softmax(name='attSoftmax')(attScores)

    # Rescaling
    attVector = L.Dot(axes=[1, 1])([attScores, x])

    x = L.Dense(64, activation='relu')(attVector)
    x = L.Dense(32)(x)

    output = L.Dense(output_nodes, activation='softmax', name='output')(x)

    model = Model(inputs=[input], outputs=[output])

    return model