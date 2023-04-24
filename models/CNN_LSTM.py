from tensorflow.keras import layers as L
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

def CNN_LSTM1(input_shape, output_nodes=12, dropout=0.3):
    # CNN
    input = L.Input(input_shape, name='input')
    x = L.Reshape([input_shape[0], input_shape[1], 1])(input)
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

    # LSTM 
    x = L.Lambda(lambda x: K.squeeze(x, -1), name='sqeeze_last_dim')(x)
    x = L.Bidirectional(L.LSTM(64, return_sequences=True))(x)
    x = L.Bidirectional(L.LSTM(64, return_sequences=True))(x)  
    xFirst = L.Lambda(lambda q: q[:, -1])(x) 
    query = L.Dense(128)(xFirst)

    # Attention
    attScores = L.Dot(axes=[1, 2])([query, x])
    attScores = L.Softmax(name='attSoftmax')(attScores)

    # Rescaling
    attVector = L.Dot(axes=[1, 1])([attScores, x])
    x = L.Dense(64, activation='relu')(attVector)
    x = L.Dense(32)(x)
    output = L.Dense(output_nodes, activation='softmax', name='output')(x)

    model = Model(inputs=[input], outputs=[output])

    return model