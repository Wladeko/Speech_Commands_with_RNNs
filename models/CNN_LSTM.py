from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers as L
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import optimizers

from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from kapre.time_frequency import Melspectrogram, Spectrogram
from kapre.utils import Normalization2D

def CNN_LSTM(X_t, y_t, dropout):
    # CNN part
    input = L.Input(X_train.shape[1:], name='input')
    x = L.Reshape((1, -1))(input)

    m = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, None),
                    padding='same', sr=16000, n_mels=80,
                    fmin=40.0, fmax=16000 / 2, power_melgram=1.0,
                    return_decibel_melgram=True, trainable_fb=False,
                    trainable_kernel=False,
                    name='mel_stft')
    m.trainable = False
    x = m(x)

    x = Normalization2D(int_axis=0, name='mel_stft_norm')(x)

    x = L.Permute((2, 1, 3))(x)

    x = L.Conv2D(20, (5, 1), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.MaxPooling2D((2, 1))(x)
    x = L.Dropout(0.1)(x)

    x = L.Conv2D(40, (3, 3), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.MaxPooling2D((2, 2))(x)
    x = L.Dropout(0.1)(x)

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

    output = L.Dense(31, activation='softmax', name='output')(x)

    model = Model(inputs=[input], outputs=[output])

    return model