import tensorflow as tf
from keras.layers import Dropout, Dense, Activation, BatchNormalization, GRU, InputLayer, Reshape

class TestGRU(tf.keras.Model):

    def __init__(self, input_shape, output_nodes, dropout):
        super().__init__()
        self.input1 = InputLayer(input_shape=input_shape)
        self.reshape1 = Reshape((input_shape, 1))
        self.normalization = BatchNormalization()
        self.GRU1 = GRU(128, return_sequences=True)
        self.dropout1 = Dropout(dropout)
        # self.GRU2 = GRU(128, return_sequences=True)
        # self.dropout2 = Dropout(dropout)
        # self.GRU3 = GRU(256)
        self.dense1 = Dense(256)
        self.dropout3 = Dropout(dropout)
        self.dense2 = Dense(output_nodes)
        self.activation = Activation('softmax')

    def call(self, inputs, training=False):
        x = self.input1(inputs)
        x = self.reshape1(x)
        x = self.normalization(x)
        x = self.GRU1(x)
        if training:
            x = self.dropout1(x, training=training)
        # x = self.GRU2(x)
        # if training:
        #     x = self.dropout2(x, training=training)
        # x = self.GRU3(x)
        x = self.dense1(x)
        if training:
            x = self.dropout3(x, training=training)
        x = self.dense2(x)
        x = self.activation(x)
    
        return x
