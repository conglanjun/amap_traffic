import tensorflow as tf
from model.EffModel import EffModel
from tensorflow.keras.layers import LSTM, Dense


class EffLSTMModel:

    def __init__(self):
        self.config = dict(
            rnn_size=256
        )
        self.EffModel = EffModel()

    def getEffLSTMModel(self, n):
        modelEff = self.EffModel.getEffModel(n)
        x = LSTM(self.config['rnn_size'])(modelEff.output)
        outputs = Dense(self.config['num_class'], activation="softmax")(x)

        model = tf.keras.Model(modelEff.input, outputs)
        model.summary()
        return model
