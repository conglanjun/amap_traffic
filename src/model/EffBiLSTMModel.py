import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
from model.EffModel import EffModel


class EffBiLSTMModel:

    def __init__(self):
        self.config = dict(
            rnn_size=256
        )
        self.EffModel = EffModel()

    def getEffBiLSTMModel(self, n):
        modelEff = self.EffModel.getEffModel(n)
        bi_x = Bidirectional(LSTM(self.config['rnn_size']), merge_mode='concat', weights=None)(modelEff.output)
        outputs = Dense(self.config['num_class'], activation="softmax")(bi_x)

        model = tf.keras.Model(modelEff.input, outputs)
        model.summary()
        return model
