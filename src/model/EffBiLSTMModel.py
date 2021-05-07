import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from model.EffModel import EffModel


class EffBiLSTMModel:

    def __init__(self):
        self.config = dict(
            num_class=3,
            net_size=224,
            rnn_size=256
        )
        self.EffModel = EffModel()

    def getEffBiLSTMModel(self, n):
        modelEff = self.EffModel.getEffModel(n)
        modelEffOut = Dropout(0.2)(modelEff.output)
        bi_x = Bidirectional(LSTM(self.config['rnn_size']), merge_mode='concat', weights=None)(modelEffOut)
        bi_x_out = Dropout(0.2)(bi_x)
        outputs = Dense(self.config['num_class'], activation="softmax")(bi_x_out)

        model = tf.keras.Model(modelEff.input, outputs)
        model.summary()
        return model
