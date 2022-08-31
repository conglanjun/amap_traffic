import tensorflow as tf
from model.EffModel import EffModel
from model.VggModel import VggModel
from model.ResnetModel import ResnetModel
from tensorflow.keras.layers import LSTM, Dense, Dropout


class EffModelDense:

    def __init__(self):
        self.config = dict(
            num_class=3,
            net_size=224,
            rnn_size=256
        )
        self.EffModel = EffModel()
        self.VggModel = VggModel()
        self.ResnetModel = ResnetModel()

    def getEffModelDense(self, n):
        modelEff = self.EffModel.getEffModel(n)
        modelEffOut = Dropout(0.2)(modelEff.output)  # (None, 5, 256)
        # x = LSTM(self.config['rnn_size'])(modelEffOut)  # (None, 256)
        x = tf.reduce_mean(modelEffOut, axis=1)
        # x = Dropout(0.2)(x)
        outputs = Dense(self.config['num_class'], activation="softmax")(x)

        model = tf.keras.Model(modelEff.input, outputs)  # (None, 3)
        model.summary()
        return model

    def getVggModelDense(self):
        modelVgg = self.VggModel.getVggModel()
        modelVggOut = Dropout(0.2)(modelVgg.output)  # (None, 5, 256)
        x = tf.reduce_mean(modelVggOut, axis=1)
        outputs = Dense(self.config['num_class'], activation="softmax")(x)

        model = tf.keras.Model(modelVgg.input, outputs)  # (None, 3)
        model.summary()
        return model

    def getResnetModelDense(self):
        modelResnet = self.ResnetModel.getResnetModel()
        modelResnetOut = Dropout(0.2)(modelResnet.output)  # (None, 5, 256)
        x = tf.reduce_mean(modelResnetOut, axis=1)
        outputs = Dense(self.config['num_class'], activation="softmax")(x)

        model = tf.keras.Model(modelResnet.input, outputs)  # (None, 3)
        model.summary()
        return model
