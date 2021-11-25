import tensorflow as tf
from model.EffModel import EffModel
from tensorflow.keras.layers import Dense, Dropout, GRU, Bidirectional


class EffBiGruModel:

    def __init__(self):
        self.config = dict(
            num_class=3,
            net_size=224,
            rnn_size=256
        )
        self.EffModel = EffModel()

    def getEffBiGruModel(self, n):
        modelEff = self.EffModel.getEffModel(n)
        modelEffOut = Dropout(0.2)(modelEff.output)  # (None, 5, 256)
        x = Bidirectional(GRU(self.config['rnn_size']), merge_mode='concat', weights=None)(modelEffOut)  # (None, 512)
        x = Dropout(0.2)(x)
        outputs = Dense(self.config['num_class'], activation="softmax")(x)

        model = tf.keras.Model(modelEff.input, outputs)
        model.summary()
        return model
