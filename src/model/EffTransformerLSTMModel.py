import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Dropout
from model.transformer.Transformer import Transformer
from model.EffModel import EffModel


class EffTransformerLSTMModel:

    def __init__(self):
        self.EffModel = EffModel()
        self.config = dict(
            num_class=3,
            net_size=224,
            rnn_size=256
        )

    def getEffTransformerLSTMModel(self, n):
        modelEff = self.EffModel.getEffModel(n)
        modelEffOut = Dropout(0.3)(modelEff.output)
        sample_transformer = Transformer(
            num_layers=2, d_model=self.config['rnn_size'], num_heads=4, dff=1024,
            input_vocab_size=256, target_vocab_size=64, pe_input=256)

        fn_out = sample_transformer(modelEffOut, True, enc_padding_mask=None, look_ahead_mask=None,
                                    dec_padding_mask=None)
        bi_x_out = Dropout(0.3)(fn_out)
        x = LSTM(self.config['rnn_size'])(bi_x_out)
        ld_x_out = Dropout(0.3)(x)

        outputs = Dense(self.config['num_class'], activation="softmax")(ld_x_out)

        model = tf.keras.Model(modelEff.input, outputs)
        model.summary()
        return model

    def getEffTransformerBLSTMModel(self, n):
        modelEff = self.EffModel.getEffTFModel(n)
        modelEffOut = Dropout(0.3)(modelEff.output)
        x = LSTM(self.config['rnn_size'])(modelEffOut)
        ld_x_out = Dropout(0.3)(x)
        outputs = Dense(self.config['num_class'], activation="softmax")(ld_x_out)

        model = tf.keras.Model(modelEff.input, outputs)
        model.summary()
        return model
