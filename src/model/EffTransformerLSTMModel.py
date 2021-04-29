import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from model.transformer.Transformer import Transformer


class EffTransformerLSTMModel:

    def __init__(self):
        pass

    def getEffTransformerLSTMModel(self, n):
        modelEff = self.getEffModel(n)
        sample_transformer = Transformer(
            num_layers=2, d_model=self.config['rnn_size'], num_heads=4, dff=1024,
            input_vocab_size=256, target_vocab_size=64, pe_input=256)

        fn_out = sample_transformer(modelEff.output, True, enc_padding_mask=None, look_ahead_mask=None,
                                    dec_padding_mask=None)

        x = LSTM(self.config['rnn_size'])(fn_out)

        outputs = Dense(self.config['num_class'], activation="softmax")(x)

        model = tf.keras.Model(modelEff.input, outputs)
        model.summary()
        return model
