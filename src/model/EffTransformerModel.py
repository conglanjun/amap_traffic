import tensorflow as tf
from tensorflow.keras.layers import Dense
from model.transformer.Transformer import Transformer


class EffTransformerModel:

    def __init__(self):
        pass

    def getEffTransformerModel(self, n):
        modelEff = self.EffModel.getEffModel(n)
        # enc_input = Dense(2048, activation=tf.keras.layers.LeakyReLU())(modelEff.output)
        sample_transformer = Transformer(
            num_layers=2, d_model=self.config['rnn_size'], num_heads=4, dff=1024,
            input_vocab_size=256, target_vocab_size=64, pe_input=256)

        # enc_input_reshape = tf.reshape(enc_input, (batchSize, 8, self.config['rnn_size']))
        fn_out = sample_transformer(modelEff.output, True, enc_padding_mask=None, look_ahead_mask=None, dec_padding_mask=None)

        # fn_reshape_out = tf.reshape(fn_out, (-1, 1280 * 1024))
        fn_reshape_out = tf.keras.layers.Flatten()(fn_out)

        outputs = Dense(self.config['num_class'], activation="softmax")(fn_reshape_out)

        model = tf.keras.Model(modelEff.input, outputs)
        model.summary()
        return model
