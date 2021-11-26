import tensorflow as tf
import efficientnet.tfkeras as efn
from tensorflow.keras.layers import Dense
from model.transformer.Transformer import Transformer


class EffModel:

    def __init__(self):
        self.config = dict(
            net_size=224,
            rnn_size=256
        )

    def transformer(self, modelEffOut):
        sample_transformer = Transformer(
            num_layers=2, d_model=self.config['rnn_size'], num_heads=4, dff=1024,
            input_vocab_size=256, target_vocab_size=64, pe_input=256)

        fn_out = sample_transformer(modelEffOut, True, enc_padding_mask=None, look_ahead_mask=None, dec_padding_mask=None)
        return fn_out

    def getEffModel(self, n=0):
        modelInput = tf.keras.Input(batch_input_shape=(None, 5, self.config['net_size'], self.config['net_size'], 3))
        modelInput0, modelInput1, modelInput2, modelInput3, modelInput4 = tf.split(modelInput, [1, 1, 1, 1, 1], 1)
        x0 = tf.squeeze(tf.keras.layers.Lambda(lambda x0: x0)(modelInput0))
        x1 = tf.squeeze(tf.keras.layers.Lambda(lambda x1: x1)(modelInput1))
        x2 = tf.squeeze(tf.keras.layers.Lambda(lambda x2: x2)(modelInput2))
        x3 = tf.squeeze(tf.keras.layers.Lambda(lambda x3: x3)(modelInput3))
        x4 = tf.squeeze(tf.keras.layers.Lambda(lambda x4: x4)(modelInput4))
        net = ''
        if n % 10 == 0:
            net = efn.EfficientNetB0(include_top=False, weights='imagenet', input_shape=(self.config['net_size'], self.config['net_size'], 3), pooling='avg')
        elif n % 10 == 1:
            net = efn.EfficientNetB1(include_top=False, weights='imagenet', input_shape=(self.config['net_size'], self.config['net_size'], 3), pooling='avg')
        elif n % 10 == 2:
            net = efn.EfficientNetB2(include_top=False, weights='imagenet', input_shape=(self.config['net_size'], self.config['net_size'], 3), pooling='avg')
        elif n % 10 == 3:
            net = efn.EfficientNetB3(include_top=False, weights='imagenet', input_shape=(self.config['net_size'], self.config['net_size'], 3), pooling='avg')
        elif n % 10 == 4:
            net = efn.EfficientNetB4(include_top=False, weights='imagenet', input_shape=(self.config['net_size'], self.config['net_size'], 3), pooling='avg')
        elif n % 10 == 5:
            net = efn.EfficientNetB5(include_top=False, weights='imagenet', input_shape=(self.config['net_size'], self.config['net_size'], 3), pooling='avg')
        elif n % 10 == 6:
            net = efn.EfficientNetB6(include_top=False, weights='imagenet', input_shape=(self.config['net_size'], self.config['net_size'], 3), pooling='avg')
        elif n % 10 == 7:
            net = efn.EfficientNetB7(include_top=False, weights='imagenet', input_shape=(self.config['net_size'], self.config['net_size'], 3), pooling='avg')

        activation = tf.keras.layers.LeakyReLU()

        self.config['rnn_size'] = 256
        ret0 = net(x0)
        ret0 = Dense(self.config['rnn_size'], activation=activation)(ret0)
        ret0 = tf.expand_dims(ret0, axis=1)
        ret1 = net(x1)
        ret1 = Dense(self.config['rnn_size'], activation=activation)(ret1)
        ret1 = tf.expand_dims(ret1, axis=1)
        ret2 = net(x2)
        ret2 = Dense(self.config['rnn_size'], activation=activation)(ret2)
        ret2 = tf.expand_dims(ret2, axis=1)
        ret3 = net(x3)
        ret3 = Dense(self.config['rnn_size'], activation=activation)(ret3)
        ret3 = tf.expand_dims(ret3, axis=1)
        ret4 = net(x4)
        ret4 = Dense(self.config['rnn_size'], activation=activation)(ret4)
        ret4 = tf.expand_dims(ret4, axis=1)
        ret = tf.concat([ret0, ret1, ret2, ret3, ret4], axis=1)
        print(ret)
        x = tf.keras.layers.Dense(self.config['rnn_size'], activation=activation)(ret)
        model = tf.keras.Model(modelInput, x)
        return model

    def getEffTFModel(self, n=0):
        modelInput = tf.keras.Input(batch_input_shape=(None, 5, self.config['net_size'], self.config['net_size'], 3))
        modelInput0, modelInput1, modelInput2, modelInput3, modelInput4 = tf.split(modelInput, [1, 1, 1, 1, 1], 1)
        x0 = tf.squeeze(tf.keras.layers.Lambda(lambda x0: x0)(modelInput0))
        x1 = tf.squeeze(tf.keras.layers.Lambda(lambda x1: x1)(modelInput1))
        x2 = tf.squeeze(tf.keras.layers.Lambda(lambda x2: x2)(modelInput2))
        x3 = tf.squeeze(tf.keras.layers.Lambda(lambda x3: x3)(modelInput3))
        x4 = tf.squeeze(tf.keras.layers.Lambda(lambda x4: x4)(modelInput4))
        net = ''
        if n % 10 == 0:
            net = efn.EfficientNetB0(include_top=False, weights='imagenet', input_shape=(self.config['net_size'], self.config['net_size'], 3), pooling='avg')
        elif n % 10 == 1:
            net = efn.EfficientNetB1(include_top=False, weights='imagenet', input_shape=(self.config['net_size'], self.config['net_size'], 3), pooling='avg')
        elif n % 10 == 2:
            net = efn.EfficientNetB2(include_top=False, weights='imagenet', input_shape=(self.config['net_size'], self.config['net_size'], 3), pooling='avg')
        elif n % 10 == 3:
            net = efn.EfficientNetB3(include_top=False, weights='imagenet', input_shape=(self.config['net_size'], self.config['net_size'], 3), pooling='avg')
        elif n % 10 == 4:
            net = efn.EfficientNetB4(include_top=False, weights='imagenet', input_shape=(self.config['net_size'], self.config['net_size'], 3), pooling='avg')
        elif n % 10 == 5:
            net = efn.EfficientNetB5(include_top=False, weights='imagenet', input_shape=(self.config['net_size'], self.config['net_size'], 3), pooling='avg')
        elif n % 10 == 6:
            net = efn.EfficientNetB6(include_top=False, weights='imagenet', input_shape=(self.config['net_size'], self.config['net_size'], 3), pooling='avg')
        elif n % 10 == 7:
            net = efn.EfficientNetB7(include_top=False, weights='imagenet', input_shape=(self.config['net_size'], self.config['net_size'], 3), pooling='avg')

        activation = tf.keras.layers.LeakyReLU()

        self.config['rnn_size'] = 256
        ret0 = net(x0)
        ret0 = Dense(self.config['rnn_size'], activation=activation)(ret0)
        ret0 = tf.expand_dims(ret0, axis=1)
        ret0 = self.transformer(ret0)
        ret1 = net(x1)
        ret1 = Dense(self.config['rnn_size'], activation=activation)(ret1)
        ret1 = tf.expand_dims(ret1, axis=1)
        ret1 = self.transformer(ret1)
        ret2 = net(x2)
        ret2 = Dense(self.config['rnn_size'], activation=activation)(ret2)
        ret2 = tf.expand_dims(ret2, axis=1)
        ret2 = self.transformer(ret2)
        ret3 = net(x3)
        ret3 = Dense(self.config['rnn_size'], activation=activation)(ret3)
        ret3 = tf.expand_dims(ret3, axis=1)
        ret3 = self.transformer(ret3)
        ret4 = net(x4)
        ret4 = Dense(self.config['rnn_size'], activation=activation)(ret4)
        ret4 = tf.expand_dims(ret4, axis=1)
        ret4 = self.transformer(ret4)
        ret = tf.concat([ret0, ret1, ret2, ret3, ret4], axis=1)
        print(ret)
        x = tf.keras.layers.Dense(self.config['rnn_size'], activation=activation)(ret)
        model = tf.keras.Model(modelInput, x)
        return model