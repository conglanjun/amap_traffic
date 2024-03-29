import tensorflow as tf
from tensorflow.keras.layers import Dense


class ResnetModel:

    def __init__(self):
        self.config = dict(
            net_size=224,
            rnn_size=256
        )

    def getResnetModel(self):
        modelInput = tf.keras.Input(batch_input_shape=(
            None, 5, self.config['net_size'], self.config['net_size'], 3))
        modelInput0, modelInput1, modelInput2, modelInput3, modelInput4 = tf.split(
            modelInput, [1, 1, 1, 1, 1], 1)
        x0 = tf.squeeze(tf.keras.layers.Lambda(lambda x0: x0)(modelInput0))
        x1 = tf.squeeze(tf.keras.layers.Lambda(lambda x1: x1)(modelInput1))
        x2 = tf.squeeze(tf.keras.layers.Lambda(lambda x2: x2)(modelInput2))
        x3 = tf.squeeze(tf.keras.layers.Lambda(lambda x3: x3)(modelInput3))
        x4 = tf.squeeze(tf.keras.layers.Lambda(lambda x4: x4)(modelInput4))
        net = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(
            self.config['net_size'], self.config['net_size'], 3), pooling='avg')
        self.config['rnn_size'] = 256
        activation = tf.keras.layers.LeakyReLU()
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
        x = tf.keras.layers.Dense(
            self.config['rnn_size'], activation=activation)(ret)
        model = tf.keras.Model(modelInput, x)
        return model
