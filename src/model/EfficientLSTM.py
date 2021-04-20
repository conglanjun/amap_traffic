import tensorflow as tf
import tensorflow.keras.backend as K
import efficientnet.tfkeras as efn
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Activation, Dropout

import numpy as np
import PIL
import sys
sys.path.append("..")
from util.DataHandler import DataHandler
from tqdm import tqdm
import random
import os
import json


class EfficientLSTM:

    def __init__(self):
        self.config = dict(
            batch_size=16,
            max_num_img=5,
            net_size=512,
            rnn_size=128,
            num_class=3,
            epochs=12,
            optimizer='adam',
            dropout=0.2
        )
        realPath = os.path.realpath(__file__)
        subStr = realPath[:realPath[: realPath[: realPath.rindex('/')].rindex('/')].rindex('/')]
        self.subStr = subStr
        # self.train_json_path = subStr + '/data/amap_traffic_annotations_train.json'
        self.train_json_path = subStr + '/data/amap_traffic_augment.json'
        self.test_json_path = subStr + '/data/amap_traffic_annotations_test.json'
        self.data_path = subStr + '/data/amap_traffic_train_0712/'
        self.data_test_path = subStr + '/data/amap_traffic_test_0712/'
        self.PREMODELPATH = subStr + '/src/model/checkpoint/' + "ep004-loss0.041-val_loss0.104.h5"

    def getEffModel(self, i):
        modelInput = tf.keras.Input(batch_input_shape=(None, 5, self.config['net_size'], self.config['net_size'], 3), name=f"imgInput{i}")
        modelInput0, modelInput1, modelInput2, modelInput3, modelInput4 = tf.split(modelInput, [1, 1, 1, 1, 1], 1)
        x0 = tf.squeeze(tf.keras.layers.Lambda(lambda x0: x0)(modelInput0))
        x1 = tf.squeeze(tf.keras.layers.Lambda(lambda x1: x1)(modelInput1))
        x2 = tf.squeeze(tf.keras.layers.Lambda(lambda x2: x2)(modelInput2))
        x3 = tf.squeeze(tf.keras.layers.Lambda(lambda x3: x3)(modelInput3))
        x4 = tf.squeeze(tf.keras.layers.Lambda(lambda x4: x4)(modelInput4))

        net = efn.EfficientNetB0(include_top=False, weights='imagenet',
                           input_shape=(self.config['net_size'], self.config['net_size'], 3),
                           pooling='avg')
        net._name = f"efficientnet-b0-{i}"
        ret0 = net(x0)
        print(ret0)
        self.config['rnn_size'] = ret0.shape[1]
        ret0 = Dropout(self.config['dropout'])(ret0)
        ret1 = net(x1)
        ret1 = Dropout(self.config['dropout'])(ret1)
        ret2 = net(x2)
        ret2 = Dropout(self.config['dropout'])(ret2)
        ret3 = net(x3)
        ret3 = Dropout(self.config['dropout'])(ret3)
        ret4 = net(x4)
        ret4 = Dropout(self.config['dropout'])(ret4)
        ret = tf.concat([ret0, ret1, ret2, ret3, ret4], axis=1)
        print(ret)
        # x = tf.keras.layers.Dense(512, activation='relu')(ret)

        model = tf.keras.Model(modelInput, ret, name=f'effNetwork{i}')
        return model

    def getEffModel1(self, i):
        modelInput = tf.keras.Input(batch_input_shape=(None, 5, self.config['net_size'], self.config['net_size'], 3), name=f"imgInput{i}")
        modelInput0, modelInput1, modelInput2, modelInput3, modelInput4 = tf.split(modelInput, [1, 1, 1, 1, 1], 1)
        x0 = tf.squeeze(tf.keras.layers.Lambda(lambda x0: x0)(modelInput0))
        x1 = tf.squeeze(tf.keras.layers.Lambda(lambda x1: x1)(modelInput1))
        x2 = tf.squeeze(tf.keras.layers.Lambda(lambda x2: x2)(modelInput2))
        x3 = tf.squeeze(tf.keras.layers.Lambda(lambda x3: x3)(modelInput3))
        x4 = tf.squeeze(tf.keras.layers.Lambda(lambda x4: x4)(modelInput4))

        net = efn.EfficientNetB0(include_top=False, weights='imagenet',
                           input_shape=(self.config['net_size'], self.config['net_size'], 3),
                           pooling='avg')
        net._name = f"efficientnet-b0-{i}"

        activation = "relu"
        activation = tf.keras.layers.LeakyReLU()

        ret0 = net(x0)
        self.config['rnn_size'] = 256
        ret0 = Dense(self.config['rnn_size'], activation=activation)(ret0)
        ret0 = Dropout(self.config['dropout'])(ret0)
        ret1 = net(x1)
        ret1 = Dense(self.config['rnn_size'], activation=activation)(ret1)
        ret1 = Dropout(self.config['dropout'])(ret1)
        ret2 = net(x2)
        ret2 = Dense(self.config['rnn_size'], activation=activation)(ret2)
        ret2 = Dropout(self.config['dropout'])(ret2)
        ret3 = net(x3)
        ret3 = Dense(self.config['rnn_size'], activation=activation)(ret3)
        ret3 = Dropout(self.config['dropout'])(ret3)
        ret4 = net(x4)
        ret4 = Dense(self.config['rnn_size'], activation=activation)(ret4)
        ret4 = Dropout(self.config['dropout'])(ret4)
        print(ret4)
        ret = tf.concat([ret0, ret1, ret2, ret3, ret4], axis=1)
        x = tf.keras.layers.Dense(self.config['rnn_size'] * 5, activation=activation)(ret)
        # x = tf.keras.layers.Dense(512, activation=activation)(x)
        # outputs = Dense(self.config['num_class'], activation="softmax")(x)

        model = tf.keras.Model(modelInput, x, name=f'effNetwork{i}')
        return model

    def getEffLSTMModel(self):
        modelEff = self.getEffModel(0)

        modelEffOutput = tf.expand_dims(modelEff.output, axis=1)

        x = Bidirectional(LSTM(self.config['rnn_size'], return_sequences=True))(modelEffOutput)
        x = Dropout(self.config['dropout'])(x)
        x = Bidirectional(LSTM(self.config['rnn_size']))(x)
        x = Dropout(self.config['dropout'])(x)
        outputs = Dense(self.config['num_class'], activation="softmax")(x)

        model = tf.keras.Model(modelEff.input, outputs)
        model.summary()

        return model

    def getEffLSTMModel1(self):
        modelEff = self.getEffModel1(0)
        # model = modelEff
        modelEffOutput = tf.expand_dims(modelEff.output, axis=1)
        activation = tf.keras.layers.LeakyReLU()

        # x = Bidirectional(LSTM(self.config['rnn_size'], return_sequences=True))(modelEffOutput)
        # x = Dropout(self.config['dropout'])(x)
        # x = Bidirectional(LSTM(self.config['rnn_size']))(x)
        x = LSTM(self.config['rnn_size'])(modelEffOutput)
        x = Dropout(self.config['dropout'])(x)
        x = Dense(self.config['rnn_size'], activation=activation)(x)
        outputs = Dense(self.config['num_class'], activation="softmax")(x)

        model = tf.keras.Model(modelEff.input, outputs)
        model.summary()

        return model

    def train(self):
        handler = DataHandler(self.train_json_path, self.test_json_path, self.data_path)
        # model = self.getEffLSTMModel()
        model = self.getEffLSTMModel1()

        if os.path.exists(self.PREMODELPATH):
            print('--load!--:', self.PREMODELPATH)
            model.load_weights(self.PREMODELPATH)

        rms = tf.keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
        adam = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

        checkpoint = ModelCheckpoint(self.subStr + '/src/model/checkpoint/' + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                     monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
        print('save path:', self.subStr + '/src/model/checkpoint/')

        batchSize = 8

        loadDict = handler.readJson(handler.train_json_path)
        batchItems = loadDict['annotations']
        random.shuffle(batchItems)
        numValidation = len(batchItems) // 10
        numTrain = len(batchItems) - numValidation

        trainData = batchItems[: numTrain]
        valiData = batchItems[numTrain:]

        model.fit_generator(handler.dataGenerator(trainData, batchSize, self.config['num_class']),
                            steps_per_epoch=max(1, numTrain // batchSize),
                            # steps_per_epoch=max(1, numTrain),
                            validation_data=handler.dataGenerator(valiData, batchSize, self.config['num_class']),
                            validation_steps=max(1, numValidation // batchSize),
                            # validation_steps=max(1, numValidation),
                            epochs=10,
                            initial_epoch=0,
                            callbacks=[checkpoint])
        model.save_weights(self.subStr + '/src/model/checkpoint/' + 'trained_weights_final.h5')

    def predict(self):
        handler = DataHandler(self.train_json_path, self.test_json_path, self.data_test_path)
        model = self.getEffLSTMModel1()

        model.load_weights(self.PREMODELPATH)
        loadDict = handler.readJson(handler.test_json_path)
        batchItems = loadDict['annotations']

        # for index, imgLine in enumerate(batchItems):
        #     img5 = handler.dataPredict(imgLine, index)
        #     result = model.predict_on_batch(img5)
        #     print(result)

        batchSize = 2
        print(len(batchItems))
        result = model.predict_generator(handler.dataPredict(batchItems, batchSize), len(batchItems) // batchSize, verbose=1)
        print(result.shape)
        with open(self.subStr + '/result.log', 'w') as rf:
            for item in result:
                rf.write(str(item) + ',\n')
        result = np.argmax(result, axis=1)
        print(result)
        for index, item in enumerate(batchItems):
            item['status'] = str(result[index])
        with open(self.subStr + '/amap_submission.json', 'w') as wf:
            json.dump(loadDict, wf)





