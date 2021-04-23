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
from util.Score import Score
from tqdm import tqdm
import random
import os
import json, array
from model.transformer.Transformer import Transformer

class EfficientLSTMV2:

    def __init__(self):
        self.config = dict(
            batch_size=16,
            max_num_img=5,
            net_size=224,
            rnn_size=256,
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
        self.test_json_path = subStr + '/data/amap_traffic_annotations_test_answer.json'
        self.data_path = subStr + '/data/amap_traffic_train_0712/'
        self.data_test_path = subStr + '/data/amap_traffic_test_0712/'
        self.PREMODELPATH = subStr + '/src/model/checkpoint/' + "B4/trained_weights_final.h5"

    def getEffModel(self):
        modelInput = tf.keras.Input(batch_input_shape=(None, 5, self.config['net_size'], self.config['net_size'], 3))
        modelInput0, modelInput1, modelInput2, modelInput3, modelInput4 = tf.split(modelInput, [1, 1, 1, 1, 1], 1)
        x0 = tf.squeeze(tf.keras.layers.Lambda(lambda x0: x0)(modelInput0))
        x1 = tf.squeeze(tf.keras.layers.Lambda(lambda x1: x1)(modelInput1))
        x2 = tf.squeeze(tf.keras.layers.Lambda(lambda x2: x2)(modelInput2))
        x3 = tf.squeeze(tf.keras.layers.Lambda(lambda x3: x3)(modelInput3))
        x4 = tf.squeeze(tf.keras.layers.Lambda(lambda x4: x4)(modelInput4))
        net = efn.EfficientNetB0(include_top=False, weights='imagenet',
                                 input_shape=(self.config['net_size'], self.config['net_size'], 3),
                                 pooling='avg')

        activation = tf.keras.layers.LeakyReLU()

        self.config['rnn_size'] = 256
        ret0 = net(x0)
        ret0 = Dense(self.config['rnn_size'], activation=activation)(ret0)
        ret1 = net(x1)
        ret1 = Dense(self.config['rnn_size'], activation=activation)(ret1)
        ret2 = net(x2)
        ret2 = Dense(self.config['rnn_size'], activation=activation)(ret2)
        ret3 = net(x3)
        ret3 = Dense(self.config['rnn_size'], activation=activation)(ret3)
        ret4 = net(x4)
        ret4 = Dense(self.config['rnn_size'], activation=activation)(ret4)
        ret = tf.concat([ret0, ret1, ret2, ret3, ret4], axis=1)
        print(ret)
        x = tf.keras.layers.Dense(self.config['rnn_size'] * 5, activation=activation)(ret)
        model = tf.keras.Model(modelInput, x)
        return model


    def getEffLSTMModel(self):
        modelEff = self.getEffModel()
        modelEffOutput = tf.expand_dims(modelEff.output, axis=1)
        x = LSTM(self.config['rnn_size'])(modelEffOutput)
        outputs = Dense(self.config['num_class'], activation="softmax")(x)

        model = tf.keras.Model(modelEff.input, outputs)
        model.summary()

        return model

    def getEffTransformerModel(self):
        modelEff = self.getEffModel()
        sample_transformer = Transformer(
            num_layers=2, d_model=self.config['rnn_size'], num_heads=4, dff=1024,
            input_vocab_size=1280, target_vocab_size=2048, pe_input=1280)

        enc_input = modelEff.output
        fn_out = sample_transformer(enc_input, training=False, enc_padding_mask=None, look_ahead_mask=None, dec_padding_mask=None)

        fn_reshape_out = tf.reshape(fn_out, (-1, 1280 * 2048))

        outputs = Dense(self.config['num_class'], activation="softmax")(fn_reshape_out)

        model = tf.keras.Model(modelEff.input, outputs)
        model.summary()

        return model


    def train(self):
        handler = DataHandler(self.train_json_path, self.test_json_path, self.data_path)
        # model = self.getEffLSTMModel()
        model = self.getEffTransformerModel()

        # if os.path.exists(self.PREMODELPATH):
        #     print('--load!--:', self.PREMODELPATH)
        #     model.load_weights(self.PREMODELPATH)

        adam = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

        checkpoint = ModelCheckpoint(
            self.subStr + '/src/model/checkpoint/' + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
            monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
        print('save path:', self.subStr + '/src/model/checkpoint/')

        batchSize = 4

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
        model = self.getEffLSTMModel()

        model.load_weights(self.PREMODELPATH)
        loadDict = handler.readJson(handler.test_json_path)
        batchItems = loadDict['annotations']

        batchSize = 2
        print(len(batchItems))
        result = model.predict_generator(handler.dataPredict(batchItems, batchSize), len(batchItems) // batchSize, verbose=1)
        print(result.shape)
        with open(self.subStr + '/result.log', 'w') as rf:
            for item in result:
                rf.write(str(item) + ',\n')
        result = np.argmax(result, axis=1)
        print(result)
        error_count = 0
        error_result = []
        prediction_list = []
        y_original_list = []
        for index, item in enumerate(batchItems):
            status = int(item['status'])
            prediction_list.append(result[index])
            y_original_list.append(status)
            if status != result[index]:
                item = str(index) + ', status:' + str(status) + ', result:' + str(result[index])
                error_result.append(item)
                error_count += 1
        print(error_count)
        error_result.append(error_count)
        with open(self.subStr + '/result_error.log', 'w') as rf:
            for item in error_result:
                rf.write(str(item) + ',\n')
        score = Score()
        precision, recall, f1 = score.sklearnEvaluate(prediction_list, y_original_list)
        print(precision, recall, f1)
        # for index, item in enumerate(batchItems):
        #     item['status'] = str(result[index])
        # with open(self.subStr + '/amap_submission.json', 'w') as wf:
        #     json.dump(loadDict, wf)




















