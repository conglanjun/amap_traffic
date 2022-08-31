from model.EffModelDense import EffModelDense
from model.EffBiGruModel import EffBiGruModel
from model.EffGruModel import EffGruModel
from model.EffBiRnnModel import EffBiRnnModel
from model.EffRnnModel import EffRnnModel
from model.EffTransformerLSTMModel import EffTransformerLSTMModel
from model.EffTransformerModel import EffTransformerModel
from model.EffBiLSTMModel import EffBiLSTMModel
from model.EffLSTMModel import EffLSTMModel
from model.EffModel import EffModel
import os
import random
from util.Score import Score
from util.DataHandler import DataHandler
from cmath import log
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
import sys
sys.path.append("..")


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
        self.EffModel = EffModel()
        self.EffLSTMModel = EffLSTMModel()
        self.EffRnnModel = EffRnnModel()
        self.EffBiRnnModel = EffBiRnnModel()
        self.EffGruModel = EffGruModel()
        self.EffBiGruModel = EffBiGruModel()
        self.EffBiLSTMModel = EffBiLSTMModel()
        self.EffTransformerModel = EffTransformerModel()
        self.EffTransformerLSTMModel = EffTransformerLSTMModel()
        self.EffModelDense = EffModelDense()
        realPath = os.path.realpath(__file__)
        subStr = realPath[:realPath[: realPath[: realPath.rindex(
            '/')].rindex('/')].rindex('/')]
        self.subStr = subStr
        # self.train_json_path = subStr + '/data/amap_traffic_annotations_train.json'
        self.train_json_path = subStr + '/data/amap_traffic_augment.json'
        self.test_json_path = subStr + '/data/amap_traffic_annotations_test_answer.json'
        self.data_path = subStr + '/data/amap_traffic_train_0712/'
        self.data_test_path = subStr + '/data/amap_traffic_test_0712/'
        self.PREMODELPATH = subStr + '/src/model/checkpoint/' + "B0/trained_weights_final.h5"

    def train(self, n, type='EffLSTMModel', saveDir='B0'):
        batchSize = 4
        handler = DataHandler(self.train_json_path,
                              self.test_json_path, self.data_path)

        model = ''

        if type == 'EffLSTMModel':
            model = self.EffLSTMModel.getEffLSTMModel(n)
        elif type == 'EffBiLSTMModel':
            model = self.EffBiLSTMModel.getEffBiLSTMModel(n)
        elif type == 'EffRnnModel':
            model = self.EffRnnModel.getEffRnnModel(n)
        elif type == 'EffBiRnnModel':
            model = self.EffBiRnnModel.getEffBiRnnModel(n)
        elif type == 'EffGruModel':
            model = self.EffGruModel.getEffGruModel(n)
        elif type == 'EffBiGruModel':
            model = self.EffBiGruModel.getEffBiGruModel(n)
        elif type == 'EffTransformerModel':
            model = self.EffTransformerModel.getEffTransformerModel(n)
        elif type == 'EffTransformerLSTMModel':
            model = self.EffTransformerLSTMModel.getEffTransformerLSTMModel(n)
        elif type == 'EffTransformerBLSTMModel':
            model = self.EffTransformerLSTMModel.getEffTransformerBLSTMModel(n)
        elif type == 'EffModelDense':
            model = self.EffModelDense.getEffModelDense(n)
        elif type == 'VggModelDense':
            model = self.EffModelDense.getVggModelDense(n)
        elif type == 'ResnetModelDense':
            model = self.EffModelDense.getResnetModelDense(n)

        epochs = 10
        if n == 0:
            saveDir = 'B0'
        elif n == 1:
            saveDir = 'B1'
        elif n == 2:
            saveDir = 'B2'
        elif n == 3:
            saveDir = 'B3'
        elif n == 4:
            saveDir = 'B4'
        elif n == 5:
            saveDir = 'B5'
        elif n == 6:
            saveDir = 'B6'
        elif n == 7:
            saveDir = 'B7'
        elif n == 10:
            saveDir = 'TFB0'
        elif n == 11:
            saveDir = 'TFB1'
        elif n == 12:
            saveDir = 'TFB2'
        elif n == 13:
            saveDir = 'TFB3'
        elif n == 14:
            saveDir = 'TFB4'
        elif n == 20:
            saveDir = 'BB0'
        elif n == 21:
            saveDir = 'BB1'
        elif n == 22:
            saveDir = 'BB2'
        elif n == 23:
            saveDir = 'BB3'
        elif n == 24:
            saveDir = 'BB4'
        elif n == 25:
            saveDir = 'BB5'
        elif n == 26:
            saveDir = 'BB6'
        elif n == 27:
            saveDir = 'BB7'
        elif n == 30:
            saveDir = 'TFLB0'
        elif n == 31:
            saveDir = 'TFLB1'
        elif n == 32:
            saveDir = 'TFLB2'
        elif n == 33:
            saveDir = 'TFLB3'
        elif n == 34:
            saveDir = 'TFLB4'
        elif n == 40:
            saveDir = 'EFTB0'
        elif n == 41:
            saveDir = 'EFTB1'
        elif n == 42:
            saveDir = 'EFTB2'
        elif n == 43:
            saveDir = 'EFTB3'
        elif n == 44:
            saveDir = 'EFTB4'
        elif n == 50:
            saveDir = 'ERB0'
        elif n == 51:
            saveDir = 'ERB1'
        elif n == 52:
            saveDir = 'ERB2'
        elif n == 53:
            saveDir = 'ERB3'
        elif n == 54:
            saveDir = 'ERB4'
        elif n == 55:
            saveDir = 'ERB5'
        elif n == 56:
            saveDir = 'ERB6'
        elif n == 57:
            saveDir = 'ERB7'
        elif n == 60:
            saveDir = 'EGB0'
        elif n == 61:
            saveDir = 'EGB1'
        elif n == 62:
            saveDir = 'EGB2'
        elif n == 63:
            saveDir = 'EGB3'
        elif n == 64:
            saveDir = 'EGB4'
        elif n == 65:
            saveDir = 'EGB5'
        elif n == 66:
            saveDir = 'EGB6'
        elif n == 67:
            saveDir = 'EGB7'
        elif n == 70:
            saveDir = 'EBRB0'
        elif n == 71:
            saveDir = 'EBRB1'
        elif n == 72:
            saveDir = 'EBRB2'
        elif n == 73:
            saveDir = 'EBRB3'
        elif n == 74:
            saveDir = 'EBRB4'
        elif n == 75:
            saveDir = 'EBRB5'
        elif n == 76:
            saveDir = 'EBRB6'
        elif n == 77:
            saveDir = 'EBRB7'
        elif n == 80:
            saveDir = 'EBGB0'
        elif n == 81:
            saveDir = 'EBGB1'
        elif n == 82:
            saveDir = 'EBGB2'
        elif n == 83:
            saveDir = 'EBGB3'
        elif n == 84:
            saveDir = 'EBGB4'
        elif n == 85:
            saveDir = 'EBGB5'
        elif n == 86:
            saveDir = 'EBGB6'
        elif n == 87:
            saveDir = 'EBGB7'
        elif n == 90:
            saveDir = 'EMDB0'
        elif n == 91:
            saveDir = 'EMDB1'
        elif n == 92:
            saveDir = 'EMDB2'
        elif n == 93:
            saveDir = 'EMDB3'
        elif n == 94:
            saveDir = 'EMDB4'
        elif n == 95:
            saveDir = 'EMDB5'
        elif n == 96:
            saveDir = 'EMDB6'
        elif n == 97:
            saveDir = 'EMDB7'
        elif n == 100:
            saveDir = 'VGG'
        elif n == 101:
            saveDir = 'Resnet'

        self.PREMODELPATH = self.subStr + '/src/model/checkpoint/' + \
            saveDir + "/trained_weights_final.h5"
        print(self.PREMODELPATH)

        if os.path.exists(self.PREMODELPATH):
            print('--load!--:', self.PREMODELPATH)
            model.load_weights(self.PREMODELPATH)

        adam = tf.keras.optimizers.Adam(
            lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(optimizer=adam,
                      loss='categorical_crossentropy', metrics=['accuracy'])

        checkpoint = ModelCheckpoint(
            self.subStr + '/src/model/checkpoint/' + saveDir +
            '/ep_{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
            monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
        print('save path:', self.subStr + '/src/model/checkpoint/')

        if not os.path.exists(self.subStr+'/src/model/checkpoint/' + saveDir):
            print('--mkdir--:', self.subStr+'/src/model/checkpoint/' + saveDir)
            os.mkdir(self.subStr+'/src/model/checkpoint/' + saveDir)

        if not os.path.exists('./logs/'+saveDir):
            print('--mkdir--:', './logs/'+saveDir)
            os.mkdir('./logs/'+saveDir)
        tbCheckpoint = TensorBoard(log_dir='./logs/'+saveDir, histogram_freq=1)
        # summary_writer = tf.summary.create_file_writer('./logs')
        # with summary_writer .as_default():
        #     tf.summary.scalar('loss',float(2),step=0)
        #     tf.summary.scalar('acc', float(80), step=0)

        loadDict = handler.readJson(handler.train_json_path)
        batchItems = loadDict['annotations']
        random.shuffle(batchItems)
        numValidation = len(batchItems) // 10
        numTrain = len(batchItems) - numValidation
        # numTrain = 8

        trainData = batchItems[: numTrain]
        valiData = batchItems[numTrain:]

        model.fit_generator(handler.dataGenerator(trainData, batchSize, self.config['num_class']),
                            steps_per_epoch=max(1, numTrain // batchSize),
                            # steps_per_epoch=max(1, numTrain),
                            validation_data=handler.dataGenerator(
                                valiData, batchSize, self.config['num_class']),
                            validation_steps=max(
                                1, numValidation // batchSize),
                            # validation_steps=max(1, numValidation),
                            epochs=epochs,
                            initial_epoch=0,
                            callbacks=[checkpoint, tbCheckpoint])
        model.save_weights(self.subStr + '/src/model/checkpoint/' +
                           saveDir + '/trained_weights_final.h5')

    def predict(self, n, type='EffLSTMModel', predictPath='', saveDir='B0'):
        batchSize = 4
        handler = DataHandler(self.train_json_path,
                              self.test_json_path, self.data_test_path)

        model = ''

        if type == 'EffLSTMModel':
            model = self.EffLSTMModel.getEffLSTMModel(n)
        elif type == 'EffBiLSTMModel':
            model = self.EffBiLSTMModel.getEffBiLSTMModel(n)
        elif type == 'EffRnnModel':
            model = self.EffRnnModel.getEffRnnModel(n)
        elif type == 'EffBiRnnModel':
            model = self.EffBiRnnModel.getEffBiRnnModel(n)
        elif type == 'EffGruModel':
            model = self.EffGruModel.getEffGruModel(n)
        elif type == 'EffBiGruModel':
            model = self.EffBiGruModel.getEffBiGruModel(n)
        elif type == 'EffTransformerModel':
            model = self.EffTransformerModel.getEffTransformerModel(n)
        elif type == 'EffTransformerLSTMModel':
            model = self.EffTransformerLSTMModel.getEffTransformerLSTMModel(n)
        elif type == 'EffTransformerBLSTMModel':
            model = self.EffTransformerLSTMModel.getEffTransformerBLSTMModel(n)
        elif type == 'EffModelDense':
            model = self.EffModelDense.getEffModelDense(n)
        elif type == 'VggModelDense':
            model = self.EffModelDense.getVggModelDense(n)
        elif type == 'ResnetModelDense':
            model = self.EffModelDense.getResnetModelDense(n)

        if n == 0:
            saveDir = 'B0'
        elif n == 1:
            saveDir = 'B1'
        elif n == 2:
            saveDir = 'B2'
        elif n == 3:
            saveDir = 'B3'
        elif n == 4:
            saveDir = 'B4'
        elif n == 5:
            saveDir = 'B5'
        elif n == 6:
            saveDir = 'B6'
        elif n == 7:
            saveDir = 'B7'
        elif n == 10:
            saveDir = 'TFB0'
        elif n == 11:
            saveDir = 'TFB1'
        elif n == 12:
            saveDir = 'TFB2'
        elif n == 13:
            saveDir = 'TFB3'
        elif n == 14:
            saveDir = 'TFB4'
        elif n == 20:
            saveDir = 'BB0'
        elif n == 21:
            saveDir = 'BB1'
        elif n == 22:
            saveDir = 'BB2'
        elif n == 23:
            saveDir = 'BB3'
        elif n == 24:
            saveDir = 'BB4'
        elif n == 25:
            saveDir = 'BB5'
        elif n == 26:
            saveDir = 'BB6'
        elif n == 27:
            saveDir = 'BB7'
        elif n == 30:
            saveDir = 'TFLB0'
        elif n == 31:
            saveDir = 'TFLB1'
        elif n == 32:
            saveDir = 'TFLB2'
        elif n == 33:
            saveDir = 'TFLB3'
        elif n == 34:
            saveDir = 'TFLB4'
        elif n == 40:
            saveDir = 'EFTB0'
        elif n == 41:
            saveDir = 'EFTB1'
        elif n == 42:
            saveDir = 'EFTB2'
        elif n == 43:
            saveDir = 'EFTB3'
        elif n == 44:
            saveDir = 'EFTB4'
        elif n == 50:
            saveDir = 'ERB0'
        elif n == 51:
            saveDir = 'ERB1'
        elif n == 52:
            saveDir = 'ERB2'
        elif n == 53:
            saveDir = 'ERB3'
        elif n == 54:
            saveDir = 'ERB4'
        elif n == 55:
            saveDir = 'ERB5'
        elif n == 56:
            saveDir = 'ERB6'
        elif n == 57:
            saveDir = 'ERB7'
        elif n == 60:
            saveDir = 'EGB0'
        elif n == 61:
            saveDir = 'EGB1'
        elif n == 62:
            saveDir = 'EGB2'
        elif n == 63:
            saveDir = 'EGB3'
        elif n == 64:
            saveDir = 'EGB4'
        elif n == 65:
            saveDir = 'EGB5'
        elif n == 66:
            saveDir = 'EGB6'
        elif n == 67:
            saveDir = 'EGB7'
        elif n == 70:
            saveDir = 'EBRB0'
        elif n == 71:
            saveDir = 'EBRB1'
        elif n == 72:
            saveDir = 'EBRB2'
        elif n == 73:
            saveDir = 'EBRB3'
        elif n == 74:
            saveDir = 'EBRB4'
        elif n == 75:
            saveDir = 'EBRB5'
        elif n == 76:
            saveDir = 'EBRB6'
        elif n == 77:
            saveDir = 'EBRB7'
        elif n == 80:
            saveDir = 'EBGB0'
        elif n == 81:
            saveDir = 'EBGB1'
        elif n == 82:
            saveDir = 'EBGB2'
        elif n == 83:
            saveDir = 'EBGB3'
        elif n == 84:
            saveDir = 'EBGB4'
        elif n == 85:
            saveDir = 'EBGB5'
        elif n == 86:
            saveDir = 'EBGB6'
        elif n == 87:
            saveDir = 'EBGB7'
        elif n == 90:
            saveDir = 'EMDB0'
        elif n == 91:
            saveDir = 'EMDB1'
        elif n == 92:
            saveDir = 'EMDB2'
        elif n == 93:
            saveDir = 'EMDB3'
        elif n == 94:
            saveDir = 'EMDB4'
        elif n == 95:
            saveDir = 'EMDB5'
        elif n == 96:
            saveDir = 'EMDB6'
        elif n == 97:
            saveDir = 'EMDB7'
        elif n == 100:
            saveDir = 'VGG'
        elif n == 101:
            saveDir = 'Resnet'

        path_load = self.subStr + '/src/model/checkpoint/' + \
            saveDir + "/trained_weights_final.h5"
        path_load = self.subStr + '/src/model/checkpoint/' + saveDir + "/" + predictPath

        print(path_load)
        model.load_weights(path_load)

        # f = h5py.File(self.PREMODELPATH)
        # for key in f.keys():
        #     print(key)
        # print(f['model_weights'].attrs.keys())

        loadDict = handler.readJson(handler.test_json_path)
        batchItems = loadDict['annotations']

        # label0 = 0
        # label1 = 0
        # label2 = 0
        # for index, item in enumerate(batchItems):
        #     status = int(item['status'])
        #     if status == 0:
        #         label0 += 1
        #     elif status == 1:
        #         label1 += 1
        #     elif status == 2:
        #         label2 += 1
        # print('label0,1,2:', label0, label1, label2)

        print(len(batchItems))
        result = model.predict_generator(handler.dataPredict(
            batchItems, batchSize), len(batchItems) // batchSize, verbose=1)
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
                item = str(index) + ', status:' + str(status) + \
                    ', result:' + str(result[index])
                error_result.append(item)
                error_count += 1
        print(error_count, "%.4f" % ((600 - error_count) / 600.0))
        error_result.append(error_count)
        with open(self.subStr + '/result_error.log', 'w') as rf:
            for item in error_result:
                rf.write(str(item) + ',\n')
        score = Score()
        precision, recall, f1 = score.sklearnEvaluate(
            prediction_list, y_original_list)
        print("precision", precision, "%.4f" %
              (precision[0] * 0.33 + precision[1] * 0.33 + precision[2] * 0.33))
        print("recall", recall, "%.4f" %
              (recall[0] * 0.33 + recall[1] * 0.33 + recall[2] * 0.33))
        print("f1", f1, "%.4f" %
              (f1[0] * 0.33 + f1[1] * 0.3333 + f1[2] * 0.33))
        # for index, item in enumerate(batchItems):
        #     item['status'] = str(result[index])
        # with open(self.subStr + '/amap_submission.json', 'w') as wf:
        #     json.dump(loadDict, wf)


# label0,1,2: 402 97 101                0.12 0.44 0.44
# model.layers[20].trainable_variables[0].numpy
# array([[ 0.036309  ,  0.03461195, -0.02674555, ...,  0.04236516,
#          0.04607766, -0.00878192],
#        [ 0.04962379,  0.01646587, -0.03508455, ...,  0.04515529,
#         -0.04109945,  0.08064999],
#        [ 0.00673465, -0.04701834,  0.01092691, ..., -0.03731103,
#         -0.00893005, -0.01128179],
#        ...,
#        [ 0.03907353,  0.01030799, -0.03931922, ..., -0.02314625,
#         -0.02106541, -0.01232917],
#        [-0.00598556, -0.02257906,  0.02021596, ..., -0.00395703,
#          0.03175794,  0.04028033],
#        [ 0.01436837,  0.03238466, -0.0258049 , ...,  0.03459903,
#         -0.04169638,  0.00162689]], dtype=float32)>>

# array([[ 0.03320085,  0.0335603 , -0.02587961, ...,  0.04870539,
#          0.04522996, -0.00530622],
#        [ 0.05220624,  0.01408922, -0.03600347, ...,  0.04476301,
#         -0.03975284,  0.08197557],
#        [ 0.00673465, -0.04701834,  0.01092691, ..., -0.03731103,
#         -0.00893005, -0.01128179],
#        ...,
#        [ 0.03907353,  0.01030799, -0.03931922, ..., -0.02314625,
#         -0.02106541, -0.01232917],
#        [-0.00598556, -0.02257906,  0.02021596, ..., -0.00395703,
#          0.03175794,  0.04028033],
#        [ 0.01436837,  0.03238466, -0.0258049 , ...,  0.03459903,
#         -0.04169638,  0.00162689]], dtype=float32)>>
