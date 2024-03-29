import tensorflow as tf
from model.EfficientLSTMV2 import EfficientLSTMV2
import os
# git token
# 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

DEVICE = "GPU"

if DEVICE == "GPU":
    print("Num GPUs Available:", len(
        tf.config.experimental.list_physical_devices('GPU')))

if __name__ == '__main__':
    lstm = EfficientLSTMV2()
    # lstm.train(0, type='EffLSTMModel', saveDir='B0')
    # lstm.train(1, type='EffLSTMModel', saveDir='B1')
    # lstm.train(2, type='EffLSTMModel', saveDir='B2')
    # lstm.train(3, type='EffLSTMModel', saveDir='B3')
    # lstm.train(4, type='EffLSTMModel', saveDir='B4')
    # lstm.train(5, type='EffLSTMModel', saveDir='B5')
    # lstm.train(6, type='EffLSTMModel', saveDir='B6')
    # lstm.train(7, type='EffLSTMModel', saveDir='B7')
    # lstm.train(10, type='EffTransformerModel')
    # lstm.train(11, type='EffTransformerModel')
    # lstm.train(12, type='EffTransformerModel')
    # lstm.train(13, type='EffTransformerModel')
    # lstm.train(14, type='EffTransformerModel')

    # lstm.train(20, type='EffBiLSTMModel', saveDir='BB0')
    # lstm.train(21, type='EffBiLSTMModel', saveDir='BB1')
    # lstm.train(22, type='EffBiLSTMModel', saveDir='BB2')
    # lstm.train(23, type='EffBiLSTMModel', saveDir='BB3')
    # lstm.train(24, type='EffBiLSTMModel', saveDir='BB4')
    # lstm.train(25, type='EffBiLSTMModel', saveDir='BB5')
    # lstm.train(26, type='EffBiLSTMModel', saveDir='BB6')
    # lstm.train(27, type='EffBiLSTMModel', saveDir='BB7')

    # lstm.train(30, type='EffTransformerLSTMModel')
    # lstm.train(31, type='EffTransformerLSTMModel')
    # lstm.train(32, type='EffTransformerLSTMModel')
    # lstm.train(33, type='EffTransformerLSTMModel')
    # lstm.train(34, type='EffTransformerLSTMModel')

    # lstm.train(40, type='EffTransformerBLSTMModel')
    # lstm.train(41, type='EffTransformerBLSTMModel')
    # lstm.train(42, type='EffTransformerBLSTMModel')
    # lstm.train(43, type='EffTransformerBLSTMModel')
    # lstm.train(44, type='EffTransformerBLSTMModel')

    # lstm.train(50, type='EffRnnModel', saveDir='ERB0')
    # lstm.train(51, type='EffRnnModel', saveDir='ERB1')
    # lstm.train(52, type='EffRnnModel', saveDir='ERB2')
    # lstm.train(53, type='EffRnnModel', saveDir='ERB3')
    # lstm.train(54, type='EffRnnModel', saveDir='ERB4')
    # lstm.train(55, type='EffRnnModel', saveDir='ERB5')
    # lstm.train(56, type='EffRnnModel', saveDir='ERB6')
    # lstm.train(57, type='EffRnnModel', saveDir='ERB7')

    # lstm.train(61, type='EffGruModel', saveDir='EGB0')
    # lstm.train(61, type='EffGruModel', saveDir='EGB1')
    # lstm.train(62, type='EffGruModel', saveDir='EGB2')
    # lstm.train(63, type='EffGruModel', saveDir='EGB3')
    # lstm.train(64, type='EffGruModel', saveDir='EGB4')
    # lstm.train(65, type='EffGruModel', saveDir='EGB5')
    # lstm.train(66, type='EffGruModel', saveDir='EGB6')
    # lstm.train(67, type='EffGruModel', saveDir='EGB7')

    # lstm.train(70, type='EffBiRnnModel', saveDir='EBRB0')
    # lstm.train(71, type='EffBiRnnModel', saveDir='EBRB1')
    # lstm.train(72, type='EffBiRnnModel', saveDir='EBRB2')
    # lstm.train(73, type='EffBiRnnModel', saveDir='EBRB3')
    # lstm.train(74, type='EffBiRnnModel', saveDir='EBRB4')
    # lstm.train(75, type='EffBiRnnModel', saveDir='EBRB5')
    # lstm.train(76, type='EffBiRnnModel', saveDir='EBRB6')
    # lstm.train(77, type='EffBiRnnModel', saveDir='EBRB7')

    # lstm.train(80, type='EffBiGruModel', saveDir='EBGB0')
    # lstm.train(81, type='EffBiGruModel', saveDir='EBGB1')
    # lstm.train(82, type='EffBiGruModel', saveDir='EBGB2')
    # lstm.train(83, type='EffBiGruModel', saveDir='EBGB3')
    # lstm.train(84, type='EffBiGruModel', saveDir='EBGB4')
    # lstm.train(85, type='EffBiGruModel', saveDir='EBGB5')
    # lstm.train(86, type='EffBiGruModel', saveDir='EBGB6')
    # lstm.train(87, type='EffBiGruModel', saveDir='EBGB7')

    # lstm.train(90, type='EffModelDense', saveDir='EMDB0')
    # lstm.train(91, type='EffModelDense', saveDir='EMDB1')
    # lstm.train(92, type='EffModelDense', saveDir='EMDB2')
    # lstm.train(93, type='EffModelDense', saveDir='EMDB3')
    # lstm.train(94, type='EffModelDense', saveDir='EMDB4')
    # lstm.train(95, type='EffModelDense', saveDir='EMDB5')
    # lstm.train(96, type='EffModelDense', saveDir='EMDB6')
    # lstm.train(97, type='EffModelDense', saveDir='EMDB7')

    lstm.train(100, type='VggModelDense', saveDir='VGG')
    lstm.train(101, type='ResnetModelDense', saveDir='Resnet')
