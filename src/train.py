import tensorflow as tf
from model.EfficientLSTMV2 import EfficientLSTMV2
import os
# git token
# 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

DEVICE = "GPU"

if DEVICE == "GPU":
    print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

if __name__ == '__main__':
    lstm = EfficientLSTMV2()
    # lstm.train(3, type='EffLSTMModel')
    # lstm.train(4, type='EffLSTMModel')
    # lstm.train(5, type='EffLSTMModel')
    # lstm.train(6, type='EffLSTMModel')
    lstm.train(7, type='EffLSTMModel')
    # lstm.train(10, type='EffTransformerModel')
    # lstm.train(11, type='EffTransformerModel')
    # lstm.train(12, type='EffTransformerModel')
    # lstm.train(13, type='EffTransformerModel')
    # lstm.train(14, type='EffTransformerModel')

    # lstm.train(20, type='EffBiLSTMModel')
    # lstm.train(21, type='EffBiLSTMModel')
    # lstm.train(22, type='EffBiLSTMModel')
    # lstm.train(23, type='EffBiLSTMModel')
    # lstm.train(24, type='EffBiLSTMModel')

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
