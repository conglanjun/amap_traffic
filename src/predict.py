import tensorflow as tf
from model.EfficientLSTMV2 import EfficientLSTMV2


DEVICE = "GPU"
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if DEVICE == "GPU":
    print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

if __name__ == '__main__':
    # lstm = EfficientLSTM()
    # lstm.train()
    lstm = EfficientLSTMV2()
    # lstm.predict(21, type='EffBiLSTMModel')
    # lstm.predict(13, type='EffTransformerModel')
    # lstm.predict(33, type='EffTransformerLSTMModel')
    lstm.predict(34, type='EffTransformerLSTMModel')
