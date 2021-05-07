import tensorflow as tf
from model.EfficientLSTMV2 import EfficientLSTMV2


DEVICE = "GPU"

if DEVICE == "GPU":
    print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

if __name__ == '__main__':
    lstm = EfficientLSTMV2()
    # lstm.train(2, type='EffLSTMModel')
    # lstm.train(10, type='EffTransformerModel')
    # lstm.train(33, type='EffTransformerLSTMModel')
    lstm.train(34, type='EffTransformerLSTMModel')
