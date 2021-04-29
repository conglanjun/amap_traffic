import tensorflow as tf
from model.EfficientLSTMV2 import EfficientLSTMV2


DEVICE = "GPU"

if DEVICE == "GPU":
    print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

if __name__ == '__main__':
    # lstm = EfficientLSTM()
    # lstm.train()
    lstm = EfficientLSTMV2()
    # lstm.train(1)
    # lstm.train(2)
    lstm.train(3)
    lstm.train(4)
