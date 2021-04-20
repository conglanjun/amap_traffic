import tensorflow as tf
from model.EfficientLSTM import EfficientLSTM


DEVICE = "GPU"

if DEVICE == "GPU":
    print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

if __name__ == '__main__':
    lstm = EfficientLSTM()
    lstm.predict()
