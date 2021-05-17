import tensorflow as tf
from model.EfficientLSTMV2 import EfficientLSTMV2
from model.EffModel import EffModel


DEVICE = "GPU"
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if DEVICE == "GPU":
    print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

if __name__ == '__main__':
    # lstm = EfficientLSTM()
    # lstm.train()
    # eff = EffModel()
    # eff.getEffTFModel(n=0)
    lstm = EfficientLSTMV2()
    # lstm.predict(24, type='EffBiLSTMModel')
    # lstm.predict(14, type='EffTransformerModel')
    lstm.predict(44, type='EffTransformerBLSTMModel', predictPath='ep011-loss0.077-val_loss0.044.h5')
    lstm.predict(44, type='EffTransformerBLSTMModel', predictPath='ep012-loss0.061-val_loss0.011.h5')
    lstm.predict(44, type='EffTransformerBLSTMModel', predictPath='ep013-loss0.044-val_loss0.062.h5')
    lstm.predict(44, type='EffTransformerBLSTMModel', predictPath='ep014-loss0.038-val_loss0.059.h5')
    lstm.predict(44, type='EffTransformerBLSTMModel', predictPath='ep015-loss0.035-val_loss0.016.h5')
    lstm.predict(44, type='EffTransformerBLSTMModel', predictPath='ep016-loss0.032-val_loss0.049.h5')
    lstm.predict(44, type='EffTransformerBLSTMModel', predictPath='ep017-loss0.026-val_loss0.075.h5')
    lstm.predict(44, type='EffTransformerBLSTMModel', predictPath='ep018-loss0.028-val_loss0.043.h5')
    lstm.predict(44, type='EffTransformerBLSTMModel', predictPath='ep019-loss0.016-val_loss0.048.h5')
    lstm.predict(44, type='EffTransformerBLSTMModel', predictPath='ep020-loss0.017-val_loss0.041.h5')
    # lstm.predict(34, type='EffTransformerLSTMModel')
    # lstm.predict(4, type='EffLSTMModel')

