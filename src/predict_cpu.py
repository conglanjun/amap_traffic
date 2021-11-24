import tensorflow as tf
from model.EfficientLSTMV2 import EfficientLSTMV2
from model.EffModel import EffModel


DEVICE = "GPU"
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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
    lstm.predict(44, type='EffTransformerBLSTMModel', predictPath='ep001-loss0.687-val_loss0.296.h5')
    lstm.predict(44, type='EffTransformerBLSTMModel', predictPath='ep002-loss0.407-val_loss0.198.h5')
    lstm.predict(44, type='EffTransformerBLSTMModel', predictPath='ep003-loss0.310-val_loss0.168.h5')
    lstm.predict(44, type='EffTransformerBLSTMModel', predictPath='ep004-loss0.270-val_loss0.185.h5')
    lstm.predict(44, type='EffTransformerBLSTMModel', predictPath='ep005-loss0.228-val_loss0.150.h5')
    lstm.predict(44, type='EffTransformerBLSTMModel', predictPath='ep006-loss0.186-val_loss0.193.h5')
    lstm.predict(44, type='EffTransformerBLSTMModel', predictPath='ep007-loss0.160-val_loss0.216.h5')
    lstm.predict(44, type='EffTransformerBLSTMModel', predictPath='ep008-loss0.142-val_loss0.193.h5')
    lstm.predict(44, type='EffTransformerBLSTMModel', predictPath='ep009-loss0.123-val_loss0.164.h5')
    lstm.predict(44, type='EffTransformerBLSTMModel', predictPath='ep010-loss0.110-val_loss0.171.h5')
    # lstm.predict(34, type='EffTransformerLSTMModel')
    # lstm.predict(4, type='EffLSTMModel')

