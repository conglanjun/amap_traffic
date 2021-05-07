from model.EfficientLSTMV2 import EfficientLSTMV2
from model.EffModelPlaidml import EffModelPlaidml

if __name__ == '__main__':
    lstm = EffModelPlaidml()
    print(lstm.getEffModel(n=0))
    # lstm.train(10, type='EffTransformerModel')
