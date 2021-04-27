import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x11 = np.linspace(0, 4, 5)
    y11 = [0.72, 0.6717, 0.72, 0.7217, 0.7017]
    y12 = [0.596, 0.6129, 0.7248, 0.6168, 0.6374]
    y13 = [0.5728, 0.5522, 0.6534, 0.5596, 0.5974]

    x2 = np.linspace(0, 3, 3)

    y21 = [0.745, 0.745, 0.7417, 0.6533, 0.6633]
    y22 = [0.6186, 0.6296, 0.6274, 0.5994, 0.6198]
    y23 = [0.5703, 0.5998, 0.585, 0.5386, 0.5701]

    width = 0.2

    plt.bar(x11-width/2, y13, width)
    plt.bar(x11+width/2, y23, width)

    labels = ['B0', 'B1', 'B2', 'B3', 'B4']

    plt.xticks(x11, labels)

    plt.tight_layout()
    plt.title('fscore', y=-0.1)
    plt.legend(['ELSTM', 'ETF'], loc='lower left')
    plt.figure(figsize=(9, 6))
    plt.show()