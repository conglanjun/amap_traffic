import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x11 = np.linspace(1, 10, 10)
    print(x11)
    # f1 score
    # B0 = [0.5972, 0.6128, 0.7, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    # B1 = [0.5972, 0.6128, 0.7, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    # B2 = [0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    # B3 = [0.5972, 0.6128, 0.7, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    # B4 = [0.5, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9]
    # B5 = [0.5972, 0.6128, 0.7, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    # B6 = [0.5972, 0.6128, 0.7, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    # B7 = [0.5972, 0.6128, 0.7, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    train = [0.7232, 0.8752, 0.9412, 0.9711, 0.979,
             0.9891, 0.9918, 0.9858, 0.9892, 0.9912]
    validation = [0.7533, 0.884, 0.932, 0.9521,
                  0.9676, 0.9632, 0.9721, 0.971, 0.9701, 0.97]

    # x2 = np.linspace(0, 3, 3)

    # width = 0.2

    # plt.bar(x11-width/2, y13, width)
    # plt.bar(x11+width/2, y23, width)

    plt.ylim(0.5, 0.75)

    # 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd','P', 'X'

    plt.plot(x11, train, color='red', marker='o', label='train')
    plt.plot(x11, validation, color='blue', marker='D', label='validation')
    # plt.plot(x11, B1, color='blue', marker='o', label='B1')
    # plt.plot(x11, B2, color='red', marker='^', label='B2')
    # plt.plot(x11, B3, color='green', marker='v', label='B3')
    # plt.plot(x11, B4, color='cyan', marker='+', label='B4')
    # plt.plot(x11, B5, color='magenta', marker='3', label='B5')
    # plt.plot(x11, B5, color='magenta', marker='p', label='B6')
    # plt.plot(x11, B5, color='magenta', marker='h', label='B7')

    # labels = ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']

    plt.xticks(x11, ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])

    plt.grid()

    plt.yticks([0.7, 0.8, 0.9, 1.0, 1.1])

    plt.ylabel('train accuracy')  # y轴标签
    plt.xlabel('epoch')  # x轴标签

    # plt.tight_layout()
    # plt.title('precision', y=-0.15)
    # plt.title('recall', y=-0.15)
    # plt.title('f1 score', y=-0.15)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.figure(figsize=(9, 6))
    plt.show()
