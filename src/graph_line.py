import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x11 = np.linspace(0, 7, 8)
    # f1 score
    yB = [0.5972, 0.6128, 0.6326, 0.6345, 0.62301, 0.6219, 0.6556, 0.6014]
    yBB = [0.5972, 0.6115, 0.6118, 0.5810, 0.6093, 0.6082, 0.6441, 0.6370]
    yRB = [0.6191, 0.5812, 0.6305, 0.5786, 0.6077, 0.5762, 0.6123, 0.6172]
    yGB = [0.6031, 0.5531, 0.6310, 0.6120, 0.6112, 0.6191, 0.6191, 0.6191]
    yBRB = [0.6184, 0.6016, 0.5862, 0.5882, 0.5589, 0.5547, 0.5849, 0.5930]
    yBGB = [0.5877, 0.5868, 0.6608, 0.6307, 0.6847, 0.6415, 0.6557, 0.6381]

    # precision
    # yB = [0.6079, 0.6122, 0.6078, 0.6222, 0.6004, 0.6079, 0.6382, 0.5899]
    # yBB = [0.5970, 0.5977, 0.5962, 0.5902, 0.5971, 0.5927, 0.6288, 0.6259]
    # yRB = [0.6143, 0.5740, 0.6140, 0.5653, 0.5938, 0.5663, 0.6137, 0.6053]
    # yGB = [0.5943, 0.5532, 0.6162, 0.5964, 0.5979, 0.6024, 0.5669, 0.5876]
    # yBRB = [0.6005, 0.5882, 0.5707, 0.5734, 0.5564, 0.5463, 0.5730, 0.6033]
    # yBGB = [0.5748, 0.5781, 0.6542, 0.6325, 0.6666, 0.6440, 0.6355, 0.6442]

    # recall
    # yB = [0.6531, 0.6492, 0.6677, 0.6672, 0.6584, 0.6620, 0.6805, 0.6502]
    # yBB = [0.6415, 0.6475, 0.6578, 0.6364, 0.6638, 0.6386, 0.6643, 0.6950]
    # yRB = [0.6647, 0.6446, 0.6881, 0.6176, 0.6474, 0.6117, 0.6285, 0.6645]
    # yGB = [0.6437, 0.6127, 0.6587, 0.6628, 0.6445, 0.6569, 0.6304, 0.6500]
    # yBRB = [0.6579, 0.6233, 0.6240, 0.6324, 0.6277, 0.5948, 0.6229, 0.6305]
    # yBGB = [0.6308, 0.6303, 0.6936, 0.6711, 0.7162, 0.6814, 0.7012, 0.6964]

    # x2 = np.linspace(0, 3, 3)

    # width = 0.2

    # plt.bar(x11-width/2, y13, width)
    # plt.bar(x11+width/2, y23, width)

    plt.ylim(0.5, 0.75)

    # 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd','P', 'X'

    plt.plot(x11, yB, color='orange', marker='*', label='ELSTM')
    plt.plot(x11, yBB, color='blue', marker='o', label='EBLSTM')
    plt.plot(x11, yRB, color='red', marker='^', label='ERNN')
    plt.plot(x11, yGB, color='green', marker='v', label='EGRU')
    plt.plot(x11, yBRB, color='cyan', marker='+', label='EBRNN')
    plt.plot(x11, yBGB, color='magenta', marker='3', label='EBGRU')

    labels = ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']

    plt.xticks(x11, labels)

    plt.grid()

    # plt.yticks([0.70, 0.75, 0.80, 0.85])

    plt.xlabel('efficient net model')  # X轴标签

    plt.tight_layout()
    # plt.title('precision', y=-0.15)
    # plt.title('recall', y=-0.15)
    plt.title('f1 score', y=-0.15)
    plt.legend(['ELSTM', 'EBLSTM', 'ERNN', 'EGRU', 'EBRNN', 'EBGRU'], loc='upper left')
    plt.figure(figsize=(9, 6))
    plt.show()