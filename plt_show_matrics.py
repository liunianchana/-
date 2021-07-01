import matplotlib.pyplot as plt


def PSM(x,y):#第一个参数，混淆矩阵，
    # 热度图，后面是指定的颜色块，gray也可以，gray_x反色也可以
    plt.imshow(x,  interpolation='nearest',cmap=plt.cm.Greens)
    for i in range(len(x)):  # 数据标签
        for j in range(len(x)):
            plt.annotate(x[i, j], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.colorbar()
    plt.tight_layout()
    a = 'D:/result/'
    plt.savefig(a+y+'_matrix.jpg')
    return












